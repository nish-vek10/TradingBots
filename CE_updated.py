import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

# === CONNECT TO MT5 === #
if not mt5.initialize():
    print("MT5 initialization failed:", mt5.last_error())
    quit()

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"MT5 Account Connected! \nAccount: {account.login}, \nBalance: {account.balance}\n") # Check Account ID and Balance

# === FETCH 5-MIN XAUUSD CANDLES === #
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M5
num_candles = 500  # Fetch last 500 candles

bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
if bars is None:
    print("Failed to fetch bars:", mt5.last_error())
    mt5.shutdown()
    quit()

# === CONVERT TO DATAFRAME === #
df = pd.DataFrame(bars)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# === CALCULATE HEIKIN-ASHI CANDLES === #
def calculate_heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_df['ha_close'].iloc[i - 1]) / 2)

    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = df[['high', 'low', 'close']].max(axis=1)
    ha_df['ha_low'] = df[['high', 'low', 'close']].min(axis=1)

    return ha_df

ha_df = calculate_heikin_ashi(df)
print("======== LAST 10 HEIKIN ASHI CANDLES ========")
print(ha_df.tail(10))  # Preview last 10 Heikin Ashi candles

# === ATR PARAMETERS === #
atr_period = 1
atr_mult = 1.85

# === CALCULATE ATR USING HEIKIN-ASHI CANDLES === #
tr = pd.DataFrame(index=ha_df.index)
tr['ha_h'] = ha_df['ha_high']
tr['ha_l'] = ha_df['ha_low']
tr['ha_c'] = ha_df['ha_close']

tr['prev_ha_c'] = tr['ha_c'].shift(1)
tr['true_range'] = tr[['ha_h', 'ha_l', 'prev_ha_c']].apply(
    lambda row: max(
        row['ha_h'] - row['ha_l'],
        abs(row['ha_h'] - row['prev_ha_c']),
        abs(row['ha_l'] - row['prev_ha_c'])
    ), axis=1
)
tr['atr'] = tr['true_range'].rolling(window=atr_period).mean()

# === CHANDELIER STOPS === #
long_stop = tr['ha_h'].rolling(window=atr_period).max() - (tr['atr'] * atr_mult)
short_stop = tr['ha_l'].rolling(window=atr_period).min() + (tr['atr'] * atr_mult)

# === SMOOTHING STOPS TO PREVENT FLIP-FLOPPING === #
long_stop_smooth = long_stop.copy()
short_stop_smooth = short_stop.copy()
for i in range(1, len(tr)):
    if tr['ha_c'].iloc[i - 1] > long_stop_smooth.iloc[i - 1]:
        long_stop_smooth.iloc[i] = max(long_stop.iloc[i], long_stop_smooth.iloc[i - 1])
    else:
        long_stop_smooth.iloc[i] = long_stop.iloc[i]

    if tr['ha_c'].iloc[i - 1] < short_stop_smooth.iloc[i - 1]:
        short_stop_smooth.iloc[i] = min(short_stop.iloc[i], short_stop_smooth.iloc[i - 1])
    else:
        short_stop_smooth.iloc[i] = short_stop.iloc[i]

# === DIRECTION LOGIC === #
dir = [1]  # Start assuming long
for i in range(1, len(tr)):
    if tr['ha_c'].iloc[i] > short_stop_smooth.iloc[i - 1]:
        dir.append(1)
    elif tr['ha_c'].iloc[i] < long_stop_smooth.iloc[i - 1]:
        dir.append(-1)
    else:
        dir.append(dir[-1])

tr['dir'] = dir
tr['dir_prev'] = tr['dir'].shift(1)
tr['buy_signal'] = (tr['dir'] == 1) & (tr['dir_prev'] == -1)
tr['sell_signal'] = (tr['dir'] == -1) & (tr['dir_prev'] == 1)

print("\n======== LAST 20 SIGNAL CANDLES ========")
debug_cols = ['ha_c', 'dir_prev', 'dir', 'buy_signal', 'sell_signal']
print(tr[debug_cols].tail(50))
print(f"\nLast evaluated candle: {tr.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")

# === PRINT CONFIRMATION SIGNALS === #
recent = tr.tail(50)

last_buy = recent[recent['buy_signal']]
last_sell = recent[recent['sell_signal']]

if not last_buy.empty and not last_sell.empty:
    if last_buy.index[-1] > last_sell.index[-1]:
        signal_time = last_buy.index[-1]
        print(f"\n⬆️ Buy signal in recent bars at {signal_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        signal_time = last_sell.index[-1]
        print(f"\n🔻 Sell signal in recent bars at {signal_time.strftime('%Y-%m-%d %H:%M:%S')}")
elif not last_buy.empty:
    signal_time = last_buy.index[-1]
    print(f"\n⬆️ Buy signal in recent bars at {signal_time.strftime('%Y-%m-%d %H:%M:%S')}")
elif not last_sell.empty:
    signal_time = last_sell.index[-1]
    print(f"\n🔻 Sell signal in recent bars at {signal_time.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("\nNo signal in recent bars")

symbol = "XAUUSD"
lot_size = 0.1
slippage = 5

def get_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return None
    return positions[0]  # Return the first position (assuming only one)

def send_order(symbol, action_type, lot=lot_size):
    price = mt5.symbol_info_tick(symbol).ask if action_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    deviation = slippage
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": price,
        "deviation": deviation,
        "magic": 123456,  # Unique ID
        "comment": "ChandelierEntryBot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed: {result.retcode}, {result.comment}")
    else:
        print(f"✅ Order placed: {result}")

# === TRADING DECISION BASED ON SIGNAL ===
position = get_position(symbol)

if not last_buy.empty and not last_sell.empty:
    if last_buy.index[-1] > last_sell.index[-1]:
        signal = "buy"
        signal_time = last_buy.index[-1]
    else:
        signal = "sell"
        signal_time = last_sell.index[-1]
elif not last_buy.empty:
    signal = "buy"
    signal_time = last_buy.index[-1]
elif not last_sell.empty:
    signal = "sell"
    signal_time = last_sell.index[-1]
else:
    signal = None

if signal:
    print(f"\n[^]Signal detected: {signal.upper()} at {signal_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if signal == "buy":
        if position is None:
            send_order(symbol, mt5.ORDER_TYPE_BUY, lot_size)
        elif position.type == mt5.ORDER_TYPE_SELL:
            print("Closing SELL to open BUY...")
            # Add closing logic here if needed
    elif signal == "sell":
        if position is None:
            send_order(symbol, mt5.ORDER_TYPE_SELL, lot_size)
        elif position.type == mt5.ORDER_TYPE_BUY:
            print("Closing BUY to open SELL...")
            # Add closing logic here if needed
else:
    print("\nNo New Actionable Signal.")

def close_position(position):
    if position is None:
        return

    action_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if action_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": action_type,
        "position": position.ticket,
        "price": price,
        "deviation": slippage,
        "magic": 123456,
        "comment": "Close opposite position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(close_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Failed to close position: {result.retcode}, {result.comment}")
    else:
        print(f"✅ Position closed: {result}")

print("\n🔁 Starting Live Trading Loop...\n")

while True:
    # Fetch fresh candles
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if bars is None:
        print("Failed to fetch bars:", mt5.last_error())
        time.sleep(60)
        continue

    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    ha_df = calculate_heikin_ashi(df)

    # Recalculate ATR, Chandelier stops, and direction
    tr = pd.DataFrame(index=ha_df.index)
    tr['ha_h'] = ha_df['ha_high']
    tr['ha_l'] = ha_df['ha_low']
    tr['ha_c'] = ha_df['ha_close']
    tr['prev_ha_c'] = tr['ha_c'].shift(1)
    tr['true_range'] = tr[['ha_h', 'ha_l', 'prev_ha_c']].apply(
        lambda row: max(
            row['ha_h'] - row['ha_l'],
            abs(row['ha_h'] - row['prev_ha_c']),
            abs(row['ha_l'] - row['prev_ha_c'])
        ), axis=1
    )
    tr['atr'] = tr['true_range'].rolling(window=atr_period).mean()

    long_stop = tr['ha_h'].rolling(window=atr_period).max() - (tr['atr'] * atr_mult)
    short_stop = tr['ha_l'].rolling(window=atr_period).min() + (tr['atr'] * atr_mult)

    long_stop_smooth = long_stop.copy()
    short_stop_smooth = short_stop.copy()
    for i in range(1, len(tr)):
        if tr['ha_c'].iloc[i - 1] > long_stop_smooth.iloc[i - 1]:
            long_stop_smooth.iloc[i] = max(long_stop.iloc[i], long_stop_smooth.iloc[i - 1])
        else:
            long_stop_smooth.iloc[i] = long_stop.iloc[i]

        if tr['ha_c'].iloc[i - 1] < short_stop_smooth.iloc[i - 1]:
            short_stop_smooth.iloc[i] = min(short_stop.iloc[i], short_stop_smooth.iloc[i - 1])
        else:
            short_stop_smooth.iloc[i] = short_stop.iloc[i]

    dir = [1]
    for i in range(1, len(tr)):
        if tr['ha_c'].iloc[i] > short_stop_smooth.iloc[i - 1]:
            dir.append(1)
        elif tr['ha_c'].iloc[i] < long_stop_smooth.iloc[i - 1]:
            dir.append(-1)
        else:
            dir.append(dir[-1])

    tr['dir'] = dir
    tr['dir_prev'] = tr['dir'].shift(1)
    tr['buy_signal'] = (tr['dir'] == 1) & (tr['dir_prev'] == -1)
    tr['sell_signal'] = (tr['dir'] == -1) & (tr['dir_prev'] == 1)

    # Evaluate latest signal
    latest = tr.iloc[-1]
    signal = None
    if latest['buy_signal']:
        signal = "buy"
    elif latest['sell_signal']:
        signal = "sell"

    position = get_position(symbol)

    if signal == "buy":
        if position is None:
            print(f"[{datetime.now()}] ⬆️ Buy Signal - Opening new LONG")
            send_order(symbol, mt5.ORDER_TYPE_BUY, lot_size)
        elif position.type == mt5.ORDER_TYPE_SELL:
            print(f"[{datetime.now()}] 🔁 Buy Signal - Closing SHORT, Opening LONG")
            close_position(position)
            time.sleep(1)
            send_order(symbol, mt5.ORDER_TYPE_BUY, lot_size)
    elif signal == "sell":
        if position is None:
            print(f"[{datetime.now()}] 🔻 Sell Signal - Opening new SHORT")
            send_order(symbol, mt5.ORDER_TYPE_SELL, lot_size)
        elif position.type == mt5.ORDER_TYPE_BUY:
            print(f"[{datetime.now()}] 🔁 Sell Signal - Closing LONG, Opening SHORT")
            close_position(position)
            time.sleep(1)
            send_order(symbol, mt5.ORDER_TYPE_SELL, lot_size)
    else:
        print(f"[{datetime.now()}] 🔄 No signal change. Holding...")

    # Wait until next 5-minute candle
    now = datetime.now()
    seconds_till_next = 300 - (now.minute % 5) * 60 - now.second
    print(f"⏳ Sleeping for {seconds_till_next} seconds...\n")
    time.sleep(seconds_till_next)
