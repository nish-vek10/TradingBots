import MetaTrader5 as mt5
import pandas as pd
import time
import mplfinance as mpf
import datetime

# === CONNECT TO MT5 === #
if not mt5.initialize():
    print("MT5 initialization failed:", mt5.last_error())
    quit()

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

initial_cap = account.balance
print(f"✅ MT5 Account Connected! \nAccount: {account.login}, \nBalance: {initial_cap:.2f}\n")

# === PARAMETERS === #
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M5    # 5-minute candles
lot_size = 0.1                  # constant lot size per trade

# === PLACE ORDER WITHOUT SL/TP === #
def place_trade(direction):
    """
    Executes a market buy/sell order without SL or TP.
    """
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if direction == "buy" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 5,
        "magic": 123456,
        "comment": f"{direction.upper()} trade from bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade failed: {result.retcode}")
    else:
        print(f"✅ {direction.upper()} trade executed at {price}")

# === CLOSE OPPOSITE ORDERS BEFORE NEW ENTRY === #
def close_existing(direction):
    """
    Closes any open trade in the opposite direction before entering a new one.
    """
    opposite_type = mt5.ORDER_TYPE_SELL if direction == "buy" else mt5.ORDER_TYPE_BUY
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.type == opposite_type:
                price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 5,
                    "magic": 123456,
                    "comment": "Closing opposite trade",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(close_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Closed opposite trade: ticket {pos.ticket}")
                else:
                    print(f"❌ Failed to close trade: {result.retcode}")


# === HEIKIN-ASHI CANDLE CALCULATION === #
def calculate_heikin_ashi(df):
    """
    Converts OHLC candles into Heikin-Ashi format.
    """
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_df['ha_close'].iloc[i - 1]) / 2)

    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = df[['high', 'low', 'close']].max(axis=1)
    ha_df['ha_low'] = df[['high', 'low', 'close']].min(axis=1)

    return ha_df

# === MAIN STRATEGY === #
def run_strategy():
    """
    Core loop: fetch data, calculate signals, plot chart, and trade.
    """
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, 500) # Fetch latest 500 bars
    if bars is None:
        print("❌ Failed to fetch bars:", mt5.last_error())
        return

    df = pd.DataFrame(bars)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Calculate Heikin-Ashi candles
    ha_df = calculate_heikin_ashi(df)
    print(ha_df.tail(10))  # preview last few Heikin Ashi candles

    # Chandelier Exit Indicator Logic
    atr_period = 1
    atr_mult = 1.85

    tr = pd.DataFrame(index=ha_df.index)
    tr['tr'] = pd.concat([
            ha_df['ha_high'] - ha_df['ha_low'],
            abs(ha_df['ha_high'] - ha_df['ha_close'].shift()),
            abs(ha_df['ha_low'] - ha_df['ha_close'].shift())
        ], axis=1).max(axis=1)

    atr = tr['tr'].rolling(window=atr_period).mean()
    upper = ha_df['ha_high'].rolling(window=atr_period).max() - atr_mult * atr
    lower = ha_df['ha_low'].rolling(window=atr_period).min() + atr_mult * atr

    # === SIGNAL CALCULATION BLOCK === #
    trend = [0] * len(ha_df)

    for i in range(1, len(ha_df)):
        if ha_df['ha_close'].iloc[i] > upper.iloc[i - 1]:
            trend[i] = 1
        elif ha_df['ha_close'].iloc[i] < lower.iloc[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    ha_df['trend'] = trend

    # Detect crossovers
    ha_df['prev_trend'] = ha_df['trend'].shift(1)
    ha_df['buy_signal'] = (ha_df['trend'] == 1) & (ha_df['prev_trend'] == -1)
    ha_df['sell_signal'] = (ha_df['trend'] == -1) & (ha_df['prev_trend'] == 1)

    # === DEBUG: SHOW LAST 20 CANDLES + SIGNALS === #
    print("\nLAST 20 CANDLES WITH SIGNALS:")
    debug = ha_df[['ha_open', 'ha_high', 'ha_low', 'ha_close', 'trend', 'buy_signal', 'sell_signal']].tail(20)
    print(debug.to_string())

    # === TRADE EXECUTION BASED ON SIGNAL === #
    recent = ha_df.iloc[-1]

    if recent['buy_signal']:
        print(f"\n⬆️ BUY signal at {recent.name}")
        close_existing("buy")
        place_trade("buy")

    elif recent['sell_signal']:
        print(f"\n🔻 SELL signal at {recent.name}")
        close_existing("sell")
        place_trade("sell")

    else:
        print(f"\n⏳ No signal at {recent.name}")

    # === PLOT THE CHART WITH SIGNALS === #
    ha_plot_df = ha_df.copy()
    ha_plot_df.rename(columns={
        'ha_open': 'Open',
        'ha_high': 'High',
        'ha_low': 'Low',
        'ha_close': 'Close'
    }, inplace=True)
    ha_plot_df = ha_plot_df[['Open', 'High', 'Low', 'Close']]

    addplots = []
    if ha_df['buy_signal'].any():
        buys = ha_df[ha_df['buy_signal']]
        addplots.append(
            mpf.make_addplot(buys['Low'] - 1.0, type='scatter', marker='^', color='green', markersize=100))
    if ha_df['sell_signal'].any():
        sells = ha_df[ha_df['sell_signal']]
        addplots.append(
            mpf.make_addplot(sells['High'] + 1.0, type='scatter', marker='v', color='red', markersize=100))

    mpf.plot(
        ha_plot_df[-100:], type='candle', style='yahoo',
        addplot=addplots, title=f"{symbol} Heikin-Ashi Strategy",
        ylabel="Price", volume=False, figratio=(12, 6), figscale=1.2
    )

# === LOOP FOREVER === #
start_time = datetime.datetime.now()
end_time = start_time + datetime.timedelta(days=7)
print(f"\n⏱️ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Running strategy...")

try:
    while datetime.datetime.now() < end_time:
        run_strategy()
        time.sleep(60 * 5)
except KeyboardInterrupt:
    print("⛔ Stopped manually.")
finally:
    mt5.shutdown()
    print("✅ MT5 disconnected.")
