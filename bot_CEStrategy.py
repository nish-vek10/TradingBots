"""
Chandelier Exit Strategy
Using Heikin-Ashi Candles
"""

import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
import os
import requests
import pandas as pd
from pytz import timezone

# === CONFIGURATION === #
mt5_symbol = "XAUUSD"
oanda_symbol = "XAU_USD"
timeframe = mt5.TIMEFRAME_M5
num_candles = 500
lot_size = 1.00
slippage = 25
atr_period = 1
atr_mult = 1.85
magic_number = 123456 # Unique ID for this EA's trades
local_tz = timezone('Europe/London')

# === ACCOUNT LOGIN CONFIG === #
mt5_login = 52421640
mt5_password = "M3Bgywv9$n8mr1"
mt5_server = "ICMarketsSC-Demo"
mt5_terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"

print("MT5 Path Exists?", os.path.exists(mt5_terminal_path))

# === CONNECT TO MT5 === #
if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server, path=mt5_terminal_path):
    print("[ERROR] MT5 initialization failed:", mt5.last_error())
    quit()

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

print(f"\nMT5 Account Connected! \nAccount: {account.login} \nBalance: ${account.balance:.2f}\n")

symbol_info = mt5.symbol_info(mt5_symbol)
if symbol_info:
    print(f"[INFO] {mt5_symbol} Lot Range: min={symbol_info.volume_min}, max={symbol_info.volume_max}, step={symbol_info.volume_step}")
else:
    print(f"[ERROR] Unable to fetch symbol info for {mt5_symbol}")

print(symbol_info._asdict())

oanda_token = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_account_id = "101-004-35770497-001"
oanda_api_url = "https://api-fxpractice.oanda.com/v3"

def fetch_oanda_candles(symbol=oanda_symbol, granularity="M5", count=500):
    """
    Fetch candle data from OANDA REST API.
    symbol: OANDA instrument name "XAU_"
    granularity: M1, M5, M15, H1, D etc.
    count: number of candles to fetch
    """
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M"  # Midpoint prices
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print("[ERROR] Failed to fetch candles from OANDA:", response.status_code, response.text)
        return None

    raw_candles = response.json().get("candles", [])
    data = {
        "time": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": []
    }

    for candle in raw_candles:
        if candle['complete']:
            utc_time = pd.to_datetime(candle['time'])
            local_time = utc_time.tz_convert(local_tz)
            data['time'].append(local_time)
            data['open'].append(float(candle['mid']['o']))
            data['high'].append(float(candle['mid']['h']))
            data['low'].append(float(candle['mid']['l']))
            data['close'].append(float(candle['mid']['c']))
            data['volume'].append(int(candle['volume']))

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df


# === UTILITY FUNCTIONS === #
def get_position(symbol):
    """
    Check for existing position on a symbol.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None

    for p in positions:
        if p.magic == magic_number:
            return p
    return None

def send_order(symbol, action_type, lot=lot_size):
    """
    Send a buy or sell order.
    """
    # === VOLUME VALIDATION ===
    if lot is None or lot <= 0:
        print(f"[ERROR] Invalid trade volume: {lot}")
        return

    # === SYMBOL INFO CHECK ===
    # Ensure symbol is selected
    mt5.symbol_select(symbol, True)

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return

    # Single safe check
    if hasattr(symbol_info, "trade_allowed") and not symbol_info.trade_allowed:
        print(f"[ERROR] Trading is not allowed for {symbol}")
        return

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"[ERROR] Symbol info not found for {symbol}")
        return

    if lot < symbol_info.volume_min or lot > symbol_info.volume_max:
        print(f"[ERROR] Lot size {lot} out of range: min={symbol_info.volume_min}, max={symbol_info.volume_max}")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[ERROR] Failed to get tick data for {symbol}")
        return

    price = tick.ask if action_type == mt5.ORDER_TYPE_BUY else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "price": price,
        "deviation": slippage,
        "magic": magic_number,
        "comment": "ChandelierEntryBot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"[ERROR] order_send() returned None. Last error: {mt5.last_error()}")
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order failed: {result.retcode}, {result.comment}")
    else:
        print(f"[OK] Order placed: {result}")

def close_position(position, symbol):
    """
    Close an open position.
    """
    if position is None:
        return
    action_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": action_type,
        "position": position.ticket,
        "price": price,
        "deviation": slippage,
        "magic": magic_number,
        "comment": "Close opposite position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(close_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Failed to close position: {result.retcode}, {result.comment}")
    else:
        print(f"[OK] Position closed: {result}")

# === HEIKIN ASHI CALCULATION === #
def calculate_heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha_df['ha_close'].iloc[i - 1]) / 2)

    ha_df['ha_open'] = ha_open

    ha_df['ha_high'] = pd.concat([df['high'], ha_df['ha_open'], ha_df['ha_close']], axis=1).max(axis=1)
    ha_df['ha_low'] = pd.concat([df['low'], ha_df['ha_open'], ha_df['ha_close']], axis=1).min(axis=1)
    return ha_df

# === CALCULATE INDICATORS: ATR, Stops, Direction === #
def calculate_indicators(df):
    ha_df = calculate_heikin_ashi(df)

    tr = pd.DataFrame(index=ha_df.index)
    tr['ha_h'] = ha_df['ha_high']
    tr['ha_l'] = ha_df['ha_low']
    tr['ha_c'] = ha_df['ha_close']
    tr['prev_ha_c'] = tr['ha_c'].shift(1)

    # Include for debugging/visualisation
    tr['ha_open'] = ha_df['ha_open']
    tr['ha_high'] = ha_df['ha_high']
    tr['ha_low'] = ha_df['ha_low']

    # True Range & ATR
    tr['true_range'] = tr[['ha_h', 'ha_l', 'prev_ha_c']].apply(
        lambda row: max(
            row['ha_h'] - row['ha_l'],
            abs(row['ha_h'] - row['prev_ha_c']),
            abs(row['ha_l'] - row['prev_ha_c'])
        ), axis=1
    )

    tr['atr'] = tr['true_range'].ewm(alpha=1/atr_period, adjust=False).mean()

    # Chandelier Stops
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

    # Direction Detection
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

    return tr

# === MAIN TRADING LOOP === #
last_candle_time = None
last_signal = None

while True:
    # Wait until next 5-minute candle close
    now = datetime.now()
    seconds = now.minute * 60 + now.second
    sleep_seconds = (300 - (seconds % 300)) % 300
    print(f"\n[*] Waiting {sleep_seconds} seconds until next 5-min candle close...")
    time.sleep(sleep_seconds)

    retry_timeout = timedelta(minutes=2)
    start_time = datetime.now()

    # Then wait for a truly new candle to be returned by OANDA
    while True:
        df = fetch_oanda_candles(symbol=oanda_symbol, granularity="M5", count=500)

        if df is None or df.empty:
            print("[ERROR] Failed to retrieve data from OANDA.")
            time.sleep(10)
            continue

        latest_candle_time = df.index[-1]

        if last_candle_time is None or latest_candle_time > last_candle_time:
            last_candle_time = latest_candle_time
            print(f"\n[OK] New 5-min candle detected: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
            break
        else:
            print(
                f"[WAIT] No new candle yet. Latest: {latest_candle_time.strftime('%H:%M:%S')} | Retrying in 10 seconds...")
            time.sleep(10)

        if datetime.now() - start_time > retry_timeout:
            print(f"[TIMEOUT] No new candle after {retry_timeout.seconds} seconds. Skipping this cycle.")
            break

    # Proceed with signal generation and trading
    print(f"[OK] Retrieved {len(df)} candles from OANDA.")

    # Print raw candlestick data for validation
    print("\n= = = = =   RAW OANDA CANDLESTICK DATA (LAST 10 CANDLES)   = = = = =")
    print(df.assign(time=df.index.strftime('%Y-%m-%d %H:%M')).set_index('time').tail(10))

    # # Print Heikin-Ashi version for validation
    # ha_debug = calculate_heikin_ashi(df)
    # print("\n= = = = =   HEIKIN-ASHI CANDLES CALCULATION (LAST 10 CANDLES):  = = = = =")
    # print(ha_debug.assign(time=ha_debug.index.strftime('%Y-%m-%d %H:%M')).set_index('time').tail(10))

    # Step 2: Calculate all indicators and signals
    tr = calculate_indicators(df)
    latest = tr.iloc[-1]

    # Step 3: Print last 10 Heikin Ashi candles with signals
    print("\n= = = = =   LAST 10 HEIKIN-ASHI CANDLES WITH SIGNALS  = = = = =")
    debug_df = tr[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'buy_signal', 'sell_signal']].copy()
    debug_df.index = debug_df.index.strftime('%Y-%m-%d %H:%M')
    debug_df['signal'] = debug_df.apply(lambda row: 'BUY' if row['buy_signal'] else ('SELL' if row['sell_signal'] else ''), axis=1)
    print(debug_df[['ha_c', 'ha_open', 'ha_high', 'ha_low', 'dir', 'signal']].tail(10))

    # Step 4: Determine signal
    signal = None
    if latest['buy_signal']:
        signal = 'BUY'
    elif latest['sell_signal']:
        signal = 'SELL'

    # Get current position info
    position = get_position(mt5_symbol)
    open_position = None
    if position:
        open_position = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
        print(f"[INFO] Open position: {open_position}, volume: {position.volume}, entry: {position.price_open}")
    else:
        print("[INFO] No open position currently.")

    if signal == 'BUY':
        if open_position == 'SELL':
            close_position(position, mt5_symbol)
        if open_position != 'BUY':
            send_order(mt5_symbol, mt5.ORDER_TYPE_BUY)

    elif signal == 'SELL':
        if open_position == 'BUY':
            close_position(position, mt5_symbol)
        if open_position != 'SELL':
            send_order(mt5_symbol, mt5.ORDER_TYPE_SELL)