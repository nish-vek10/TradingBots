"""
MT5 Forward-Test Bot: MACD + RSI Swing Strategy

Trades both long & short, tracks equity, stops after specified duration.
"""

import MetaTrader5 as mt5
import pandas as pd
import time, datetime as dt


# === CONFIGURATION === #
SYMBOLS = ["XAUUSD"]            # Trading Symbol
TIMEFRAME = mt5.TIMEFRAME_M5    # 5-minute bars
LOOKBACK = 20                   # swing lookback
RISK_PER_TRADE_PCT = 0.005      # 0.5% risk per trade
RISK_REWARD = 2                 # 1:2 R/R
MAX_OPEN_TRADES = 4             # max simultaneous trades per symbol
TEST_DURATION_DAYS = 14         # forward-test duration

# === INITIALIZE MT5 === #
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed\n")

time.sleep(1)  # Ensure MT5 connection stabilizes

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

INITIAL_CAPITAL = account.balance
print(f"MT5 Account Connected! \nAccount: {account.login}, \nBalance: {INITIAL_CAPITAL}\n")

# Fetch and check symbol info for all symbols
symbol_info = {}
for sym in SYMBOLS:
    info = mt5.symbol_info(sym)
    if info is None or not info.visible:
        raise ValueError(f"Symbol {sym} not available in Market Watch\n")
    symbol_info[sym] = info
    print(f"Connected Symbol {sym} info: {info}\n")

# === UTILITIES === #
def get_latest_data(symbol, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def add_indicators(df):
    df['EMA12'] = df['close'].ewm(span=12).mean()
    df['EMA26'] = df['close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_SIG'] = df['MACD'].ewm(span=9).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def place_order(symbol, side, volume, sl, tp):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == 'buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if side == 'buy' else mt5.ORDER_TYPE_SELL
    info = symbol_info[symbol]

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": round(sl, info.digits),
        "tp": round(tp, info.digits),
        "deviation": 20,
        "magic": 234000,
        "comment": "MACD_RSI_BOT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(req)

# === MAIN BOT LOOP === #
start_time = dt.datetime.now()
end_time = start_time + dt.timedelta(days=TEST_DURATION_DAYS)

# Track open positions per symbol
open_positions = {sym: [] for sym in SYMBOLS}

print(f"Starting forward-test for symbols {SYMBOLS} from {start_time} to {end_time}\n")

while dt.datetime.now() < end_time:
    for sym in SYMBOLS:
        df = get_latest_data(sym, bars=LOOKBACK + 50)
        if df is None or len(df) <= LOOKBACK + 2:
            continue

        df = add_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        swing_high = df['high'].iloc[-LOOKBACK - 1:-1].max()
        swing_low = df['low'].iloc[-LOOKBACK - 1:-1].min()

        # Update capital and symbol info
        account = mt5.account_info()
        if account is None:
            print(f"⚠️ Failed to fetch account info for {sym}")
            continue
        capital = account.balance
        info = symbol_info[sym]

        # Entry signals
        long_sig = ((prev.MACD < prev.MACD_SIG and last.MACD > last.MACD_SIG and last.RSI > 50) or
                    (prev.RSI < 30 and last.RSI > 30 and last.MACD > last.MACD_SIG))
        short_sig = ((prev.MACD > prev.MACD_SIG and last.MACD < last.MACD_SIG and last.RSI < 50) or
                     (prev.RSI > 70 and last.RSI < 70 and last.MACD < last.MACD_SIG))

        # Long entry
        if long_sig and len(open_positions[sym]) < MAX_OPEN_TRADES and last.close > swing_low:
            entry = last.close
            risk = entry - swing_low
            if risk > 0:
                volume = (capital * RISK_PER_TRADE_PCT) / risk
                volume = max(info.volume_min, min(volume, info.volume_max))
                result = place_order(sym, 'buy', volume, swing_low, entry + RISK_REWARD * risk)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    open_positions[sym].append({'type': 'long', 'entry': entry, 'volume': volume,
                                                'sl': swing_low, 'tp': entry + RISK_REWARD * risk})
                    print(
                        f"{dt.datetime.now()} [{sym}] LONG opened @ {entry:.5f}, SL {swing_low:.5f}, TP {entry + RISK_REWARD * risk:.5f}")

        # Short entry
        if short_sig and len(open_positions[sym]) < MAX_OPEN_TRADES and last.close < swing_high:
            entry = last.close
            risk = swing_high - entry
            if risk > 0:
                volume = (capital * RISK_PER_TRADE_PCT) / risk
                volume = max(info.volume_min, min(volume, info.volume_max))
                result = place_order(sym, 'sell', volume, swing_high, entry - RISK_REWARD * risk)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    open_positions[sym].append({'type': 'short', 'entry': entry, 'volume': volume,
                                                'sl': swing_high, 'tp': entry - RISK_REWARD * risk})
                    print(
                        f"{dt.datetime.now()} [{sym}] SHORT opened @ {entry:.5f}, SL {swing_high:.5f}, TP {entry - RISK_REWARD * risk:.5f}")

        # Update existing orders & equity (simplified)
        positions = mt5.positions_get(symbol=sym) or []
        unrealized_profit = sum(pos.profit for pos in positions)
        equity = capital + unrealized_profit
        print(f"{dt.datetime.now()} [{sym}] Trades: {len(positions)}, Equity: ${equity:.2f}", end='\r')

    # Sleep until next 5-minute bar
    now = dt.datetime.now()
    seconds_to_next_5min = (5 - now.minute % 5) * 60 - now.second
    time.sleep(seconds_to_next_5min)

# === END TEST ===#
print("\nTest finished.")
account = mt5.account_info()
final_balance = account.balance if account else INITIAL_CAPITAL
profit = final_balance - INITIAL_CAPITAL
print(f"Final Balance: ${final_balance:.2f}, Net P/L: ${profit:.2f}")

mt5.shutdown()