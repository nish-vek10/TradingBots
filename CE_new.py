import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# === STEP_1: CONNECT TO MT5 === #
if not mt5.initialize():
    print("❌ Initialize failed:", mt5.last_error())
    quit()
else:
    print("✅ MT5 Initialised")

account = mt5.account_info()
if account is None:
    raise RuntimeError(f"Failed to retrieve account info: {mt5.last_error()}\n")

initial_cap = account.balance
print(f"✅ MT5 Account Connected! \nAccount: {account.login}, \nBalance: {initial_cap:.2f}\n")

# === STEP_2: INITIALISE PARAMETERS === #
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M5
bars = 500
lot_size = 0.1
atr_period = 1
atr_multiplier = 1.85

# === STEP_3: HEIKIN-ASHI CANDLE CALCULATION === #
def heikin_ashi(df):
    ha_df = pd.DataFrame(index=df.index)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [(df['open'][0] + df['close'][0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['ha_close'][i-1]) / 2)
    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = df[['high', 'low', 'close']].max(axis=1)
    ha_df['ha_low'] = df[['high', 'low', 'close']].min(axis=1)

    return ha_df

# ==== STEP_4: FETCH DATA FROM MT5 CHART ==== #
def fetch_data():
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print("❌ Failed to get data")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df

# ==== STEP_5: ATR CALCULATION ==== #
def calculate_atr(df, period):
    high_low = df['ha_high'] - df['ha_low']
    high_close = np.abs(df['ha_high'] - df['ha_close'].shift(1))
    low_close = np.abs(df['ha_low'] - df['ha_close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr

# ==== STEP_6: CHANDELIER EXIT STRATEGY ====
def chandelier_exit_strategy(df):
    atr = calculate_atr(df, atr_period)
    long_stop = df['ha_high'].rolling(atr_period).max() - (atr_multiplier * atr)
    short_stop = df['ha_low'].rolling(atr_period).min() + (atr_multiplier * atr)

    long_stop_final = [np.nan]
    short_stop_final = [np.nan]
    direction = [1]  # 1=long, -1=short

    for i in range(1, len(df)):
        ls = long_stop.iloc[i]
        ss = short_stop.iloc[i]

        prev_ls = long_stop_final[i-1] if not np.isnan(long_stop_final[i-1]) else ls
        prev_ss = short_stop_final[i-1] if not np.isnan(short_stop_final[i-1]) else ss
        prev_dir = direction[i-1]

        ls = max(ls, prev_ls) if df['ha_close'].iloc[i-1] > prev_ls else ls
        ss = min(ss, prev_ss) if df['ha_close'].iloc[i-1] < prev_ss else ss

        new_dir = 1 if df['ha_close'].iloc[i] > prev_ss else -1 if df['ha_close'].iloc[i] < prev_ls else prev_dir

        long_stop_final.append(ls)
        short_stop_final.append(ss)
        direction.append(new_dir)

    df['long_stop'] = long_stop_final
    df['short_stop'] = short_stop_final
    df['direction'] = direction
    df['buy_signal'] = (df['direction'] == 1) & (df['direction'].shift(1) == -1)
    df['sell_signal'] = (df['direction'] == -1) & (df['direction'].shift(1) == 1)

    return df


# ==== STEP_7: PLOT GRAPH FOR SIGNAL VISUALS ====
def plot_chart(df):
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['ha_close'], label='HA Close', color='blue')
    plt.plot(df.index, df['long_stop'], label='Long Stop', linestyle='--', color='green')
    plt.plot(df.index, df['short_stop'], label='Short Stop', linestyle='--', color='red')

    buy_signals = df[df['buy_signal']]
    sell_signals = df[df['sell_signal']]

    plt.scatter(buy_signals.index, buy_signals['ha_close'], color='green', marker='^', s=100, label='Buy Signal')
    for i in buy_signals.index:
        plt.text(i, buy_signals.loc[i, 'ha_close'], 'B', color='green', fontsize=12, ha='center')

    plt.scatter(sell_signals.index, sell_signals['ha_close'], color='red', marker='v', s=100, label='Sell Signal')
    for i in sell_signals.index:
        plt.text(i, sell_signals.loc[i, 'ha_close'], 'S', color='red', fontsize=12, ha='center')

    plt.title(f"{symbol} - Chandelier Exit Signals")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# ==== STEP_8: PLACE ORDER WITHOUT SL/TP === #
def place_order(signal):
    order_type = mt5.ORDER_TYPE_BUY if signal == "buy" else mt5.ORDER_TYPE_SELL
    ticket = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": mt5.symbol_info_tick(symbol).ask if signal == "buy" else mt5.symbol_info_tick(symbol).bid,
        "deviation": 5,
        "magic": 10072025,
        "comment": "ChandelierBot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })
    print(f"✅ Order sent: {signal.upper()}, Ticket: {ticket}")

# ==== STEP_9: MANAGE EXISTING POSITIONS ==== #
def close_existing(opposite_type):
    positions = mt5.positions_get(symbol=symbol)
    for pos in positions:
        if pos.type == opposite_type:
            price = mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 5,
                "magic": 10072025,
                "comment": "Close for opposite signal",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            })
            print("❎ Closed Existing Opposite Position.")

# ==== STEP_10: MAIN LOOP ==== #
last_signal_time = None

while True:
    try:
        df = fetch_data()
        if df is None:
            time.sleep(60)
            continue

        ha_df = heikin_ashi(df)
        df = pd.concat([df, ha_df], axis=1)
        df = chandelier_exit_strategy(df)

        # Show last 10 HA candles
        print("\n= = = = =  LAST 10 HEIKIN-ASHI CANDLES = = = = =")
        print(df[['ha_open', 'ha_high', 'ha_low', 'ha_close']].tail(10))

        latest = df.iloc[-1]
        current_time = df.index[-1]

        if latest['buy_signal'] and current_time != last_signal_time:
            close_existing(mt5.ORDER_TYPE_SELL)
            place_order("buy")
            last_signal_time = current_time

        elif latest['sell_signal'] and current_time != last_signal_time:
            close_existing(mt5.ORDER_TYPE_BUY)
            place_order("sell")
            last_signal_time = current_time

        plot_chart(df.tail(100))  # Optional: remove if running headless

        time.sleep(300)  # Wait for next 5M candle
    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(60)