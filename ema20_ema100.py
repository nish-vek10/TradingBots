import yfinance as yf               # For stock data
import pandas as pd                 # For data manipulation
import numpy as np                  # For numerical operations
import matplotlib.pyplot as plt     # For visualizations
import datetime
import os

# === PARAMETERS === #
ticker = input("Enter ticker symbol (e.g., AAPL, BTC-USD, TSLA): ").upper()
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
start_date = "2020-06-01"   # YYYY-MM-DD
initial_capital = 100000    # Account size
risk_per_trade_pct = 0.01   # Risk X% of capital per trade
risk_reward_ratio = 3       # R:R Ratio
lookback = 10               # Candles to look back for swing high/low

# === DOWNLOAD DATA === #
data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)

# Fix: Flatten columns (handles multi-index from yfinance)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(data)

# Compute EMAs
data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()

# Helper: Find recent swing highs/lows
def get_swing_low(series):
    return series.min()

def get_swing_high(series):
    return series.max()

# --- TRADE MANAGEMENT ---
open_trades = []
closed_trades = []
capital = initial_capital
equity_curve = []

# Backtest Loop
for i in range(lookback + 1, len(data)):
    date = data.index[i]
    close = data['Close'].iloc[i]
    open_ = data['Open'].iloc[i]
    low = data['Low'].iloc[i]
    high = data['High'].iloc[i]
    ema20 = data['EMA20'].iloc[i]
    ema100 = data['EMA100'].iloc[i]

    # --- ENTRY LOGIC ---
    # LONG ENTRY
    if (
        ema20 > ema100 and
        abs(close - ema20) / ema20 < 0.01 and
        close > open_  # Bullish candle
    ):
        swing_low = get_swing_low(data['Low'].iloc[i - lookback:i])
        sl = swing_low
        entry_price = close
        risk_per_share = entry_price - sl
        if risk_per_share <= 0:
            continue  # invalid trade setup

        # Calculate position size (shares) based on risk per trade capital
        max_risk_amount = capital * risk_per_trade_pct
        position_size = max_risk_amount / risk_per_share
        tp = entry_price + risk_reward_ratio * (entry_price - sl)

        open_trades.append({
            'side': 'long',
            'entry_date': date,
            'entry_price': entry_price,
            'stop_loss': sl,
            'take_profit': tp,
            'position_size': position_size,
            'exit_date': None,
            'exit_price': None,
            'pnl': None,
            'pnl_pct': None,
        })

    # SHORT ENTRY
    if (
        ema20 < ema100 and
        abs(close - ema100) / ema100 < 0.01 and
        close < open_  # Bearish candle
    ):
        swing_high = get_swing_high(data['High'].iloc[i - lookback:i])
        sl = swing_high
        entry_price = close
        risk_per_share = sl - entry_price
        if risk_per_share <= 0:
            continue  # invalid trade setup

        max_risk_amount = capital * risk_per_trade_pct
        position_size = max_risk_amount / risk_per_share
        tp = entry_price - risk_reward_ratio * (sl - entry_price)

        # Append new short trade
        open_trades.append({
            'side': 'short',
            'entry_date': date,
            'entry_price': entry_price,
            'stop_loss': sl,
            'take_profit': tp,
            'position_size': position_size,
            'exit_date': None,
            'exit_price': None,
            'pnl': None,
            'pnl_pct': None,
        })

    # --- EXIT LOGIC ---
    for trade in open_trades[:]:
        side = trade['side']
        position_size = trade['position_size']

        if side == 'long':
            # Stop loss
            if low <= trade['stop_loss']:
                exit_price = trade['stop_loss']
                pnl_per_share = exit_price - trade['entry_price']
            # Take profit
            elif high >= trade['take_profit']:
                exit_price = trade['take_profit']
                pnl_per_share = exit_price - trade['entry_price']
            else:
                continue  # no exit

        elif side == 'short':
            if high >= trade['stop_loss']:
                exit_price = trade['stop_loss']
                pnl_per_share = trade['entry_price'] - exit_price
            elif low <= trade['take_profit']:
                exit_price = trade['take_profit']
                pnl_per_share = trade['entry_price'] - exit_price
            else:
                continue

        # Calculate PnL
        pnl = pnl_per_share * position_size
        pnl_pct = round(pnl / capital * 100, 2)  # % as 2 decimal places

        # Update trade info
        trade['exit_price'] = exit_price
        trade['exit_date'] = date
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct

        # Update capital
        capital += pnl

        # Move trade from open to closed
        closed_trades.append(trade)
        open_trades.remove(trade)

        # Track equity over time (daily)
    equity_curve.append({'date': date, 'equity': capital})

# --- POST PROCESS RESULTS ---
trade_log = pd.DataFrame(closed_trades)
equity_df = pd.DataFrame(equity_curve).set_index('date')

# Metrics calculations
total_trades = len(trade_log)
total_profit = trade_log['pnl'].sum() if total_trades > 0 else 0
avg_pnl = trade_log['pnl'].mean() if total_trades > 0 else 0
win_rate = (trade_log['pnl'] > 0).sum() / total_trades if total_trades > 0 else 0

# Max Drawdown calculation
equity_df['peak'] = equity_df['equity'].cummax()
equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
max_drawdown = equity_df['drawdown'].min()

# Sharpe Ratio (annualized, assuming 252 trading days)
if len(equity_df) > 1:
    returns = equity_df['equity'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan
else:
    sharpe_ratio = np.nan

# --- OUTPUT RESULTS ---
print("\n--- Trade Summary ---")
print(trade_log[['side', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'position_size', 'pnl']])

print(f"\nInitial Capital:  ${initial_capital:,.2f}")
print(f"Risk per Trade:     {risk_per_trade_pct*100:,.2f}%")
print(f"Final Capital:      ${capital:,.2f}")
print(f"Net PnL:            ${total_profit:,.2f}")
print(f"Total Trades:       {total_trades}")
print(f"Winning Trades:     {(trade_log['pnl'] > 0).sum()}")
print(f"Losing Trades:      {(trade_log['pnl'] <= 0).sum()}")
print(f"Win Rate:           {win_rate:.2%}")
print(f"Average PnL/Trade:  ${avg_pnl:,.2f}")
print(f"Max Drawdown:       ${max_drawdown:,.2f}")
print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")

# Prepare trade summary DataFrame
summary_data = {
    'Initial Capital ($)': [f"${initial_capital:,.2f}"],
    'Final Capital ($)': [f"${capital:,.2f}"],
    'Net PnL ($)': [f"${total_profit:,.2f}"],
    'Total Trades': [total_trades],
    'Winning Trades': [(trade_log['pnl'] > 0).sum()],
    'Losing Trades': [(trade_log['pnl'] <= 0).sum()],
    'Win Rate (%)': [f"{win_rate * 100:.2f}%"],
    'Average PnL/Trade ($)': [f"${avg_pnl:,.2f}"],
    'Max Drawdown ($)': [f"${max_drawdown:,.2f}"],
    'Sharpe Ratio': [f"{sharpe_ratio:.2f}"]
}

summary_df = pd.DataFrame(summary_data)

# --- SAVE TRADE LOGS --- #
# Your target folder
save_folder = r"C:\Users\anish\Desktop\Anish\AutomationTrading\EMA_Trading_Strategy"

# Create folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

output_file = os.path.join(save_folder, f"{ticker}_TRADING_REPORT.xlsx")

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, sheet_name='Trade Summary', index=False)
    trade_log.to_excel(writer, sheet_name='Trade Log', index=False)
    equity_df.to_excel(writer, sheet_name='Equity Curve')

print(f"\nAll results exported to: {output_file}")

# --- PLOTTING --- #

# Price + EMAs
plt.figure(figsize=(16, 8))
plt.plot(data['Close'], label='Close', alpha=0.6)
plt.plot(data['EMA20'], label='EMA 20')
plt.plot(data['EMA100'], label='EMA 100')

for _, trade in trade_log.iterrows():
    color = 'green' if trade['pnl'] > 0 else 'red'
    marker_entry = '^' if trade['side'] == 'long' else 'v'
    plt.plot(trade['entry_date'], trade['entry_price'], marker=marker_entry, color=color)
    plt.plot(trade['exit_date'], trade['exit_price'], marker='x', color=color)
    plt.plot([trade['entry_date'], trade['exit_date']],
             [trade['entry_price'], trade['exit_price']], color=color, alpha=0.5)

plt.title(f"{ticker} Price with EMA20 & EMA100 - Multi Entry Strategy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Equity Curve Figure
plt.figure(figsize=(16, 6))
plt.plot(equity_df.index, equity_df['equity'], label='Equity Curve', color='blue')
plt.title('Equity Curve Over Time')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show all figures and block until all are closed manually
plt.show()