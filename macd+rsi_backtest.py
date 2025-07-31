import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
import datetime as dt
import os

# === PARAMETERS === #
ticker = input("Enter ticker symbol (e.g., AAPL, TSLA, BTC-USD): ").upper()

end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=7) # yfinance 5-min data for past ~60 days available
interval = "5m"

initial_capital = 100000
risk_per_trade_pct = 0.005
risk_reward_ratio = 2
lookback = 10  # Candles to look back for swing high/low
max_open_trades = 4  # Maximum allowed simultaneous open positions

# === DOWNLOAD 1-MINUTE DATA === #
print(f"Downloading {ticker} {interval} data...")
data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

if data.empty:
    raise ValueError("No data found. Try a different ticker or time frame.")

# Fix multi-index columns from yfinance if any
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# === INDICATORS === #
# Calculate indicators: MACD and RSI
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# RSI calculation function
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data['Close'], 14)

# Helper functions for swing highs/lows
def get_swing_low(prices, idx, lookback):
    # returns lowest low in last 'lookback' bars before idx
    if idx - lookback < 0:
        return np.nan
    return prices.iloc[idx - lookback:idx].min()

def get_swing_high(prices, idx, lookback):
    # returns highest high in last 'lookback' bars before idx
    if idx - lookback < 0:
        return np.nan
    return prices.iloc[idx - lookback:idx].max()

# --- TRADE MANAGEMENT ---
open_trades = []
closed_trades = []
capital = initial_capital
equity_curve = []

for i in range(lookback + 1, len(data)):
    date = data.index[i]
    close = data['Close'].iloc[i]
    open_ = data['Open'].iloc[i]
    low = data['Low'].iloc[i]
    high = data['High'].iloc[i]
    macd = data['MACD'].iloc[i]
    signal = data['MACD_Signal'].iloc[i]
    rsi = data['RSI'].iloc[i]

    prev_macd = data['MACD'].iloc[i - 1]
    prev_signal = data['MACD_Signal'].iloc[i - 1]
    prev_rsi = data['RSI'].iloc[i - 1]

    # Skip if indicators are NaN
    if np.isnan([macd, signal, rsi, prev_macd, prev_signal, prev_rsi]).any():
        equity_curve.append({'date': date, 'equity': capital})
        continue

    # --- ENTRY LOGIC ---

    # Long Entry Conditions:
    # MACD crossover up AND RSI above 50
    # OR RSI crossing above 30 (out of oversold) AND MACD above signal
    long_entry_signal = (
            ((prev_macd < prev_signal) and (macd > signal) and (rsi > 50)) or
            ((prev_rsi < 30) and (rsi > 30) and (macd > signal))
    )

    if len(open_trades) <= max_open_trades and long_entry_signal:
        swing_low = get_swing_low(data['Low'], i, lookback)
        if np.isnan(swing_low):
            # Cannot define swing low yet
            pass
        else:
            sl = swing_low
            entry_price = close
            risk_per_share = entry_price - sl
            if risk_per_share > 0:
                max_risk_amount = capital * risk_per_trade_pct
                position_size = max_risk_amount / risk_per_share
                tp = entry_price + risk_reward_ratio * risk_per_share

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

    # Short Entry Conditions:
    # MACD crossover down AND RSI below 50
    # OR RSI crossing below 70 (out of overbought) AND MACD below signal
    short_entry_signal = (
            ((prev_macd > prev_signal) and (macd < signal) and (rsi < 50)) or
            ((prev_rsi > 70) and (rsi < 70) and (macd < signal))
    )

    if len(open_trades) <= max_open_trades and short_entry_signal:
        swing_high = get_swing_high(data['High'], i, lookback)
        if np.isnan(swing_high):
            pass
        else:
            sl = swing_high
            entry_price = close
            risk_per_share = sl - entry_price
            if risk_per_share > 0:
                max_risk_amount = capital * risk_per_trade_pct
                position_size = max_risk_amount / risk_per_share
                tp = entry_price - risk_reward_ratio * risk_per_share

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
        sl = trade['stop_loss']
        tp = trade['take_profit']
        pos_size = trade['position_size']

        # Long trade exits
        if side == 'long':
            # Stop loss hit
            if low <= sl:
                exit_price = sl
                pnl_per_share = exit_price - trade['entry_price']
            # Take profit hit
            elif high >= tp:
                exit_price = tp
                pnl_per_share = exit_price - trade['entry_price']
            else:
                continue  # no exit this candle

        # Short trade exits
        else:
            # Stop loss hit
            if high >= sl:
                exit_price = sl
                pnl_per_share = trade['entry_price'] - exit_price
            # Take profit hit
            elif low <= tp:
                exit_price = tp
                pnl_per_share = trade['entry_price'] - exit_price
            else:
                continue  # no exit this candle

        # Calculate PnL
        pnl = pnl_per_share * pos_size
        pnl_pct = pnl / capital * 100

        # Update trade info
        trade['exit_date'] = date
        trade['exit_price'] = exit_price
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct

        # Update capital and close trade
        capital += pnl
        closed_trades.append(trade)
        open_trades.remove(trade)

    # Track equity daily
    equity_curve.append({'date': date, 'equity': capital})

# === RESULTS === #
trade_log = pd.DataFrame(closed_trades)
equity_df = pd.DataFrame(equity_curve).set_index('date')

# Metrics
total_trades = len(trade_log)
total_profit = trade_log['pnl'].sum() if total_trades > 0 else 0
avg_pnl = trade_log['pnl'].mean() if total_trades > 0 else 0
pct_return = (capital - initial_capital) / initial_capital * 100
win_rate = (trade_log['pnl'] > 0).sum() / total_trades if total_trades > 0 else 0

# Max Drawdown calculation
equity_df['peak'] = equity_df['equity'].cummax()
equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
max_drawdown = equity_df['drawdown'].min()
lowest_equity = equity_df['equity'].min()

# Sharpe Ratio (annualized)
if len(equity_df) > 1:
    returns = equity_df['equity'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * (6.5 * 60 / 5))  # Adjust for 5-min bars in trading hours
else:
    sharpe_ratio = np.nan

# Output results
print("\n--- Trade Summary ---")
print(trade_log[['side', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'position_size', 'pnl']])

print(f"Risk per Trade:     {risk_per_trade_pct*100:.2f}%")
print(f"Initial Capital:    ${initial_capital:,.2f}")
print(f"Final Capital:      ${capital:,.2f}")
print(f"Net PnL:            ${total_profit:,.2f}")
print(f"Net Return (%):     {pct_return:,.2f}%")
print("------------------------------------------------------")
print(f"Total Trades:       {total_trades}")
print(f"Winning Trades:     {(trade_log['pnl'] > 0).sum()}")
print(f"Losing Trades:      {(trade_log['pnl'] <= 0).sum()}")
print(f"Win Rate:           {win_rate:.2%}")
print("------------------------------------------------------")
print(f"Average PnL/Trade:  ${avg_pnl:,.2f}")
print(f"Max Drawdown:       ${max_drawdown:,.2f}")
print(f"Lowest Equity:      ${lowest_equity:,.2f}")
print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")

# Prepare trade summary DataFrame
summary_data = {
    'Initial Capital ($)': [f"${initial_capital:,.2f}"],
    'Final Capital ($)': [f"${capital:,.2f}"],
    'Net PnL ($)': [f"${total_profit:,.2f}"],
    'Net Return (%)': [f"{pct_return:.2f}%"],
    'Total Trades': [total_trades],
    'Winning Trades': [(trade_log['pnl'] > 0).sum()],
    'Losing Trades': [(trade_log['pnl'] <= 0).sum()],
    'Win Rate (%)': [f"{win_rate * 100:.2f}%"],
    'Average PnL/Trade ($)': [f"${avg_pnl:,.2f}"],
    'Max Drawdown ($)': [f"${max_drawdown:,.2f}"],
    'Sharpe Ratio': [f"{sharpe_ratio:.2f}"]
}

summary_df = pd.DataFrame(summary_data)

# --- REMOVE TIMEZONES --- #
# Remove timezone from datetime columns and index to avoid Excel error
if not trade_log.empty:
    trade_log['entry_date'] = pd.to_datetime(trade_log['entry_date']).dt.tz_localize(None)
    trade_log['exit_date'] = pd.to_datetime(trade_log['exit_date']).dt.tz_localize(None)
equity_df.index = equity_df.index.tz_localize(None)

# Save folder & output
save_folder = r"C:\Users\anish\Desktop\Anish\AutomationTrading\MACD_RSI_Trading_Strategy"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

output_file = os.path.join(save_folder, f"{ticker}_MACD_RSI_TRADING_REPORT.xlsx")

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, sheet_name='Trade Summary', index=False)
    trade_log.to_excel(writer, sheet_name='Trade Log', index=False)
    equity_df.to_excel(writer, sheet_name='Equity Curve')

print(f"\nAll results exported to: {output_file}")

# --- PLOTTING --- #
# # PLOT CLOSE PRICE ONLY
# plt.figure(figsize=(15, 6))
# plt.plot(data['Close'], label='Close', alpha=0.6)
#
# # Plot markers for trades
# for trade in closed_trades:
#     entry_date = trade['entry_date']
#     exit_date = trade['exit_date']
#     entry_price = trade['entry_price']
#     exit_price = trade['exit_price']
#     pnl = trade['pnl']
#
#     # Winning trade
#     if pnl > 0:
#         # Green upward triangle for entry
#         plt.plot(entry_date, entry_price, marker='^', color='green', markersize=10, label='Buy (Win)' if 'Buy (Win)' not in plt.gca().get_legend_handles_labels()[1] else "")
#         # Circle on exit
#         plt.plot(exit_date, exit_price, marker='o', color='green', markersize=6)
#     else:
#         # Red downward triangle for entry
#         plt.plot(entry_date, entry_price, marker='v', color='red', markersize=10, label='Sell (Loss)' if 'Sell (Loss)' not in plt.gca().get_legend_handles_labels()[1] else "")
#         # Circle on exit
#         plt.plot(exit_date, exit_price, marker='o', color='red', markersize=6)
#
# plt.title(f"{ticker} Close Price Over Time")
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()



# === INTERACTIVE PLOT USING PLOTLY === #
# Plot close prices
fig_plt = go.Figure()
fig_plt.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='royalblue')
))

entry_win_shown = False
entry_loss_shown = False
exit_shown = False

# Add markers for each trade
for trade in closed_trades:
    entry_color = 'green' if trade['pnl'] > 0 else 'red'
    exit_color = entry_color
    shape_entry = 'triangle-up' if trade['side'] == 'long' else 'triangle-down'

    # Determine if this is a win or loss entry
    is_win = trade['pnl'] > 0
    entry_legend = "Entry (Win)" if is_win else "Entry (Loss)"
    show_entry_legend = not entry_win_shown if is_win else not entry_loss_shown
    if is_win:
        entry_win_shown = True
    else:
        entry_loss_shown = True

    # Control legend for exits
    show_exit = not exit_shown
    exit_shown = True

    # Entry marker
    fig_plt.add_trace(go.Scatter(
        x=[trade['entry_date']], y=[trade['entry_price']],
        mode='markers',
        marker=dict(symbol=shape_entry, color=entry_color, size=8),
        name=entry_legend,
        showlegend=show_entry_legend
    ))

    # Exit marker
    fig_plt.add_trace(go.Scatter(
        x=[trade['exit_date']], y=[trade['exit_price']],
        mode='markers',
        marker=dict(symbol='circle', color=entry_color, size=6),
        name="Exit",
        showlegend=show_exit
    ))

    # Connecting line (optional, legend hidden)
    fig_plt.add_trace(go.Scatter(
        x=[trade['entry_date'], trade['exit_date']],
        y=[trade['entry_price'], trade['exit_price']],
        mode='lines',
        line=dict(color=entry_color, width=1),
        opacity=0.4,
        showlegend=False  # always hide this
    ))

# Plot the figure
fig_plt.update_layout(
    title=f"{ticker} - Close Price with Trade Markers",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_white",
    autosize=True,
    margin=dict(l=10, r=10, t=40, b=10),
)
fig_plt.show()

# PLOT EQUITY CURVE
fig_eq = go.Figure()

fig_eq.add_trace(go.Scatter(
    x=equity_df.index,
    y=equity_df['equity'],
    mode='lines',
    name='Equity Curve',
    line=dict(color='blue', width=2)
))

fig_eq.update_layout(
    title='Interactive Equity Curve Over Time',
    xaxis_title='Date',
    yaxis_title='Equity ($)',
    template='plotly_white',
    height=600,
    hovermode='x unified'
)
fig_eq.show()