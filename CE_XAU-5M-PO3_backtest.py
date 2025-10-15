"""
Backtester — CE_XAU_5M_PO3 (Normal+Reverse with PO3-style regime)
- Heikin-Ashi + Chandelier Exit (ATR=RMA, period=1, mult=1.85)
- Dynamic risk: 0.5% of CURRENT equity per trade
- Normal (trend):   SL=5.5,  TP1=6.0 (60%), then SL->BE+buffer
- Reverse (consol): SL=3.0,  TP1=3.5 (60%), then SL->BE+buffer
- Entry: at next bar open after signal (no lookahead)
- Exit: opposite signal flips/flat (runner closes)
- Period: editable via DATE_FROM / DATE_TO
- Output: one XLSX with 3 sheets (Trades, Summary, EquityCurve + chart)
- Prints PARAMETERS and a concise SUMMARY to terminal
"""

import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict

import requests
import pandas as pd
import numpy as np

# =========================
# USER CONFIG
# =========================
OANDA_TOKEN = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
INSTRUMENT  = "XAU_USD"
GRANULARITY = "M5"
DATE_FROM   = "2025-05-01"
DATE_TO     = "2025-10-15"
LOCAL_TZ    = "Europe/London"

# Risk / sizing
RISK_PCT_PER_TRADE          = 0.50          # % of CURRENT equity
INIT_EQUITY                 = 100000.0      # starting equity
CONTRACT_VALUE_PER_DOLLAR   = 100.0       # per $1 underlying move @ 1.0 lot

# CE / HA params
USE_HEIKIN_ASHI = True
ATR_PERIOD      = 1
ATR_MULT        = 1.85

# Regime (PO3-style heuristic) params
PERSISTENCE_K   = 12
SMA_LEN         = 20
ATR_TREND_LEN   = 20
DIST_ATR_MULT   = 1.0
SWEEP_LOOKBACK  = 288  # ~ 24h @ M5

# Normal (trend) mode
SL_NORMAL    = 5.5
TP1_NORMAL   = 6.5
TP1_FRAC     = 0.60
BE_BUFFER    = 0.20   # buffer for SL->BE

# Reverse (consolidation) mode
SL_REVERSE   = 3.0
TP1_REVERSE  = 3.5

# Output
OUT_DIR      = r"C:\Users\anish\OneDrive\Desktop\Anish\Trading Bots\CE_XAU_5M_PO3 BACKTEST"
TAG          = datetime.now().strftime("%Y%m%d_%H%M%S")  # changeable tag to avoid overwrite
OUT_BASENAME = f"Backtest_CE_XAU_{GRANULARITY}_PO3_{DATE_FROM}_to_{DATE_TO}_{TAG}.xlsx"


# =========================
# UTILS / LOG
# =========================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {msg}", flush=True)

def _redact_token(tok: str) -> str:
    if not tok:
        return ""
    if len(tok) <= 8:
        return "****"
    return tok[:4] + "…[redacted]…" + tok[-4:]

def build_params_dict() -> "OrderedDict[str, str]":
    """
    Collect all key knobs so the Summary sheet + terminal log capture the config used.
    Token is redacted.
    """
    p = OrderedDict()
    # Data source
    p["Broker / Feed"] = "OANDA Practice (mid)"
    p["Instrument"] = INSTRUMENT
    p["Granularity"] = GRANULARITY
    p["Date From"] = DATE_FROM
    p["Date To"] = DATE_TO
    p["Local TZ"] = LOCAL_TZ
    p["OANDA Token"] = _redact_token(OANDA_TOKEN)

    # Risk / sizing
    p["Initial Equity"] = f"${INIT_EQUITY:,.2f}"
    p["Risk % / Trade"] = f"{RISK_PCT_PER_TRADE:.2f}%"
    p["$ per $1 move (1 lot)"] = f"{CONTRACT_VALUE_PER_DOLLAR:.2f}"

    # CE / HA
    p["Use Heikin-Ashi"] = str(USE_HEIKIN_ASHI)
    p["ATR Period"] = str(ATR_PERIOD)
    p["ATR Mult"] = f"{ATR_MULT:.2f}"

    # Regime (PO3-style)
    p["PO3 Persistence K"] = str(PERSISTENCE_K)
    p["PO3 SMA Len"] = str(SMA_LEN)
    p["PO3 ATR Trend Len"] = str(ATR_TREND_LEN)
    p["PO3 Dist*ATR Mult"] = f"{DIST_ATR_MULT:.2f}"
    p["PO3 Sweep Lookback"] = str(SWEEP_LOOKBACK)

    # Normal mode
    p["Normal SL ($)"] = f"{SL_NORMAL:.2f}"
    p["Normal TP1 ($)"] = f"{TP1_NORMAL:.2f}"
    p["TP1 Fraction"] = f"{TP1_FRAC:.2f}"
    p["BE Buffer ($)"] = f"{BE_BUFFER:.2f}"

    # Reverse mode
    p["Reverse SL ($)"] = f"{SL_REVERSE:.2f}"
    p["Reverse TP1 ($)"] = f"{TP1_REVERSE:.2f}"

    # Output
    p["Output Dir"] = OUT_DIR
    p["Output File (base)"] = OUT_BASENAME

    return p


# =========================
# DATA FETCH (OANDA)
# =========================
def oanda_candles(instrument: str, granularity: str, date_from: str, date_to: str, token: str) -> pd.DataFrame:
    """
    Pulls complete candles from OANDA (mid-prices).
    Returns tz-aware index in UTC; later we convert to LOCAL_TZ.
    """
    base = "https://api-fxpractice.oanda.com/v3"
    headers = {"Authorization": f"Bearer {token}"}

    # safety delta per granularity so we never ask for future bars
    def _granularity_delta(g):
        g = g.upper()
        if g == "M1":   return pd.Timedelta(minutes=1)
        if g == "M5":   return pd.Timedelta(minutes=5)
        if g == "M15":  return pd.Timedelta(minutes=15)
        if g == "M30":  return pd.Timedelta(minutes=30)
        if g == "H1":   return pd.Timedelta(hours=1)
        if g == "H4":   return pd.Timedelta(hours=4)
        if g == "D":    return pd.Timedelta(days=1)
        return pd.Timedelta(minutes=5)

    start = pd.Timestamp(date_from, tz="UTC")
    # inclusive end requested, but clamp to "now - one bar"
    now_utc = pd.Timestamp.now(tz="UTC")
    bar = _granularity_delta(granularity)
    requested_end = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)  # inclusive
    end = min(requested_end, now_utc - bar)  # ensure 'to' is not in the future
    if end <= start:
        raise ValueError("Requested date range is empty after clamping to current time. "
                         "Pick an earlier DATE_TO or a coarser granularity.")

    frames = []
    step = pd.Timedelta(days=5)
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cur.isoformat(),
            "to": nxt.isoformat(),
            "alignmentTimezone": "UTC",
            "includeFirst": "true"
        }
        r = requests.get(f"{base}/instruments/{instrument}/candles", headers=headers, params=params, timeout=(10, 30))
        if r.status_code != 200:
            raise RuntimeError(f"OANDA error {r.status_code}: {r.text[:300]}")
        arr = r.json().get("candles", [])
        rows = []
        for c in arr:
            if not c.get("complete", False):
                continue
            t = pd.to_datetime(c["time"], utc=True)
            mid = c.get("mid", {})
            rows.append([t, float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"]), int(c["volume"])])
        if rows:
            frames.append(pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]))
        cur = nxt
        time.sleep(0.1)

    if not frames:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    return df


# =========================
# INDICATORS
# =========================
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2.0)
    ha["ha_close"] = ha_close
    ha["ha_open"]  = pd.Series(ha_open, index=df.index)
    ha["ha_high"]  = pd.concat([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"]   = pd.concat([df["low"],  ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    return ha

def rma(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series.copy()
    alpha = 1.0 / length
    out = series.copy().astype(float)
    seed = series.rolling(length, min_periods=length).mean()
    out.iloc[:length-1] = np.nan
    out.iloc[length-1] = seed.iloc[length-1]
    for i in range(length, len(series)):
        out.iloc[i] = out.iloc[i-1] + alpha * (series.iloc[i] - out.iloc[i-1])
    return out.ffill()

def chandelier_engine(df: pd.DataFrame, use_ha=True, atr_period=1, atr_mult=1.85) -> pd.DataFrame:
    if use_ha:
        ha = heikin_ashi(df)
        o, h, l, c = ha["ha_open"], ha["ha_high"], ha["ha_low"], ha["ha_close"]
    else:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    tr1 = (h - l)
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = rma(tr, atr_period)
    atr_val = atr_mult * atr

    n = max(1, atr_period)
    hh = h.rolling(n, min_periods=n).max()
    ll = l.rolling(n, min_periods=n).min()
    long_stop  = hh - atr_val
    short_stop = ll + atr_val

    lss = long_stop.copy()
    sss = short_stop.copy()
    for i in range(1, len(df)):
        long_prev  = lss.iloc[i-1] if pd.notna(lss.iloc[i-1]) else long_stop.iloc[i]
        short_prev = sss.iloc[i-1] if pd.notna(sss.iloc[i-1]) else short_stop.iloc[i]
        if c.iloc[i-1] > long_prev:
            lss.iloc[i] = max(long_stop.iloc[i], long_prev)
        else:
            lss.iloc[i] = long_stop.iloc[i]
        if c.iloc[i-1] < short_prev:
            sss.iloc[i] = min(short_stop.iloc[i], short_prev)
        else:
            sss.iloc[i] = short_stop.iloc[i]

    dir_vals = [1]
    for i in range(1, len(df)):
        if c.iloc[i] > sss.iloc[i-1]:
            dir_vals.append(1)
        elif c.iloc[i] < lss.iloc[i-1]:
            dir_vals.append(-1)
        else:
            dir_vals.append(dir_vals[-1])

    out = pd.DataFrame(index=df.index)
    out["ha_open"]=o; out["ha_high"]=h; out["ha_low"]=l; out["ha_close"]=c
    out["long_stop_smooth"]  = lss
    out["short_stop_smooth"] = sss
    out["dir"] = dir_vals
    out["dir_prev"] = out["dir"].shift(1)
    out["buy_norm"]  = (out["dir"]==1) & (out["dir_prev"]==-1)
    out["sell_norm"] = (out["dir"]==-1) & (out["dir_prev"]==1)
    return out

def regime_po3_heuristic(df: pd.DataFrame, ce: pd.DataFrame) -> pd.Series:
    """
    Returns "trend" or "consol" per bar.
    - persistence of CE dir >= PERSISTENCE_K
    - |close - SMA20| >= ATR20 * DIST_ATR_MULT
    Otherwise -> consolidation
    With a small "sweep" nudge if we take out prev-day H/L within recent bars.
    """
    c = df["close"]
    sma = c.rolling(SMA_LEN, min_periods=1).mean()
    tr  = (df["high"] - df["low"]).abs()
    atr20 = rma(tr, ATR_TREND_LEN)

    dir_series = ce["dir"].fillna(0)
    same_count = (dir_series != dir_series.shift(1)).cumsum()
    persistence = same_count.groupby(same_count).transform('count')

    dist_ok = (c - sma).abs() >= (atr20 * DIST_ATR_MULT)
    pers_ok = persistence >= PERSISTENCE_K
    trend = pers_ok & dist_ok

    # Sweep nudge
    day = c.index.tz_convert("UTC").floor("D")
    daily_high = df["high"].groupby(day).transform("max").shift(1)
    daily_low  = df["low"].groupby(day).transform("min").shift(1)
    sweep = (df["high"] > daily_high) | (df["low"] < daily_low)
    sweep_recent = sweep.rolling(SWEEP_LOOKBACK, min_periods=1).max().astype(bool)

    regime = np.where(trend | sweep_recent, "trend", "consol")
    return pd.Series(regime, index=df.index)


# =========================
# EXECUTION MODEL
# =========================
@dataclass
class Trade:
    open_time: pd.Timestamp
    close_time: Optional[pd.Timestamp]
    side: str                 # 'BUY' or 'SELL'
    regime: str               # 'normal' or 'reverse'
    entry: float
    exit: float
    lots: float
    risk_pct: float
    sl_price: float
    tp1_price: float
    tp1_hit_time: Optional[pd.Timestamp]
    sl_to_be_time: Optional[pd.Timestamp]
    result: float             # PnL in $
    return_pct: float         # PnL / equity_at_entry
    reason_exit: str

def value_per_1usd_per_lot() -> float:
    return CONTRACT_VALUE_PER_DOLLAR

def size_lots(equity: float, sl_distance: float, risk_pct: float) -> Tuple[float, float]:
    risk_amt = equity * (risk_pct / 100.0)
    v = value_per_1usd_per_lot()
    risk_per_lot = v * sl_distance
    lots = risk_amt / risk_per_lot if risk_per_lot > 0 else 0.0
    lots = max(0.0, math.floor(lots * 100.0) / 100.0)  # 0.01 steps
    return lots, risk_amt

def apply_sl_tp1_intrabar(side: str, entry: float, sl: float, tp1: float,
                          bar_high: float, bar_low: float) -> Tuple[bool, bool]:
    """
    Returns (sl_hit, tp1_hit) inside the current bar using H/L.
    """
    if side == "BUY":
        if bar_low <= sl:
            return True, False
        if bar_high >= tp1:
            return False, True
    else:
        if bar_high >= sl:
            return True, False
        if bar_low <= tp1:
            return False, True
    return False, False


# =========================
# BACKTEST
# =========================
def backtest(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # localize to LOCAL_TZ for readability
    df = df_raw.tz_convert(LOCAL_TZ)

    # indicators
    ce = chandelier_engine(df, use_ha=USE_HEIKIN_ASHI, atr_period=ATR_PERIOD, atr_mult=ATR_MULT)
    regime_series = regime_po3_heuristic(df, ce)  # "trend" or "consol"

    sig_buy_norm  = ce["buy_norm"].fillna(False)
    sig_sell_norm = ce["sell_norm"].fillna(False)

    equity = INIT_EQUITY
    open_trade: Optional[Dict] = None
    trades: List[Trade] = []
    equity_curve = []

    opens = df["open"]; highs = df["high"]; lows = df["low"]; closes = df["close"]

    for i in range(1, len(df)-1):  # leave last bar for cleanliness
        t = df.index[i]
        nxt_t = df.index[i+1]

        regime_trend = (regime_series.iloc[i] == "trend")
        mode = "normal" if regime_trend else "reverse"

        buy_sig  = bool(sig_buy_norm.iloc[i])
        sell_sig = bool(sig_sell_norm.iloc[i])
        if mode == "reverse":
            buy_sig, sell_sig = sell_sig, buy_sig

        # ENTRY / FLIP at next open if a signal triggers
        if buy_sig or sell_sig:
            desired_side = "BUY" if buy_sig else "SELL"

            if open_trade is not None and open_trade["side"] != desired_side:
                exit_px = float(opens.iloc[i+1])
                pnl = (exit_px - open_trade["entry"]) * value_per_1usd_per_lot() * (1 if open_trade["side"]=="BUY" else -1) * open_trade["lots"]
                equity += pnl
                trades.append(Trade(
                    open_time=open_trade["open_time"], close_time=nxt_t, side=open_trade["side"],
                    regime=open_trade["regime"], entry=open_trade["entry"], exit=exit_px, lots=open_trade["lots"],
                    risk_pct=open_trade["risk_pct"], sl_price=open_trade["sl"], tp1_price=open_trade["tp1"],
                    tp1_hit_time=open_trade["tp1_hit_time"], sl_to_be_time=open_trade["sl_to_be_time"],
                    result=pnl, return_pct=(pnl/open_trade["equity_on_entry"] if open_trade["equity_on_entry"]>0 else 0.0),
                    reason_exit="Flip"
                ))
                open_trade = None

            if open_trade is None:
                entry_px = float(opens.iloc[i+1])
                if mode == "normal":
                    sl_dist, tp1_dist = SL_NORMAL, TP1_NORMAL
                else:
                    sl_dist, tp1_dist = SL_REVERSE, TP1_REVERSE

                sl = entry_px - sl_dist if desired_side=="BUY" else entry_px + sl_dist
                tp1= entry_px + tp1_dist if desired_side=="BUY" else entry_px - tp1_dist

                lots, _risk_amt = size_lots(equity, sl_dist, RISK_PCT_PER_TRADE)
                if lots > 0:
                    open_trade = {
                        "open_time": nxt_t,
                        "side": desired_side,
                        "regime": mode,
                        "entry": entry_px,
                        "sl": sl,
                        "tp1": tp1,
                        "lots": lots,
                        "risk_pct": RISK_PCT_PER_TRADE,
                        "tp1_hit_time": None,
                        "sl_to_be_time": None,
                        "equity_on_entry": equity,
                        "be_active": False,
                    }

        # Manage open position intrabar on NEXT bar
        if open_trade is not None:
            hi = float(highs.iloc[i+1]); lo = float(lows.iloc[i+1])
            side = open_trade["side"]
            sl = open_trade["sl"]; tp1 = open_trade["tp1"]
            lots = open_trade["lots"]

            sl_hit, tp1_hit = apply_sl_tp1_intrabar(side, open_trade["entry"], sl, tp1, hi, lo)

            if sl_hit:
                exit_px = sl
                pnl = (exit_px - open_trade["entry"]) * value_per_1usd_per_lot() * (1 if side=="BUY" else -1) * lots
                equity += pnl
                trades.append(Trade(
                    open_time=open_trade["open_time"], close_time=df.index[i+1], side=side,
                    regime=open_trade["regime"], entry=open_trade["entry"], exit=exit_px, lots=lots,
                    risk_pct=open_trade["risk_pct"], sl_price=open_trade["sl"], tp1_price=open_trade["tp1"],
                    tp1_hit_time=open_trade["tp1_hit_time"], sl_to_be_time=open_trade["sl_to_be_time"],
                    result=pnl, return_pct=(pnl/open_trade["equity_on_entry"] if open_trade["equity_on_entry"]>0 else 0.0),
                    reason_exit="SL"
                ))
                open_trade = None

            elif tp1_hit and open_trade["tp1_hit_time"] is None:
                # Take 60% at TP1, then move SL to BE+buffer
                part_frac = TP1_FRAC
                part_lots = round(lots * part_frac, 2)
                if part_lots > 0:
                    exit_px = tp1
                    pnl = (exit_px - open_trade["entry"]) * value_per_1usd_per_lot() * (1 if side=="BUY" else -1) * part_lots
                    equity += pnl
                    lots_new = round(lots - part_lots, 2)
                    open_trade["lots"] = lots_new
                    open_trade["tp1_hit_time"] = df.index[i+1]
                    be_px = open_trade["entry"] + (BE_BUFFER if side=="BUY" else -BE_BUFFER)
                    open_trade["sl"] = be_px
                    open_trade["sl_to_be_time"] = df.index[i+1]
                    open_trade["be_active"] = True

                if open_trade["lots"] <= 0:
                    trades.append(Trade(
                        open_time=open_trade["open_time"], close_time=df.index[i+1], side=side,
                        regime=open_trade["regime"], entry=open_trade["entry"], exit=tp1, lots=part_lots,
                        risk_pct=open_trade["risk_pct"], sl_price=be_px, tp1_price=open_trade["tp1"],
                        tp1_hit_time=open_trade["tp1_hit_time"], sl_to_be_time=open_trade["sl_to_be_time"],
                        result=0.0, return_pct=0.0, reason_exit="TP1_full"
                    ))
                    open_trade=None

        # Equity curve point (close of bar i)
        equity_curve.append({"time": t, "equity": equity})

    # Close any remaining position at last close
    if open_trade is not None:
        final_t = df.index[-1]
        exit_px = float(closes.iloc[-1])
        pnl = (exit_px - open_trade["entry"]) * value_per_1usd_per_lot() * (1 if open_trade["side"]=="BUY" else -1) * open_trade["lots"]
        equity += pnl
        trades.append(Trade(
            open_time=open_trade["open_time"], close_time=final_t, side=open_trade["side"],
            regime=open_trade["regime"], entry=open_trade["entry"], exit=exit_px, lots=open_trade["lots"],
            risk_pct=open_trade["risk_pct"], sl_price=open_trade["sl"], tp1_price=open_trade["tp1"],
            tp1_hit_time=open_trade["tp1_hit_time"], sl_to_be_time=open_trade["sl_to_be_time"],
            result=pnl, return_pct=(pnl/open_trade["equity_on_entry"] if open_trade["equity_on_entry"]>0 else 0.0),
            reason_exit="EOD"
        ))
        equity_curve.append({"time": final_t, "equity": equity})

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    if not trades_df.empty:
        trades_df.sort_values("open_time", inplace=True)
    eq_df = pd.DataFrame(equity_curve).drop_duplicates("time") if equity_curve else pd.DataFrame(columns=["time","equity"])
    if not eq_df.empty:
        eq_df.set_index("time", inplace=True)
    return trades_df, eq_df, ce


# =========================
# METRICS / SUMMARY
# =========================
def _streak_stats_with_pnl(pnls: List[float], win: bool = True):
    """
    Returns (max_len, count, best_sum, total_sum_across_max_len_streaks)
    For wins: streaks of pnl>0; for losses: pnl<0 (sums will be negative).
    """
    cmp = (lambda x: x > 0) if win else (lambda x: x < 0)
    max_len = 0
    count = 0
    sums_for_max = []
    cur_len = 0
    cur_sum = 0.0

    def close_run():
        nonlocal max_len, count, sums_for_max, cur_len, cur_sum
        if cur_len == 0:
            return
        if cur_len > max_len:
            max_len = cur_len
            count = 1
            sums_for_max = [cur_sum]
        elif cur_len == max_len:
            count += 1
            sums_for_max.append(cur_sum)
        cur_len = 0
        cur_sum = 0.0

    for v in pnls:
        if cmp(v):
            cur_len += 1
            cur_sum += v
        else:
            close_run()
    close_run()

    best_sum = (max(sums_for_max) if win else (min(sums_for_max) if sums_for_max else 0.0)) if sums_for_max else 0.0
    total_sum = sum(sums_for_max) if sums_for_max else 0.0
    return max_len, count, best_sum, total_sum

def _mt5_drawdown_stats(equity_series, initial_deposit):
    """
    Returns:
      maximal_dd_abs, maximal_dd_pct,
      absolute_dd_abs, absolute_dd_pct
    """
    peak = float('-inf')
    max_dd_abs = 0.0
    max_dd_pct = 0.0

    for eq in equity_series:
        if eq > peak:
            peak = eq
        dd_abs = peak - eq
        dd_pct = (dd_abs / peak * 100.0) if peak > 0 else 0.0
        if dd_abs > max_dd_abs:
            max_dd_abs = dd_abs
            max_dd_pct = dd_pct

    min_eq = min(equity_series) if len(equity_series) else initial_deposit
    abs_dd_abs = max(0.0, initial_deposit - min_eq)
    abs_dd_pct = (abs_dd_abs / initial_deposit * 100.0) if initial_deposit > 0 else 0.0

    return max_dd_abs, max_dd_pct, abs_dd_abs, abs_dd_pct

def compute_summary(trades_df: pd.DataFrame, eq_df: pd.DataFrame, params_dict: "OrderedDict[str, str]") -> pd.DataFrame:
    # Always include parameters first
    params_rows = [("— PARAMETERS —", "")]
    params_rows += [(k, v) for k, v in params_dict.items()]
    params_rows.append(("", ""))  # spacer

    if trades_df.empty:
        return pd.DataFrame(params_rows + [("No trades", "-")], columns=["Metric","Value"])

    start_time = trades_df["open_time"].min()
    end_time   = trades_df["close_time"].max()

    total_trades = len(trades_df)
    long_trades  = (trades_df["side"]=="BUY").sum()
    short_trades = (trades_df["side"]=="SELL").sum()
    longs_pct  = f"{(long_trades/total_trades*100.0):.1f}%" if total_trades>0 else "0.0%"
    shorts_pct = f"{(short_trades/total_trades*100.0):.1f}%" if total_trades>0 else "0.0%"

    wins = (trades_df["result"]>0).sum()
    win_rate = wins / total_trades if total_trades>0 else 0.0
    risk_pct = trades_df["risk_pct"].iloc[0] if "risk_pct" in trades_df and not trades_df["risk_pct"].empty else RISK_PCT_PER_TRADE

    # Streak lengths + counts + PnL
    pnl_list = trades_df["result"].tolist()
    w_len, w_cnt, w_best_sum, w_total_sum = _streak_stats_with_pnl(pnl_list, win=True)
    l_len, l_cnt, l_best_sum, l_total_sum = _streak_stats_with_pnl(pnl_list, win=False)

    # Largest profit / loss per trade (amount + %), side + open time
    max_win_amt = None; max_loss_amt = None
    max_win_pct = None; max_loss_pct = None
    max_win_side = "";  max_loss_side = ""
    max_win_time = "";  max_loss_time = ""

    if not trades_df.empty:
        idx_max = trades_df["result"].idxmax()
        idx_min = trades_df["result"].idxmin()
        if pd.notna(idx_max):
            row_max = trades_df.loc[idx_max]
            max_win_amt = float(row_max["result"])
            max_win_pct = float(row_max.get("return_pct", np.nan))
            max_win_side = str(row_max.get("side",""))
            max_win_time = str(row_max.get("open_time",""))
        if pd.notna(idx_min):
            row_min = trades_df.loc[idx_min]
            max_loss_amt = float(row_min["result"])
            max_loss_pct = float(row_min.get("return_pct", np.nan))
            max_loss_side = str(row_min.get("side",""))
            max_loss_time = str(row_min.get("open_time",""))

    # Equity & MT5-style drawdowns
    if not eq_df.empty:
        start_eq = float(eq_df["equity"].iloc[0])
        end_eq = float(eq_df["equity"].iloc[-1])
        eq_series = eq_df["equity"].astype(float).tolist()
    else:
        start_eq = INIT_EQUITY
        end_eq = INIT_EQUITY + trades_df["result"].sum()
        eq_series = [start_eq, end_eq]

    total_return_amt = end_eq - start_eq
    total_return_pct = (total_return_amt / start_eq) * 100.0 if start_eq > 0 else 0.0

    max_dd_abs, max_dd_pct, abs_dd_abs, abs_dd_pct = _mt5_drawdown_stats(eq_series, INIT_EQUITY)

    rows = [
        ("— SUMMARY —", ""),
        ("Start Time", str(start_time)),
        ("End Time", str(end_time)),
        ("Total Trades", f"{total_trades}"),
        ("Long Trades", f"{long_trades} ({longs_pct})"),
        ("Short Trades", f"{short_trades} ({shorts_pct})"),
        ("Win Rate", f"{win_rate * 100.0:.2f}%"),
        ("Risk per Trade", f"{risk_pct:.2f}% of balance"),
        ("Max Consecutive Wins",
         f"{w_len} ({w_cnt} streaks) — Best streak PnL: ${w_best_sum:,.2f} | Total across max-len: ${w_total_sum:,.2f}"),
        ("Max Consecutive Losses",
         f"{l_len} ({l_cnt} streaks) — Worst streak PnL: ${l_best_sum:,.2f} | Total across max-len: ${l_total_sum:,.2f}"),
        ("Maximal Drawdown ($/% )", f"${max_dd_abs:,.2f} / {max_dd_pct:.2f}%"),
        ("Absolute Drawdown ($/% )", f"${abs_dd_abs:,.2f} / {abs_dd_pct:.2f}%"),
        ("Total Return ($)", f"{total_return_amt:,.2f}"),
        ("Total Return (%)", f"{total_return_pct:.2f}%"),
        ("End Equity", f"{end_eq:,.2f}"),
    ]

    if max_win_amt is not None:
        rows.append(("Largest Profit ($ / %)",
                     f"${max_win_amt:,.2f} / {max_win_pct * 100.0:.2f}% — {max_win_side} @ {max_win_time}"))
    if max_loss_amt is not None:
        rows.append(("Largest Loss ($ / %)",
                     f"${max_loss_amt:,.2f} / {max_loss_pct * 100.0:.2f}% — {max_loss_side} @ {max_loss_time}"))

    params_df  = pd.DataFrame(params_rows, columns=["Metric","Value"])
    metrics_df = pd.DataFrame(rows,        columns=["Metric","Value"])
    return pd.concat([params_df, metrics_df], ignore_index=True)


# =========================
# WRITE XLSX
# =========================
def write_xlsx(trades_df: pd.DataFrame, summary_df: pd.DataFrame, eq_df: pd.DataFrame, out_path: str):
    def _make_naive(df: pd.DataFrame, cols: list, to_tz: str = LOCAL_TZ) -> pd.DataFrame:
        """Convert any tz-aware datetime columns to the given tz, then drop tz (Excel-safe)."""
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
                df[c] = df[c].dt.tz_convert(to_tz).dt.tz_localize(None)
        return df

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Sheet 1: Trades
        if not trades_df.empty:
            tdf = trades_df.copy()
            tdf = _make_naive(tdf, cols=["open_time","close_time","tp1_hit_time","sl_to_be_time"], to_tz=LOCAL_TZ)
            cols = [
                "open_time","close_time","side","regime","entry","exit","lots","risk_pct",
                "sl_price","tp1_price","tp1_hit_time","sl_to_be_time","result","return_pct","reason_exit"
            ]
            cols = [c for c in cols if c in tdf.columns] + [c for c in tdf.columns if c not in cols]
            tdf[cols].to_excel(writer, sheet_name="Trades", index=False)
        else:
            pd.DataFrame({"info":["No trades"]}).to_excel(writer, sheet_name="Trades", index=False)

        # Sheet 2: Summary (already contains parameters + metrics)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 3: EquityCurve (+ chart)
        if not eq_df.empty:
            eq_out = eq_df.reset_index().copy()
            eq_out["time"] = pd.to_datetime(eq_out["time"], errors="coerce", utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
            eq_out.to_excel(writer, sheet_name="EquityCurve", index=False)

            wb  = writer.book
            ws  = writer.sheets["EquityCurve"]
            chart = wb.add_chart({'type': 'line'})
            chart.add_series({
                'name': 'Equity',
                'categories': ['EquityCurve', 1, 0, len(eq_out), 0],
                'values':     ['EquityCurve', 1, 1, len(eq_out), 1],
            })
            chart.set_title({'name': 'Equity Curve'})
            chart.set_x_axis({'name': 'Time'})

            # y-axis floor to nearest 10k beneath min equity
            min_equity = float(eq_out['equity'].min()) if not eq_out.empty else INIT_EQUITY
            y_min = (int(min_equity) // 10_000) * 10_000
            if y_min >= min_equity:
                y_min = max(0, y_min - 10_000)
            chart.set_y_axis({'name': 'Equity ($)', 'min': y_min})

            ws.insert_chart('D2', chart)
        else:
            pd.DataFrame({"time": [], "equity": []}).to_excel(writer, sheet_name="EquityCurve", index=False)


# =========================
# MAIN
# =========================
def main():
    log("Downloading candles from OANDA…")
    df = oanda_candles(INSTRUMENT, GRANULARITY, DATE_FROM, DATE_TO, OANDA_TOKEN)
    if df.empty:
        log("No data received. Check token/instrument/dates.")
        return

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    log(f"Got {len(df)} candles. Running backtest…")

    params = build_params_dict()

    trades_df, eq_df, _ = backtest(df)
    summary_df = compute_summary(trades_df, eq_df, params)

    # Terminal prints — parameters first
    log("========== PARAMETERS ==========")
    for k, v in params.items():
        log(f"{k}: {v}")

    log("========== SUMMARY ==========")
    # Skip echoing the parameter header row (it's already printed)
    for _, row in summary_df.iterrows():
        if row["Metric"] and row["Metric"] not in ("— PARAMETERS —", ""):
            if row["Metric"] == "— SUMMARY —":
                continue
            log(f"{row['Metric']}: {row['Value']}")

    out_path = os.path.join(OUT_DIR, OUT_BASENAME)
    write_xlsx(trades_df, summary_df, eq_df, out_path)
    log(f"Report written → {out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[ERROR] {e}")
        raise
