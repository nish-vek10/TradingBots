import pandas as pd
import numpy as np
from datetime import datetime
from pytz import timezone
import requests

"""""
~ TR = Mean True Range per bar in USD. Higher = more intrabar movement.
~ ATR = Mean Average True Range over your chosen ATR period (14 bars). This smooths TR to show sustained volatility.
~ range = Mean high–low range per bar in USD.

All are averages across all bars in that hour of day over the dataset you pulled.


""""

# --- CONFIG ---
oanda_token = "37ee33b35f88e073a08d533849f7a24b-524c89ef15f36cfe532f0918a6aee4c2"
oanda_api_url = "https://api-fxpractice.oanda.com/v3"
local_tz = timezone('Europe/London')
symbol = "XAU_USD"
granularity = "M5"
num_candles = 5000  # fetch enough history (e.g. ~2–3 weeks)
atr_period = 14

def fetch_oanda_candles(symbol, granularity, count):
    url = f"{oanda_api_url}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {oanda_token}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    raw = resp.json().get("candles", [])
    data = {"time": [], "open": [], "high": [], "low": [], "close": []}
    for c in raw:
        if c.get("complete", False):
            utc = pd.to_datetime(c["time"], utc=True)
            lt = utc.tz_convert(local_tz)
            data["time"].append(lt)
            data["open"].append(float(c["mid"]["o"]))
            data["high"].append(float(c["mid"]["h"]))
            data["low"].append(float(c["mid"]["l"]))
            data["close"].append(float(c["mid"]["c"]))
    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df

# Fetch data
df = fetch_oanda_candles(symbol, granularity, num_candles)

# Compute True Range
df["c_prev"] = df["close"].shift(1)
df["tr1"] = df["high"] - df["low"]
df["tr2"] = (df["high"] - df["c_prev"]).abs()
df["tr3"] = (df["low"] - df["c_prev"]).abs()
df["TR"] = df[["tr1","tr2","tr3"]].max(axis=1)

# Compute ATR (SMA seed, then RMA)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()  # simple for now

# Compute high–low range
df["range"] = df["high"] - df["low"]

# Group by hour-of-day
df["hour"] = df.index.hour
group = df.groupby("hour").agg({
    "TR": "mean",
    "ATR": "mean",
    "range": "mean"
})

print("\n=== Hour-of-Day Volatility Metrics (London Time) ===")
print(group.round(4))

# Optionally: visualize
import matplotlib.pyplot as plt
group.plot(y=["TR", "ATR", "range"], figsize=(10,5), grid=True, title="Volatility by Hour (XAUUSD 5m)")
plt.xlabel("Hour of Day (BST)")
plt.ylabel("Mean Value")
plt.show()
