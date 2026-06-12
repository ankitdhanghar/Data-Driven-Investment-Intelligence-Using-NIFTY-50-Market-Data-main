"""
indicators.py  —  Add technical indicators to stock data.

Main function: add_all_indicators(df) adds everything at once.
"""

import numpy as np
import pandas as pd


def add_all_indicators(df):
    """
    Add all technical indicators to a stock DataFrame in one shot.
    Pass in a single-stock DataFrame (from load_stock) and get it back
    with ~25 new columns.
    """
    close = df["Close"]

    # ── Returns ────────────────────────────────────────────────────
    df["Returns"]     = close.pct_change()
    df["Log_Returns"] = np.log(close / close.shift(1))

    # ── Simple Moving Averages ─────────────────────────────────────
    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    # ── Exponential Moving Averages ────────────────────────────────
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # ── Price vs Moving Averages (%) ───────────────────────────────
    df["Price_vs_SMA20"]  = (close - df["SMA_20"])  / df["SMA_20"]  * 100
    df["Price_vs_SMA50"]  = (close - df["SMA_50"])  / df["SMA_50"]  * 100
    df["Price_vs_SMA200"] = (close - df["SMA_200"]) / df["SMA_200"] * 100

    # ── RSI (14-day) ───────────────────────────────────────────────
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ───────────────────────────────────────────────────────
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── Bollinger Bands (20-day, 2 std) ────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Upper"]    = bb_mid + 2 * bb_std
    df["BB_Lower"]    = bb_mid - 2 * bb_std
    df["BB_Width"]    = np.where(bb_mid > 0,
                                 (df["BB_Upper"] - df["BB_Lower"]) / bb_mid * 100, 0)
    band_range = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Position"] = np.where(band_range > 0, (close - df["BB_Lower"]) / band_range, 0.5)

    # ── ATR (Average True Range, 14-day) ───────────────────────────
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - close.shift(1)).abs()
    low_close  = (df["Low"]  - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # ── Volatility (20-day, annualized) ────────────────────────────
    df["Volatility_20"] = df["Returns"].rolling(20).std() * np.sqrt(252)

    # ── Volume Ratio (vs 20-day average) ───────────────────────────
    if "Volume" in df.columns:
        vol_avg = df["Volume"].rolling(20).mean().replace(0, np.nan)
        df["Volume_Ratio"] = df["Volume"] / vol_avg
    else:
        df["Volume_Ratio"] = 1.0

    # ── Lag features ───────────────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"Returns_Lag{lag}"] = df["Returns"].shift(lag)

    # ── Calendar features ──────────────────────────────────────────
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"]     = df.index.month

    # ── Target: did price go UP next day? ──────────────────────────
    df["Target"] = (close.shift(-1) > close).astype(int)

    return df
