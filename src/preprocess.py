"""
preprocess.py  —  Load and clean NIFTY-50 stock data.

Functions
---------
load_stock(symbol)       -> single stock DataFrame
load_all_stocks()        -> dict of all stock DataFrames
load_metadata()          -> company name + sector info
get_price_panel()        -> wide DataFrame (dates x stocks) for portfolio work
get_aligned_returns()    -> wide DataFrame of daily returns (aligned dates)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Stocks to skip (INFRATEL has 0 rows)
SKIP = {"NIFTY50_all", "stock_metadata", "INFRATEL"}

# Risk-free rate and trading days (used in risk calculations later)
RISK_FREE_RATE = 0.065
TRADING_DAYS = 252


def load_metadata():
    """Load company names and sectors."""
    meta = pd.read_csv(RAW_DIR / "stock_metadata.csv")
    meta.columns = meta.columns.str.strip()
    meta = meta.rename(columns={"Company Name": "Company_Name"})
    meta["Symbol"] = meta["Symbol"].replace("M&M", "MM")
    return meta[["Symbol", "Company_Name", "Industry"]].set_index("Symbol")


def load_stock(symbol):
    """
    Load one stock CSV → cleaned DataFrame with DatetimeIndex.
    Handles duplicates, forward-fills small gaps, drops bad rows.
    Keeps only: Open, High, Low, Close, VWAP, Prev Close, Volume, Turnover.
    """
    path = RAW_DIR / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip()
    df = df.sort_values("Date").set_index("Date")
    df.index = pd.DatetimeIndex(df.index)

    df = df[~df.index.duplicated(keep="last")]

    # Rename legacy column name before dropping extras
    if "Volume" not in df.columns and "Traded Quantity" in df.columns:
        df = df.rename(columns={"Traded Quantity": "Volume"})

    # Drop non-price metadata columns from the CSV
    drop_cols = ["Symbol", "Series", "Last", "Trades",
                 "Deliverable Volume", "%Deliverble"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Coerce price / volume columns to numeric
    for col in ["Open", "High", "Low", "Close", "VWAP", "Prev Close", "Volume", "Turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.ffill(limit=3)
    df = df.dropna(subset=["Close"])

    df["Symbol"] = symbol
    return df


def load_all_stocks():
    """Load all 49 stocks → dict {symbol: DataFrame}."""
    symbols = sorted([p.stem for p in RAW_DIR.glob("*.csv") if p.stem not in SKIP])
    stocks = {}
    for s in symbols:
        try:
            df = load_stock(s)
            if len(df) > 0:
                stocks[s] = df
        except Exception as e:
            print(f"Skipping {s}: {e}")
    return stocks


def get_price_panel(start_date="2011-01-01"):
    """
    Wide DataFrame: rows=dates, columns=stocks, values=Close price.
    Stocks that were not yet listed by `start_date` are excluded.
    Small intra-date gaps (≤5 days) are forward-filled.
    Useful for portfolio and correlation work.
    """
    stocks = load_all_stocks()
    prices = pd.DataFrame({sym: df["Close"] for sym, df in stocks.items()})
    prices = prices.loc[start_date:]
    prices = prices.ffill(limit=5)           # bridge small gaps
    prices = prices.dropna(axis=1, how="any") # drop stocks not listed by start_date
    prices = prices.dropna(how="any")         # drop any remaining incomplete rows
    return prices


def get_aligned_returns(start_date="2011-01-01"):
    """
    Wide DataFrame of daily returns, aligned across all stocks.
    Used for portfolio optimization and correlation analysis.
    """
    prices = get_price_panel(start_date)
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns
