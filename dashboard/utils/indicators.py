"""Technical indicator formulas — verbatim copies of the notebook logic.

The notebook is the source of truth; this module exists so the dashboard
can compute the same numbers without importing the .ipynb file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(window=window).mean()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["Daily_Returns"] = df["Close"].pct_change()
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["MA_Week"]    = df["Close"].rolling(window=7).mean()
    df["MA_Month"]   = df["Close"].rolling(window=30).mean()
    df["MA_3Months"] = df["Close"].rolling(window=90).mean()
    df["MA_Week_Ratio"]    = df["Close"] / df["MA_Week"]
    df["MA_Month_Ratio"]   = df["Close"] / df["MA_Month"]
    df["MA_3Months_Ratio"] = df["Close"] / df["MA_3Months"]
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["volatility"] = df["Daily_Returns"].rolling(window=window).std()
    return df


def add_rsi(df: pd.DataFrame, span: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=span, adjust=False).mean()
    avg_loss = loss.ewm(span=span, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20):
    bb_mid = df["Close"].rolling(window=window).mean()
    bb_std = df["Close"].rolling(window=window).std()
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    df["BB_Position"] = (df["Close"] - bb_mid) / (2 * bb_std)
    return df, bb_upper, bb_mid, bb_lower


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"]  - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_Ratio"] = true_range.rolling(window=window).mean() / df["Close"]
    return df


def add_nifty_features(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    n = nifty.copy()
    n["Nifty_Daily_Return"] = n["Close"].pct_change()
    n["Nifty_Return_Lag1"]  = n["Nifty_Daily_Return"].shift(1)
    delta_n = n["Close"].diff()
    gain_n = delta_n.clip(lower=0)
    loss_n = -delta_n.clip(upper=0)
    avg_gain_n = gain_n.ewm(span=14, adjust=False).mean()
    avg_loss_n = loss_n.ewm(span=14, adjust=False).mean()
    n["Nifty_RSI"] = 100 - (100 / (1 + avg_gain_n / avg_loss_n))

    keep = n[["Date", "Nifty_Daily_Return", "Nifty_Return_Lag1", "Nifty_RSI"]]
    df = df.merge(keep, on="Date", how="left")
    df["TCS_vs_Nifty_Spread"] = df["Daily_Returns"] - df["Nifty_Daily_Return"]
    return df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if horizon == 1:
        df["Next_Day_Return"] = df["Daily_Returns"].shift(-1)
        df["Target"] = np.nan
        df.loc[df["Next_Day_Return"] > 0, "Target"] = 1
        df.loc[df["Next_Day_Return"] <= 0, "Target"] = 0
    else:
        df["Target"] = (df["Close"].shift(-horizon) > df["Close"]).astype("Int64")
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df["Return_Lag1"] = df["Daily_Returns"].shift(1)
    df["Return_Lag2"] = df["Daily_Returns"].shift(2)
    df["Return_Lag3"] = df["Daily_Returns"].shift(3)
    df["RSI_Lag1"]    = df["RSI"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    return df


# The exact 13-feature set used in the notebook's final comparison
FEATURE_COLUMNS = [
    "Nifty_RSI", "MACD_Signal", "MA_Week", "TCS_vs_Nifty_Spread", "Close",
    "Return_Lag1", "Nifty_Daily_Return", "RSI_Lag1", "ATR_Ratio", "Return_Lag3",
    "Nifty_Return_Lag1", "RSI", "MACD_Histogram",
]


def build_full_dataset(stock_df: pd.DataFrame, nifty_df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Apply the full feature-engineering pipeline as it appears in the notebook."""
    df = stock_df.copy()
    df = add_volume_ratio(df)
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df, _u, _m, _l = add_bollinger(df)
    df = add_atr(df)
    df = add_nifty_features(df, nifty_df)
    df = add_target(df, horizon=horizon)
    df = add_lags(df)
    df = df.dropna().reset_index(drop=True)
    return df
