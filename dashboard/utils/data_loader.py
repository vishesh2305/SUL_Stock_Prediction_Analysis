"""Reads the same yfinance-format CSVs used by the notebooks.

Notebooks are NOT modified. This module mirrors the cleanup logic
(rename 'Price' -> 'Date', drop the two ticker rows, cast types).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data"

STOCKS = {
    "TCS":      DATA_ROOT / "IT_Data" / "TCS.NS_data.csv",
    "INFY":     DATA_ROOT / "IT_Data" / "INFY.NS_data.csv",
    "WIPRO":    DATA_ROOT / "IT_Data" / "WIPRO.NS_data.csv",
    "CUPID":    DATA_ROOT / "CUPID"   / "CUPID.NS_data.csv",
    "NESTLE":   DATA_ROOT / "nestle"  / "NESTLEIND.NS_data.csv",
    "HDFCBANK": DATA_ROOT / "Banking_Data" / "HDFCBANK.NS_data.csv",
    "ICICIBANK":DATA_ROOT / "Banking_Data" / "ICICIBANK.NS_data.csv",
    "SBIN":     DATA_ROOT / "Banking_Data" / "SBIN.NS_data.csv",
    "MARUTI":   DATA_ROOT / "AutoMobile_Data" / "MARUTI.NS_data.csv",
    "M&M":      DATA_ROOT / "AutoMobile_Data" / "M&M.NS_data.csv",
    "TATAMOTORS":DATA_ROOT / "AutoMobile_Data" / "TATAMOTORS.NS_data.csv",
    "CIPLA":    DATA_ROOT / "Pharma_Data" / "CIPLA.NS_data.csv",
    "DRREDDY":  DATA_ROOT / "Pharma_Data" / "DRREDDY.NS_data.csv",
    "SUNPHARMA":DATA_ROOT / "Pharma_Data" / "SUNPHARMA.NS_data.csv",
}

NIFTY_PATH = DATA_ROOT / "Nifty" / "NIFTY50_data.csv"

PRIMARY_STOCKS = ["TCS", "INFY", "WIPRO", "CUPID", "NESTLE"]


def _clean_yf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Price": "Date"})
    df = df.iloc[2:].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ("Close", "High", "Low", "Open"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_stock(symbol: str) -> pd.DataFrame:
    if symbol not in STOCKS:
        raise KeyError(f"Unknown stock {symbol}. Choices: {list(STOCKS)}")
    return _clean_yf(STOCKS[symbol])


@st.cache_data(show_spinner=False)
def load_nifty() -> pd.DataFrame:
    return _clean_yf(NIFTY_PATH)
