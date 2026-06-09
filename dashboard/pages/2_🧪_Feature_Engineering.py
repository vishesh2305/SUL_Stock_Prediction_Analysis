"""Feature Engineering page — toggle indicators overlaid on price."""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.utils.data_loader import PRIMARY_STOCKS, load_stock, load_nifty
from dashboard.utils.indicators import (
    add_returns, add_moving_averages, add_macd, add_rsi, add_bollinger, add_atr,
)

st.set_page_config(page_title="Feature Engineering", page_icon="🧪", layout="wide")

st.title("🧪 Feature Engineering")
st.caption("Toggle technical indicators on/off. The formula behind each one is shown alongside.")

# ---- Controls ----
ctl, _ = st.columns([1, 3])
with ctl:
    symbol = st.selectbox("Stock", PRIMARY_STOCKS, index=0)

raw = load_stock(symbol)
raw = add_returns(raw)
raw = add_moving_averages(raw)
raw = add_macd(raw)
raw = add_rsi(raw)
raw, bb_u, bb_m, bb_l = add_bollinger(raw)
raw = add_atr(raw)

# Limit display to last ~3 years for clarity
display = raw.tail(750).copy()
bb_u = bb_u.loc[display.index]
bb_m = bb_m.loc[display.index]
bb_l = bb_l.loc[display.index]

st.markdown("### Pick indicators to overlay")
c1, c2, c3, c4, c5 = st.columns(5)
show_ma     = c1.checkbox("Moving Avgs (7/30/90)", True)
show_bb     = c2.checkbox("Bollinger Bands", False)
show_macd   = c3.checkbox("MACD panel", True)
show_rsi    = c4.checkbox("RSI panel", True)
show_atr    = c5.checkbox("ATR ratio panel", False)

# ---- Plot ----
extras = sum([show_macd, show_rsi, show_atr])
row_heights = [0.6 - 0.1*extras] + [0.4 / max(extras, 1)] * extras if extras else [1.0]
fig = make_subplots(
    rows=1 + extras, cols=1, shared_xaxes=True,
    row_heights=row_heights, vertical_spacing=0.04,
)

# Price line
fig.add_trace(go.Scatter(x=display["Date"], y=display["Close"], name="Close",
                         line=dict(color="#E8E8E8", width=2)), row=1, col=1)

if show_ma:
    fig.add_trace(go.Scatter(x=display["Date"], y=display["MA_Week"],
                             name="MA 7", line=dict(color="#FFD27F", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display["Date"], y=display["MA_Month"],
                             name="MA 30", line=dict(color="#FF8C00", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display["Date"], y=display["MA_3Months"],
                             name="MA 90", line=dict(color="#D9534F", width=1.2)), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=display["Date"], y=bb_u, name="BB Upper",
                             line=dict(color="#7FB3D5", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=display["Date"], y=bb_l, name="BB Lower",
                             line=dict(color="#7FB3D5", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(127,179,213,0.10)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=display["Date"], y=bb_m, name="BB Mid",
                             line=dict(color="#5DADE2", width=1, dash="dash")), row=1, col=1)

row = 2
if show_macd:
    fig.add_trace(go.Bar(x=display["Date"], y=display["MACD_Histogram"], name="MACD hist",
                         marker_color=np.where(display["MACD_Histogram"] >= 0, "#26A69A", "#EF5350")),
                  row=row, col=1)
    fig.add_trace(go.Scatter(x=display["Date"], y=display["MACD"], name="MACD",
                             line=dict(color="#FF8C00", width=1.5)), row=row, col=1)
    fig.add_trace(go.Scatter(x=display["Date"], y=display["MACD_Signal"], name="Signal",
                             line=dict(color="#7FB3D5", width=1.5)), row=row, col=1)
    row += 1

if show_rsi:
    fig.add_trace(go.Scatter(x=display["Date"], y=display["RSI"], name="RSI(14)",
                             line=dict(color="#BB86FC", width=1.5)), row=row, col=1)
    fig.add_hline(y=70, line_color="#EF5350", line_dash="dash", row=row, col=1)
    fig.add_hline(y=30, line_color="#26A69A", line_dash="dash", row=row, col=1)
    row += 1

if show_atr:
    fig.add_trace(go.Scatter(x=display["Date"], y=display["ATR_Ratio"], name="ATR ratio",
                             line=dict(color="#FFCA28", width=1.5)), row=row, col=1)
    row += 1

fig.update_layout(height=620, margin=dict(l=0, r=0, t=30, b=0),
                  hovermode="x unified", legend=dict(orientation="h", y=1.06))
st.plotly_chart(fig, use_container_width=True)

# ---- Formula explainers ----
st.divider()
st.subheader("The formula behind each indicator")

tabs = st.tabs(["Moving Averages", "MACD", "RSI", "Bollinger", "ATR", "Nifty cross-asset"])

with tabs[0]:
    st.code(
        '''df["MA_Week"]    = df["Close"].rolling(window=7).mean()
df["MA_Month"]   = df["Close"].rolling(window=30).mean()
df["MA_3Months"] = df["Close"].rolling(window=90).mean()
df["MA_Week_Ratio"]    = df["Close"] / df["MA_Week"]
df["MA_Month_Ratio"]   = df["Close"] / df["MA_Month"]
df["MA_3Months_Ratio"] = df["Close"] / df["MA_3Months"]''',
        language="python",
    )
    st.markdown("Ratios > 1 → price above its trend. Used as a feature so the model sees the **trend regime**, not the absolute price.")

with tabs[1]:
    st.code(
        '''ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]           = ema_12 - ema_26
df["MACD_Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]''',
        language="python",
    )
    st.markdown("MACD crossing its signal line → momentum shift. The histogram (difference) is the strongest feature of the three for the classifier.")

with tabs[2]:
    st.code(
        '''delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(span=14, adjust=False).mean()
avg_loss = loss.ewm(span=14, adjust=False).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))''',
        language="python",
    )
    st.markdown("Uses Wilder's EMA smoothing (not SMA). >70 = overbought; <30 = oversold. We also lag it (`RSI_Lag1`) so the model only ever sees yesterday's value.")

with tabs[3]:
    st.code(
        '''BB_Middle = df["Close"].rolling(window=20).mean()
BB_Std    = df["Close"].rolling(window=20).std()
BB_Upper  = BB_Middle + 2 * BB_Std
BB_Lower  = BB_Middle - 2 * BB_Std
df["BB_Position"] = (df["Close"] - BB_Middle) / (2 * BB_Std)''',
        language="python",
    )
    st.markdown("`BB_Position` ∈ [-1, +1] roughly. It encodes both **where** the price sits in its volatility envelope and **how wide** the envelope is.")

with tabs[4]:
    st.code(
        '''prev_close = df["Close"].shift(1)
tr1 = df["High"] - df["Low"]
tr2 = (df["High"] - prev_close).abs()
tr3 = (df["Low"]  - prev_close).abs()
true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["ATR_Ratio"] = true_range.rolling(window=14).mean() / df["Close"]''',
        language="python",
    )
    st.markdown("ATR scaled by price → comparable across stocks at different price levels. High ATR ratio = stock is in a volatile regime.")

with tabs[5]:
    st.code(
        '''# Same pipeline applied to NIFTY50, then merged on Date
df["Nifty_Daily_Return"] = nifty["Close"].pct_change()
df["Nifty_Return_Lag1"]  = df["Nifty_Daily_Return"].shift(1)
df["Nifty_RSI"]          = <Wilder RSI on Nifty close>
df["TCS_vs_Nifty_Spread"] = df["Daily_Returns"] - df["Nifty_Daily_Return"]''',
        language="python",
    )
    st.markdown("These cross-asset features were the **biggest single-cell accuracy bump** in the notebook — single stocks are noisy, but the market's regime carries signal.")
