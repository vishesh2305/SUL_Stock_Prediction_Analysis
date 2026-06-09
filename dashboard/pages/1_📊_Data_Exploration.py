"""Data Exploration page — candlestick, volume, returns distribution."""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.utils.data_loader import PRIMARY_STOCKS, load_stock

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

st.title("📊 Data Exploration")
st.caption("Candlestick + volume + return distribution for every stock in the project.")

# ---- Controls ----
col_l, col_r = st.columns([1, 3])
with col_l:
    symbol = st.selectbox("Stock", PRIMARY_STOCKS, index=0)
    df = load_stock(symbol)
    date_min, date_max = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.date_input(
        "Date range", value=(date_min, date_max),
        min_value=date_min, max_value=date_max,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start, end = pd.Timestamp(date_min), pd.Timestamp(date_max)
    show_volume = st.checkbox("Show volume", True)

with col_r:
    mask = (df["Date"] >= start) & (df["Date"] <= end)
    plot_df = df.loc[mask].copy()

    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25] if show_volume else [1.0],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=plot_df["Date"], open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name=symbol,
            increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
        ),
        row=1, col=1,
    )
    if show_volume:
        fig.add_trace(
            go.Bar(x=plot_df["Date"], y=plot_df["Volume"], name="Volume",
                   marker_color="#5DADE2", opacity=0.6),
            row=2, col=1,
        )
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False, showlegend=False,
        title=f"{symbol} — OHLC & Volume",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Quick stats ----
st.subheader("Snapshot")
plot_df["Daily_Return"] = plot_df["Close"].pct_change()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows in window", f"{len(plot_df):,}")
c2.metric("First close",   f"₹{plot_df['Close'].iloc[0]:,.2f}")
c3.metric("Last close",    f"₹{plot_df['Close'].iloc[-1]:,.2f}",
          f"{(plot_df['Close'].iloc[-1] / plot_df['Close'].iloc[0] - 1):.1%}")
c4.metric("Daily vol (σ)", f"{plot_df['Daily_Return'].std():.4f}")

# ---- Returns distribution ----
st.subheader("Daily returns distribution")
ret = plot_df["Daily_Return"].dropna()
hist = go.Figure()
hist.add_trace(go.Histogram(x=ret, nbinsx=80, marker_color="#FF8C00",
                            opacity=0.85, name="Daily returns"))
hist.add_vline(x=0, line_color="white", line_dash="dash")
hist.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                   xaxis_title="Daily return", yaxis_title="Frequency",
                   bargap=0.02)
st.plotly_chart(hist, use_container_width=True)

with st.expander("Why this matters for the model"):
    st.markdown(
        """
        The model treats this distribution as the source of its target:
        days with positive next-day return are labeled **1**, the rest **0**.
        A fat-tailed, near-zero-mean distribution like this is precisely
        why models hover near 50% accuracy — the **signal-to-noise ratio is low**.
        The feature engineering page shows the indicators we built to claw
        signal back out of this noise.
        """
    )
