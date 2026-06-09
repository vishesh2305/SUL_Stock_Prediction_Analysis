"""Live Prediction page — pick a date + model, see probability of Up."""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import PRIMARY_STOCKS, load_stock, load_nifty
from dashboard.utils.indicators import build_full_dataset, FEATURE_COLUMNS
from dashboard.utils.models import load_bundle

st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Live Prediction")
st.caption("Pick a stock, model, and historical date — the dashboard reconstructs the "
           "feature vector and shows the up/down probability the model would have produced.")

# ---- Controls ----
c1, c2, c3 = st.columns(3)
with c1:
    symbol = st.selectbox("Stock", PRIMARY_STOCKS, index=0)
with c2:
    horizon = st.radio("Horizon", [1, 5], horizontal=True,
                       format_func=lambda h: f"{h}-day")
with c3:
    bundle = load_bundle(symbol, horizon=horizon)
    if bundle is None:
        st.error("Artifacts missing. Run train_artifacts.py.")
        st.stop()
    artifact, meta = bundle
    model_name = st.selectbox("Model", list(artifact["models"].keys()))

# Build the feature dataset and restrict to test-set dates only (no peeking at train)
nifty = load_nifty()
stock = load_stock(symbol)
df_full = build_full_dataset(stock, nifty, horizon=horizon)
test_dates = pd.to_datetime(meta["test_dates"])
df_test = df_full[df_full["Date"].isin(test_dates)].reset_index(drop=True)

st.markdown("##### Pick a date from the held-out test set")
date_pick = st.select_slider(
    "Date",
    options=df_test["Date"].dt.date.tolist(),
    value=df_test["Date"].dt.date.iloc[len(df_test) // 2],
)

row = df_test[df_test["Date"].dt.date == date_pick].iloc[0]
x = row[FEATURE_COLUMNS].values.reshape(1, -1)
x_scaled = artifact["scaler"].transform(x)

model = artifact["models"][model_name]
try:
    prob_up = float(model.predict_proba(x_scaled)[0, 1])
except Exception:
    prob_up = float(model.predict(x_scaled)[0])
prediction = 1 if prob_up >= 0.5 else 0
truth = int(row["Target"])

# ---- Output ----
col_l, col_r = st.columns([1, 1.4])

with col_l:
    st.markdown(f"### {symbol} · {date_pick} · {horizon}-day direction")
    st.markdown(f"**Model:** {model_name}")

    # Gauge for probability
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_up * 100,
        number={"suffix": "%", "font": {"size": 38}},
        title={"text": "Probability of <b>Up</b>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FF8C00"},
            "steps": [
                {"range": [0, 50],  "color": "#2A1F1F"},
                {"range": [50, 100], "color": "#1F2A2A"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.85, "value": 50},
        },
    ))
    gauge.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(gauge, use_container_width=True)

    verdict_pred = "📈 UP" if prediction == 1 else "📉 DOWN"
    verdict_true = "📈 UP" if truth == 1 else "📉 DOWN"
    correct = "✅ Correct" if prediction == truth else "❌ Wrong"
    st.markdown(f"**Prediction:** {verdict_pred} &nbsp;·&nbsp; **Actual:** {verdict_true} &nbsp;·&nbsp; {correct}")

with col_r:
    st.markdown("### Feature snapshot used for this prediction")
    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Raw value": x.ravel(),
        "Scaled value (z-score)": x_scaled.ravel(),
    })
    st.dataframe(
        feat_df.style.format({"Raw value": "{:.4f}", "Scaled value (z-score)": "{:+.2f}"}),
        use_container_width=True, hide_index=True, height=480,
    )

st.divider()

# ---- Context: price around this date ----
st.subheader("Price action around this date")
ctx_window = 30
ctx_dates = pd.date_range(end=pd.Timestamp(date_pick), periods=ctx_window + 5, freq="D")
ctx = stock[(stock["Date"] >= ctx_dates.min()) & (stock["Date"] <= pd.Timestamp(date_pick) + pd.Timedelta(days=horizon + 2))]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=ctx["Date"], open=ctx["Open"], high=ctx["High"], low=ctx["Low"], close=ctx["Close"],
    increasing_line_color="#26A69A", decreasing_line_color="#EF5350", name="OHLC",
))
fig.add_vline(x=pd.Timestamp(date_pick), line_color="#FF8C00", line_dash="dash",
              annotation_text="Prediction date")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
                  xaxis_rangeslider_visible=False, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "These predictions are made on **held-out test data the model never saw during training**. "
    "All features are computed from prior days only — no leakage of the answer."
)
