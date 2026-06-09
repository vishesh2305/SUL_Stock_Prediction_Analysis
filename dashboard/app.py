"""SUL Stock Prediction — Interactive Dashboard
Hero / landing page.

Run locally:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of where streamlit is launched from
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json

import streamlit as st

from dashboard.utils.data_loader import PRIMARY_STOCKS
from dashboard.utils.models import load_bundle

st.set_page_config(
    page_title="SUL Stock Prediction — Interactive ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-title  { font-size: 2.6rem; font-weight: 800; margin-bottom: 0.2rem; }
    .hero-sub    { font-size: 1.1rem; color: #B6BAC3; margin-bottom: 1.4rem; }
    .pill        { display:inline-block; padding: 4px 10px; border-radius: 999px;
                   background:#1F2A40; color:#FF8C00; font-size:0.85rem; margin-right:6px; }
    .metric-card { background:#1A1F2E; border:1px solid #2A3145; border-radius:10px;
                   padding:14px 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Hero --------------------------------------------------------------------
st.markdown('<div class="hero-title">📈 SUL Stock Prediction — Interactive ML Showcase</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A live walkthrough of a multi-stock ML pipeline: 13 engineered features, '
    '5 classifiers, time-series CV, and threshold-tuned predictions — all reproduced from the original notebooks.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div>'
    '<span class="pill">Python</span>'
    '<span class="pill">scikit-learn</span>'
    '<span class="pill">Streamlit</span>'
    '<span class="pill">Plotly</span>'
    '<span class="pill">Time-Series ML</span>'
    '<span class="pill">Technical Indicators</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()


# ---- KPI strip ---------------------------------------------------------------
def _best_metric_across_stocks():
    best_acc = ("?", "?", 0.0)
    best_auc = ("?", "?", 0.0)
    stocks_loaded = 0
    for sym in PRIMARY_STOCKS:
        b = load_bundle(sym, horizon=1)
        if b is None:
            continue
        stocks_loaded += 1
        _, meta = b
        for mname, m in meta["metrics"].items():
            if m["accuracy"] > best_acc[2]:
                best_acc = (sym, mname, m["accuracy"])
            if m.get("roc_auc", 0) and m["roc_auc"] > best_auc[2]:
                best_auc = (sym, mname, m["roc_auc"])
    return stocks_loaded, best_acc, best_auc


stocks_loaded, best_acc, best_auc = _best_metric_across_stocks()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks modeled", stocks_loaded, help="Pre-trained pipelines available in this dashboard")
c2.metric("Features engineered", "13", help="Technical + cross-asset features")
c3.metric("Best accuracy", f"{best_acc[2]:.2%}", f"{best_acc[0]} · {best_acc[1]}")
c4.metric("Best ROC-AUC", f"{best_auc[2]:.3f}", f"{best_auc[0]} · {best_auc[1]}")

st.divider()


# ---- What the dashboard contains --------------------------------------------
st.subheader("What's inside")

col_a, col_b = st.columns([1, 1])

with col_a:
    st.markdown(
        """
        **1. Data Exploration** — interactive candlestick charts, returns
        distribution, and volume profiles for every stock.

        **2. Feature Engineering** — toggle technical indicators (RSI, MACD,
        Bollinger, ATR) overlaid on price; the formula behind each one is shown
        right next to the chart.

        **3. Model Comparison** — five classifiers benchmarked on the same
        chronological 80/20 split: accuracy, ROC-AUC, confusion matrices, ROC curves.
        """
    )
with col_b:
    st.markdown(
        """
        **4. Live Prediction** — pick any historical date and any model;
        the dashboard reconstructs the feature vector and shows the up/down
        probability the model would have produced.

        **5. Code Walkthrough** — the exact notebook cells, grouped by phase
        (data load → indicators → split → train → evaluate), rendered with
        syntax highlighting.
        """
    )


st.divider()


# ---- The headline result -----------------------------------------------------
st.subheader("Headline results")

rows = []
for sym in PRIMARY_STOCKS:
    b = load_bundle(sym, horizon=1)
    if b is None:
        continue
    _, meta = b
    # Pick the best AUC model per stock
    best_name, best = max(
        meta["metrics"].items(),
        key=lambda kv: kv[1].get("roc_auc") or 0,
    )
    rows.append({
        "Stock": sym,
        "Best Model": best_name,
        "Accuracy": f"{best['accuracy']:.2%}",
        "ROC-AUC":  f"{(best.get('roc_auc') or 0):.3f}",
        "Test rows": meta["test_size"],
        "Test start": meta["split_date"],
    })

import pandas as pd
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.warning(
        "No artifacts found. Run `python dashboard/train_artifacts.py` once to generate them."
    )


# ---- About / footer ----------------------------------------------------------
st.divider()
st.subheader("About this project")
st.markdown(
    """
    This dashboard is a **read-only companion** to a set of Jupyter notebooks that
    predict next-day and 5-day direction for Indian equities. The notebooks
    themselves are **untouched** — this app re-applies their feature pipeline and
    re-fits the same five classifiers (Logistic Regression, KNN, RBF SVM,
    Decision Tree, Random Forest) so the results you see here are reproducible
    from the same CSV data.

    - **Stocks covered**: TCS, INFY, WIPRO, CUPID, NESTLE
    - **Features**: 13 engineered columns (Nifty cross-asset, MACD, RSI lag,
      moving averages, ATR ratio, returns lags)
    - **Split**: chronological 80/20 (no shuffling — this is a time series)
    - **Targets**: 1-day direction and 5-day direction

    📓 Notebooks and source code live in the GitHub repo linked in the LinkedIn post.
    """
)
