"""Model Comparison page — accuracy, ROC, and confusion matrices."""
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

from dashboard.utils.data_loader import PRIMARY_STOCKS
from dashboard.utils.models import load_bundle

st.set_page_config(page_title="Model Comparison", page_icon="🤖", layout="wide")

st.title("🤖 Model Comparison")
st.caption("Five classifiers, the same 13 features, the same chronological 80/20 split.")

ctl1, ctl2 = st.columns([1, 1])
with ctl1:
    symbol = st.selectbox("Stock", PRIMARY_STOCKS, index=0)
with ctl2:
    horizon = st.radio("Prediction horizon", [1, 5], horizontal=True,
                       format_func=lambda h: f"{h}-day direction")

bundle = load_bundle(symbol, horizon=horizon)
if bundle is None:
    st.error(f"No artifact for {symbol} (h={horizon}). Run `python dashboard/train_artifacts.py`.")
    st.stop()

_, meta = bundle
metrics = meta["metrics"]

# ---- Headline KPIs ----
best_acc_name = max(metrics, key=lambda n: metrics[n]["accuracy"])
best_auc_name = max(metrics, key=lambda n: metrics[n].get("roc_auc") or 0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train rows", f"{meta['train_size']:,}")
c2.metric("Test rows",  f"{meta['test_size']:,}", help=f"Test set starts {meta['split_date']}")
c3.metric("Best accuracy", f"{metrics[best_acc_name]['accuracy']:.2%}", best_acc_name)
c4.metric("Best ROC-AUC",  f"{(metrics[best_auc_name].get('roc_auc') or 0):.3f}", best_auc_name)

st.divider()

# ---- Side-by-side bars: accuracy + auc ----
df_summary = pd.DataFrame([
    {"Model": n, "Accuracy": m["accuracy"], "ROC-AUC": m.get("roc_auc") or 0.0,
     "Precision (Up)": m["report"].get("1", {}).get("precision", 0),
     "Recall (Up)":    m["report"].get("1", {}).get("recall", 0),
     "F1 (Up)":        m["report"].get("1", {}).get("f1-score", 0)}
    for n, m in metrics.items()
])

st.subheader("Headline scores")
fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "ROC-AUC"),
                        horizontal_spacing=0.12)
fig_bar.add_trace(
    go.Bar(x=df_summary["Model"], y=df_summary["Accuracy"], marker_color="#FF8C00"),
    row=1, col=1,
)
fig_bar.add_trace(
    go.Bar(x=df_summary["Model"], y=df_summary["ROC-AUC"], marker_color="#5DADE2"),
    row=1, col=2,
)
fig_bar.add_hline(y=0.5, line_color="red", line_dash="dash",
                  annotation_text="Random baseline", row=1, col=2)
fig_bar.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(
    df_summary.style.format({
        "Accuracy": "{:.2%}", "ROC-AUC": "{:.3f}",
        "Precision (Up)": "{:.2f}", "Recall (Up)": "{:.2f}", "F1 (Up)": "{:.2f}",
    }),
    use_container_width=True, hide_index=True,
)

st.divider()

# ---- ROC curves overlay ----
st.subheader("ROC curves")
colors = ["#FF8C00", "#5DADE2", "#26A69A", "#BB86FC", "#FFCA28"]
fig_roc = go.Figure()
for (name, m), color in zip(metrics.items(), colors):
    fig_roc.add_trace(go.Scatter(
        x=m["fpr"], y=m["tpr"], mode="lines",
        name=f"{name} (AUC = {(m.get('roc_auc') or 0):.3f})",
        line=dict(color=color, width=2),
    ))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(color="red", dash="dash")))
fig_roc.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      legend=dict(orientation="v", x=0.62, y=0.05))
st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# ---- Confusion matrices ----
st.subheader("Confusion matrices (test set)")
cm_cols = st.columns(len(metrics))
for col, (name, m) in zip(cm_cols, metrics.items()):
    cm = np.array(m["confusion_matrix"])
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Pred Down", "Pred Up"], y=["Actual Down", "Actual Up"],
        colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}", textfont={"size": 16},
    ))
    cm_fig.update_layout(title=name, height=300,
                         margin=dict(l=20, r=10, t=40, b=20))
    col.plotly_chart(cm_fig, use_container_width=True)

st.divider()

# ---- Detailed classification report ----
st.subheader("Classification reports")
chosen = st.selectbox("Inspect model", list(metrics.keys()))
report = metrics[chosen]["report"]
rep_rows = []
for k, v in report.items():
    if isinstance(v, dict):
        rep_rows.append({"Label": k, **v})
rep_df = pd.DataFrame(rep_rows)
st.dataframe(rep_df, use_container_width=True, hide_index=True)
