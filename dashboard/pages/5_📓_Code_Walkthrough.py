"""Code Walkthrough — render notebook cells grouped by phase."""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collections import defaultdict
import streamlit as st

from dashboard.utils.notebook_reader import NOTEBOOKS, load_code_cells, classify

st.set_page_config(page_title="Code Walkthrough", page_icon="📓", layout="wide")

st.title("📓 Code Walkthrough")
st.caption("Cells pulled live from the notebook files (read-only). Grouped by pipeline phase.")

ctl1, _ = st.columns([1, 3])
with ctl1:
    available = [s for s, p in NOTEBOOKS.items() if p.exists()]
    if not available:
        st.error("No notebooks found at project root.")
        st.stop()
    symbol = st.selectbox("Notebook", available, index=0)

cells = load_code_cells(symbol)
if not cells:
    st.warning("No code cells found.")
    st.stop()

# Group by classified section
buckets = defaultdict(list)
for c in cells:
    buckets[classify(c["source"])].append(c)

# Preserve a sensible reading order
ORDER = [
    "Data load & cleaning", "Returns & volume", "Moving averages", "MACD", "RSI",
    "Bollinger Bands", "ATR", "Nifty merge", "Target & lags", "Train/test split",
    "Feature ranking", "Scaling", "Logistic Regression", "KNN", "SVM",
    "Decision Tree", "Random Forest", "Threshold tuning", "Evaluation",
    "PCA / t-SNE", "Misc",
]

st.markdown(f"**{len(cells)} code cells** across **{len(buckets)} phases**.")

shown_sections = [s for s in ORDER if s in buckets]

# Quick nav
st.markdown("##### Jump to a phase")
nav_cols = st.columns(4)
for i, s in enumerate(shown_sections):
    nav_cols[i % 4].markdown(f"- {s} ({len(buckets[s])})")

st.divider()

for section in shown_sections:
    cells_in = buckets[section]
    with st.expander(f"**{section}**  ·  {len(cells_in)} cell(s)", expanded=(section in ("Data load & cleaning", "Target & lags"))):
        for c in cells_in:
            st.markdown(f"`cell #{c['index']}`")
            st.code(c["source"], language="python")

st.divider()
st.info(
    "Every cell shown here is read directly from the .ipynb file. "
    "Edit the notebook in Jupyter, refresh this page, and the new code appears — "
    "no copy-paste, no drift."
)
