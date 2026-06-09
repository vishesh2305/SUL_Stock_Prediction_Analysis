"""Read-only nbformat extractor — pulls code cells from notebooks for display.

We never write back. This is purely so the dashboard's Code Walkthrough page
stays in sync with whatever the notebook currently says.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import re

import nbformat
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS = {
    "TCS":    PROJECT_ROOT / "IT_Notebook_TCS.ipynb",
    "INFY":   PROJECT_ROOT / "IT_Notebook_INFY.ipynb",
    "WIPRO":  PROJECT_ROOT / "IT_Notebook_WIPRO.ipynb",
    "CUPID":  PROJECT_ROOT / "IT_Notebook_CUPID.ipynb",
    "NESTLE": PROJECT_ROOT / "IT_Notebook_NESTLE.ipynb",
}


@st.cache_data(show_spinner=False)
def load_code_cells(symbol: str) -> List[Dict[str, str]]:
    path = NOTEBOOKS.get(symbol)
    if path is None or not path.exists():
        return []
    nb = nbformat.read(str(path), as_version=4)
    cells = []
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            src = (c.source or "").strip()
            if not src:
                continue
            cells.append({"index": i, "source": src})
    return cells


_SECTION_PATTERNS = [
    ("Data load & cleaning", re.compile(r"read_csv|to_datetime|sort_values|astype\(float\)", re.I)),
    ("Returns & volume",     re.compile(r"Daily_Returns|Volume_Ratio|pct_change", re.I)),
    ("Moving averages",      re.compile(r"MA_Week|MA_Month|MA_3Months", re.I)),
    ("MACD",                 re.compile(r"MACD", re.I)),
    ("RSI",                  re.compile(r"\bRSI\b|gain.*loss|ewm\(span=14", re.I)),
    ("Bollinger Bands",      re.compile(r"BB_Position|Bollinger|BB_Upper", re.I)),
    ("ATR",                  re.compile(r"ATR_Ratio|true_range", re.I)),
    ("Nifty merge",          re.compile(r"Nifty|df_nifty", re.I)),
    ("Target & lags",        re.compile(r"Target|Return_Lag|Day_of_Week|Next_Day_Return", re.I)),
    ("Train/test split",     re.compile(r"split_index|iloc\[0:split_index", re.I)),
    ("Feature ranking",      re.compile(r"mutual_info_classif|MI_Score|RandomForest.*feature_importances", re.I)),
    ("Scaling",              re.compile(r"StandardScaler|fit_transform", re.I)),
    ("Logistic Regression",  re.compile(r"LogisticRegression", re.I)),
    ("KNN",                  re.compile(r"KNeighborsClassifier", re.I)),
    ("SVM",                  re.compile(r"\bSVC\b|kernel=", re.I)),
    ("Decision Tree",        re.compile(r"DecisionTreeClassifier", re.I)),
    ("Random Forest",        re.compile(r"RandomForestClassifier", re.I)),
    ("Threshold tuning",     re.compile(r"threshold_grid|best_threshold", re.I)),
    ("Evaluation",           re.compile(r"classification_report|confusion_matrix|roc_auc", re.I)),
    ("PCA / t-SNE",          re.compile(r"\bPCA\b|TSNE", re.I)),
]


def classify(source: str) -> str:
    for name, pat in _SECTION_PATTERNS:
        if pat.search(source):
            return name
    return "Misc"
