"""Trains the same 5 classifiers the notebook trains, saves artifacts.

The notebook stays untouched. This module replicates its modeling logic:
chronological 80/20 split, StandardScaler, then LogReg / KNN / SVM / DT / RF
fit on the 13-feature subset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

from .indicators import FEATURE_COLUMNS, build_full_dataset
from .data_loader import load_stock, load_nifty

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class TrainedBundle:
    symbol: str
    horizon: int
    feature_columns: list
    train_size: int
    test_size: int
    split_date: str
    scaler: StandardScaler
    models: Dict[str, object]
    metrics: Dict[str, dict]
    test_dates: list
    test_close: list
    y_test: list


def _build_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN (k=11)":          KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=6, random_state=42, n_jobs=-1,
        ),
    }


def train_for(symbol: str, horizon: int = 1) -> TrainedBundle:
    nifty = load_nifty()
    stock = load_stock(symbol)
    full = build_full_dataset(stock, nifty, horizon=horizon)

    split = int(len(full) * 0.8)
    train = full.iloc[:split]
    test  = full.iloc[split:]

    X_train = train[FEATURE_COLUMNS].values
    X_test  = test[FEATURE_COLUMNS].values
    y_train = train["Target"].astype(int).values
    y_test  = test["Target"].astype(int).values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = _build_models()
    metrics: Dict[str, dict] = {}

    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        preds = mdl.predict(X_test_s)
        try:
            probs = mdl.predict_proba(X_test_s)[:, 1]
            auc = float(roc_auc_score(y_test, probs))
            fpr, tpr, _ = roc_curve(y_test, probs)
        except Exception:
            probs = preds.astype(float)
            auc = float("nan")
            fpr, tpr = [0.0, 1.0], [0.0, 1.0]

        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc":  auc,
            "confusion_matrix": cm,
            "report": report,
            "fpr": list(map(float, fpr)),
            "tpr": list(map(float, tpr)),
            "probs": list(map(float, probs)),
            "preds": list(map(int, preds)),
        }

    return TrainedBundle(
        symbol=symbol,
        horizon=horizon,
        feature_columns=FEATURE_COLUMNS,
        train_size=len(train),
        test_size=len(test),
        split_date=str(test["Date"].iloc[0].date()),
        scaler=scaler,
        models=models,
        metrics=metrics,
        test_dates=[str(d.date()) for d in test["Date"]],
        test_close=test["Close"].tolist(),
        y_test=y_test.tolist(),
    )


def save_bundle(bundle: TrainedBundle) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{bundle.symbol}_h{bundle.horizon}"

    joblib.dump(
        {"scaler": bundle.scaler, "models": bundle.models, "features": bundle.feature_columns},
        ARTIFACTS_DIR / f"{key}_models.joblib",
    )

    meta = {
        "symbol":  bundle.symbol,
        "horizon": bundle.horizon,
        "feature_columns": bundle.feature_columns,
        "train_size": bundle.train_size,
        "test_size":  bundle.test_size,
        "split_date": bundle.split_date,
        "metrics": bundle.metrics,
        "test_dates": bundle.test_dates,
        "test_close": bundle.test_close,
        "y_test": bundle.y_test,
    }
    (ARTIFACTS_DIR / f"{key}_meta.json").write_text(json.dumps(meta, indent=2))


def load_bundle(symbol: str, horizon: int = 1):
    key = f"{symbol}_h{horizon}"
    models_path = ARTIFACTS_DIR / f"{key}_models.joblib"
    meta_path = ARTIFACTS_DIR / f"{key}_meta.json"
    if not models_path.exists() or not meta_path.exists():
        return None
    artifact = joblib.load(models_path)
    meta = json.loads(meta_path.read_text())
    return artifact, meta
