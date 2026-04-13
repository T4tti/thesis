"""
model/trainer.py — Train LightGBM credit-rating classifier at startup.

Strategy:
    - 12 shared financial features → 3-class target (IG / HY / Distressed)
    - Pipeline: SimpleImputer (median) → LGBMClassifier (class_weight='balanced')
    - Stratified 5-Fold CV to report generalisation metrics
    - Final fit on full dataset; model kept in-memory

How to Run (standalone test):
    python -m model.trainer

Expected Output:
    [Trainer] Loaded 8680 rows, 3 unique ratings.
    [Trainer] CV F1-weighted : 0.xxxx ± 0.xxxx
    [Trainer] Training complete.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES: list[str] = [
    "current_ratio",
    "debt_equity_ratio",
    "gross_profit_margin",
    "operating_profit_margin",
    "ebit_margin",
    "pretax_profit_margin",
    "net_profit_margin",
    "asset_turnover",
    "roe",
    "roa",
    "operating_cashflow_ps",
    "free_cashflow_ps",
]

TARGET: str = "rating_detail"
RATING_LABELS: list[str] = ["IG", "HY", "Distressed"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(
    data_path: Path,
) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
    """Load data, train LightGBM pipeline, return (pipeline, encoder, metrics).

    Args:
        data_path: Path to merged_credit_rating_common_3groups.csv.

    Returns:
        pipeline     : Fitted sklearn Pipeline ready for predict / predict_proba.
        label_encoder: Fitted LabelEncoder (classes_ maps int → rating string).
        metrics      : Dict with CV scores and dataset statistics.
    """
    # ── 1. Load & validate ─────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    log.info("[Trainer] Loaded %d rows from %s", len(df), data_path.name)

    # Keep only rows with valid target
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()

    # Only keep the 3 expected label values
    df = df[df[TARGET].isin(RATING_LABELS)]
    log.info("[Trainer] %d rows after label filtering (%s).", len(df), RATING_LABELS)

    # Ensure all feature columns exist (fill missing with NaN)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    X = df[FEATURES].astype(float)
    y_raw = df[TARGET]

    # ── 2. Label encode ────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(sorted(RATING_LABELS))   # Alphabetical: ["Distressed","HY","IG"] → 0,1,2
    y = le.transform(y_raw)

    class_dist: Dict[str, int] = y_raw.value_counts().to_dict()
    log.info("[Trainer] Class distribution: %s", class_dist)

    # ── 3. Build pipeline ──────────────────────────────────────────────────
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", lgb.LGBMClassifier(
            n_estimators=600,
            max_depth=7,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])

    # ── 4. Stratified 5-Fold CV ────────────────────────────────────────────
    log.info("[Trainer] Running 5-fold stratified CV …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"],
        return_train_score=False,
        n_jobs=-1,
    )
    metrics: Dict[str, Any] = {
        "cv_accuracy_mean":          float(np.mean(cv_res["test_accuracy"])),
        "cv_accuracy_std":           float(np.std(cv_res["test_accuracy"])),
        "cv_balanced_accuracy_mean": float(np.mean(cv_res["test_balanced_accuracy"])),
        "cv_f1_weighted_mean":       float(np.mean(cv_res["test_f1_weighted"])),
        "cv_f1_weighted_std":        float(np.std(cv_res["test_f1_weighted"])),
        "cv_f1_macro_mean":          float(np.mean(cv_res["test_f1_macro"])),
        "n_samples":                 int(len(X)),
        "n_features":                len(FEATURES),
        "class_distribution":        class_dist,
        "label_classes":             list(le.classes_),
    }
    log.info(
        "[Trainer] CV  accuracy=%.4f±%.4f  f1_weighted=%.4f±%.4f  f1_macro=%.4f",
        metrics["cv_accuracy_mean"],
        metrics["cv_accuracy_std"],
        metrics["cv_f1_weighted_mean"],
        metrics["cv_f1_weighted_std"],
        metrics["cv_f1_macro_mean"],
    )

    # ── 5. Final fit on full data ──────────────────────────────────────────
    log.info("[Trainer] Fitting final model on full dataset …")
    pipeline.fit(X, y)
    log.info("[Trainer] Training complete. Model ready.")

    return pipeline, le, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DATA_PATH = Path(__file__).parents[1] / "data" / "merged_credit_rating_common_3groups.csv"
    pipe, le, m = train_model(DATA_PATH)
    print("\nMetrics:", m)
