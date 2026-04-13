"""Quick benchmark for target selection on merged credit rating data.

Compares 3 candidate targets with the same feature pipeline:
- binary_rating
- rating_class
- rating_detail

Usage:
    e:/thesis/.venv/Scripts/python.exe e:/thesis/src/pipelines/benchmark_targets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "processed" / "merged_credit_rating_common.csv"
DEFAULT_REPORT_PATH = ROOT_DIR / "data" / "reports" / "benchmark_targets_report.csv"


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a consistent feature table for all target benchmarks."""
    data = df.copy()

    # Parse date into lightweight calendar features.
    if "rating_date" in data.columns:
        dt = pd.to_datetime(data["rating_date"], errors="coerce")
        data["rating_year"] = dt.dt.year
        data["rating_month"] = dt.dt.month
        data["rating_quarter"] = dt.dt.quarter

    # Avoid direct label leakage by excluding target columns from X.
    drop_cols = {"binary_rating", "rating_class", "rating_detail", "rating_date"}
    features = [c for c in data.columns if c not in drop_cols]
    return data[features]


def collapse_rare_classes(y: pd.Series, min_count: int) -> pd.Series:
    """Collapse very rare classes into '__RARE__' to stabilize split/training."""
    vc = y.value_counts(dropna=False)
    rare = set(vc[vc < min_count].index)
    if not rare:
        return y.astype("string")

    y_out = y.astype("string").copy()
    y_out = y_out.where(~y_out.isin(rare), "__RARE__")
    return y_out


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a fast and robust mixed-type classification pipeline."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ohe",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=10,
                        ),
                    ),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1500,
        class_weight=None,
        random_state=42,
    )

    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])


def make_ordinal_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a fast ordinal pipeline (regression over ordered class index)."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ohe",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            min_frequency=10,
                        ),
                    ),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=1.0, random_state=42)

    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])


def get_ordered_labels(target_col: str, observed_labels: pd.Series) -> List[str]:
    """Return ordered labels for ordinal evaluation."""
    rating_class_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
    rating_detail_order = [
        "AAA",
        "AA+",
        "AA",
        "AA-",
        "A+",
        "A",
        "A-",
        "BBB+",
        "BBB",
        "BBB-",
        "BB+",
        "BB",
        "BB-",
        "B+",
        "B",
        "B-",
        "CCC+",
        "CCC",
        "CCC-",
        "CC+",
        "CC",
        "C",
        "D",
    ]

    observed = set(observed_labels.astype("string").tolist())
    if target_col == "rating_class":
        return [x for x in rating_class_order if x in observed]
    if target_col == "rating_detail":
        return [x for x in rating_detail_order if x in observed]

    # binary fallback
    if target_col == "binary_rating":
        return [x for x in ["1", "0"] if x in observed]

    return sorted(observed)


def run_one_target(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    seed: int,
    rare_threshold: int,
) -> Dict[str, float]:
    """Run benchmark for a single target and return metrics."""
    work = df.copy()
    y = work[target_col]

    # Ensure target is string for multi-class consistency.
    y = y.astype("string")

    # rating_detail is highly imbalanced in this dataset.
    if target_col == "rating_detail":
        y = collapse_rare_classes(y, min_count=rare_threshold)
        # If the grouped rare bucket is still too small, drop it for a stable split.
        vc_tmp = y.value_counts()
        if "__RARE__" in vc_tmp.index and vc_tmp["__RARE__"] < 2:
            y = y[y != "__RARE__"]
            work = work.loc[y.index]

    mask = y.notna()
    X = build_feature_table(work.loc[mask])
    y = y.loc[mask]

    class_counts = y.value_counts()
    can_stratify = bool((class_counts >= 2).all())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if can_stratify else None,
    )

    pipe = make_pipeline(X_train)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    result = {
        "mode": "multiclass",
        "target": target_col,
        "rows": float(len(y)),
        "classes": float(y.nunique()),
        "min_class_count": float(class_counts.min()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }

    # Notch-distance confusion for ordered labels.
    if target_col in {"rating_class", "rating_detail"}:
        ordered_labels = get_ordered_labels(target_col, y)
        label_to_rank = {label: idx for idx, label in enumerate(ordered_labels)}
        y_test_rank = y_test.map(label_to_rank)
        pred_rank = pd.Series(pred, index=y_test.index).map(label_to_rank)
        valid = y_test_rank.notna() & pred_rank.notna()
        if valid.any():
            diff = np.abs(y_test_rank[valid].astype(int).values - pred_rank[valid].astype(int).values)
            result["notch_exact"] = float((diff == 0).mean())
            result["notch_off_1"] = float((diff == 1).mean())
            result["notch_off_ge_2"] = float((diff >= 2).mean())

    return result


def run_one_target_ordinal(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    seed: int,
    rare_threshold: int,
) -> Dict[str, float]:
    """Run ordinal benchmark by predicting rank index with regression."""
    work = df.copy()
    y = work[target_col].astype("string")

    if target_col == "rating_detail":
        y = collapse_rare_classes(y, min_count=rare_threshold)
        vc_tmp = y.value_counts()
        if "__RARE__" in vc_tmp.index and vc_tmp["__RARE__"] < 2:
            y = y[y != "__RARE__"]
            work = work.loc[y.index]

    mask = y.notna()
    X = build_feature_table(work.loc[mask])
    y = y.loc[mask]

    ordered_labels = get_ordered_labels(target_col, y)
    label_to_rank = {label: idx for idx, label in enumerate(ordered_labels)}

    y_rank = y.map(label_to_rank)
    valid = y_rank.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    y_rank = y_rank.loc[valid].astype(int)

    class_counts = y.value_counts()
    can_stratify = bool((class_counts >= 2).all())

    X_train, X_test, y_train_rank, y_test_rank, y_train_lbl, y_test_lbl = train_test_split(
        X,
        y_rank,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if can_stratify else None,
    )

    pipe = make_ordinal_pipeline(X_train)
    pipe.fit(X_train, y_train_rank)

    pred_rank_cont = pipe.predict(X_test)
    pred_rank = np.rint(pred_rank_cont).astype(int)
    pred_rank = np.clip(pred_rank, 0, len(ordered_labels) - 1)
    pred_lbl = pd.Series(pred_rank, index=y_test_lbl.index).map({i: c for i, c in enumerate(ordered_labels)})

    abs_err = np.abs(y_test_rank.values - pred_rank)

    result = {
        "mode": "ordinal",
        "target": target_col,
        "rows": float(len(y)),
        "classes": float(y.nunique()),
        "min_class_count": float(class_counts.min()),
        "accuracy": float(accuracy_score(y_test_lbl, pred_lbl)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_lbl, pred_lbl)),
        "f1_macro": float(f1_score(y_test_lbl, pred_lbl, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_lbl, pred_lbl, average="weighted", zero_division=0)),
        "mae_rank": float(mean_absolute_error(y_test_rank, pred_rank)),
        "within_1_notch": float((abs_err <= 1).mean()),
        "qwk": float(cohen_kappa_score(y_test_rank, pred_rank, weights="quadratic")),
        "notch_exact": float((abs_err == 0).mean()),
        "notch_off_1": float((abs_err == 1).mean()),
        "notch_off_ge_2": float((abs_err >= 2).mean()),
    }
    return result


def run_one_target_ordinal_cv(
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    rare_threshold: int,
    n_splits: int,
    n_repeats: int,
) -> Dict[str, float]:
    """Run repeated stratified CV for ordinal benchmark and aggregate metrics."""
    work = df.copy()
    y = work[target_col].astype("string")

    if target_col == "rating_detail":
        y = collapse_rare_classes(y, min_count=rare_threshold)
        vc_tmp = y.value_counts()
        if "__RARE__" in vc_tmp.index and vc_tmp["__RARE__"] < 2:
            y = y[y != "__RARE__"]
            work = work.loc[y.index]

    mask = y.notna()
    X = build_feature_table(work.loc[mask])
    y = y.loc[mask]

    ordered_labels = get_ordered_labels(target_col, y)
    label_to_rank = {label: idx for idx, label in enumerate(ordered_labels)}

    y_rank = y.map(label_to_rank)
    valid = y_rank.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    y_rank = y_rank.loc[valid].astype(int)

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    fold_metrics: List[Dict[str, float]] = []
    total = 0
    exact = 0
    off1 = 0
    off2 = 0

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_rank, y_test_rank = y_rank.iloc[train_idx], y_rank.iloc[test_idx]
        y_test_lbl = y.iloc[test_idx]

        pipe = make_ordinal_pipeline(X_train)
        pipe.fit(X_train, y_train_rank)

        pred_rank_cont = pipe.predict(X_test)
        pred_rank = np.rint(pred_rank_cont).astype(int)
        pred_rank = np.clip(pred_rank, 0, len(ordered_labels) - 1)
        pred_lbl = pd.Series(pred_rank, index=y_test_lbl.index).map({i: c for i, c in enumerate(ordered_labels)})

        abs_err = np.abs(y_test_rank.values - pred_rank)

        fold_metrics.append({
            "accuracy": float(accuracy_score(y_test_lbl, pred_lbl)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_lbl, pred_lbl)),
            "f1_macro": float(f1_score(y_test_lbl, pred_lbl, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test_lbl, pred_lbl, average="weighted", zero_division=0)),
            "mae_rank": float(mean_absolute_error(y_test_rank, pred_rank)),
            "within_1_notch": float((abs_err <= 1).mean()),
            "qwk": float(cohen_kappa_score(y_test_rank, pred_rank, weights="quadratic")),
        })

        total += abs_err.size
        exact += int((abs_err == 0).sum())
        off1 += int((abs_err == 1).sum())
        off2 += int((abs_err >= 2).sum())

    fm = pd.DataFrame(fold_metrics)
    result = {
        "mode": "ordinal_cv",
        "target": target_col,
        "rows": float(len(y)),
        "classes": float(y.nunique()),
        "min_class_count": float(y.value_counts().min()),
        "cv_splits": float(n_splits),
        "cv_repeats": float(n_repeats),
        "accuracy": float(fm["accuracy"].mean()),
        "balanced_accuracy": float(fm["balanced_accuracy"].mean()),
        "f1_macro": float(fm["f1_macro"].mean()),
        "f1_weighted": float(fm["f1_weighted"].mean()),
        "mae_rank": float(fm["mae_rank"].mean()),
        "within_1_notch": float(fm["within_1_notch"].mean()),
        "qwk": float(fm["qwk"].mean()),
        "accuracy_std": float(fm["accuracy"].std(ddof=0)),
        "balanced_accuracy_std": float(fm["balanced_accuracy"].std(ddof=0)),
        "f1_macro_std": float(fm["f1_macro"].std(ddof=0)),
        "mae_rank_std": float(fm["mae_rank"].std(ddof=0)),
        "qwk_std": float(fm["qwk"].std(ddof=0)),
        "notch_exact": float(exact / total) if total else np.nan,
        "notch_off_1": float(off1 / total) if total else np.nan,
        "notch_off_ge_2": float(off2 / total) if total else np.nan,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick benchmark for target selection")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to merged dataset CSV",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=20,
        help="Min samples per class for rating_detail; lower classes are grouped into __RARE__",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="CSV report output path",
    )
    parser.add_argument(
        "--ordinal",
        action="store_true",
        help="Also run ordinal benchmark for ordered targets",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run RepeatedStratifiedKFold for ordinal targets",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    df = pd.read_csv(data_path)

    targets: List[str] = ["binary_rating", "rating_class", "rating_detail"]
    results = []
    for target in targets:
        if target not in df.columns:
            print(f"[WARN] Missing target column: {target}")
            continue

        metrics = run_one_target(
            df=df,
            target_col=target,
            test_size=args.test_size,
            seed=args.seed,
            rare_threshold=args.rare_threshold,
        )
        results.append(metrics)

    if args.ordinal:
        for target in ["rating_class", "rating_detail"]:
            if target not in df.columns:
                continue
            metrics = run_one_target_ordinal(
                df=df,
                target_col=target,
                test_size=args.test_size,
                seed=args.seed,
                rare_threshold=args.rare_threshold,
            )
            results.append(metrics)

    if args.ordinal and args.cv:
        for target in ["rating_class", "rating_detail"]:
            if target not in df.columns:
                continue
            metrics = run_one_target_ordinal_cv(
                df=df,
                target_col=target,
                seed=args.seed,
                rare_threshold=args.rare_threshold,
                n_splits=args.cv_splits,
                n_repeats=args.cv_repeats,
            )
            results.append(metrics)

    if not results:
        raise RuntimeError("No benchmark results produced. Check target columns.")

    report = pd.DataFrame(results)
    report = report.sort_values(by=["mode", "f1_macro", "balanced_accuracy"], ascending=[True, False, False])

    print("\n=== Quick Benchmark: Target Comparison ===")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
