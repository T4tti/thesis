"""Benchmark 3 augmented train datasets on rating_class.

So sánh hiệu quả 3 kỹ thuật tăng cường dữ liệu:
- CTGAN
- TabDDPM
- SMOTE

Output:
  - data/reports/benchmark_augmented_train_report.csv

Usage:
  e:/thesis/.venv/Scripts/python.exe e:/thesis/src/pipelines/benchmark_augmented_trains.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipelines.benchmark_targets import build_feature_table, make_pipeline


DEFAULT_CTGan_PATH = ROOT_DIR / "data" / "processed" / "train_augmented_ctgan.csv"
DEFAULT_TABDDPM_PATH = ROOT_DIR / "data" / "processed" / "train_augmented_tabddpm.csv"
DEFAULT_SMOTE_PATH = ROOT_DIR / "data" / "processed" / "train_smote_augmented.csv"
DEFAULT_TIMEGAN_PATH = ROOT_DIR / "data" / "processed" / "train_augmented_timegan.csv"
DEFAULT_OUT_PATH = ROOT_DIR / "data" / "reports" / "benchmark_augmented_train_report.csv"


def run_dataset_benchmark(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    seed: int,
    cv_splits: int,
    cv_repeats: int,
) -> Dict[str, float]:
    work = df.copy()
    if target_col not in work.columns:
        raise ValueError(f"[{dataset_name}] Missing target column: {target_col}")

    y = work[target_col].astype("string")
    mask = y.notna()
    y = y.loc[mask]
    X = build_feature_table(work.loc[mask])

    # Drop all-NaN columns to avoid unstable preprocessing behavior.
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"[{dataset_name}] Need at least 2 classes in target for benchmark")
    if (class_counts < 2).any():
        too_rare = class_counts[class_counts < 2]
        raise ValueError(f"[{dataset_name}] Some classes have <2 samples: {too_rare.to_dict()}")

    effective_splits = int(max(2, min(cv_splits, int(class_counts.min()))))

    splitter = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=cv_repeats,
        random_state=seed,
    )

    fold_metrics: List[Dict[str, float]] = []
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = make_pipeline(X_train)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        fold_metrics.append({
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        })

    fm = pd.DataFrame(fold_metrics)
    synthetic_ratio = float(pd.to_numeric(work.get("is_synthetic", pd.Series(np.nan, index=work.index)), errors="coerce").mean())

    return {
        "dataset": dataset_name,
        "target": target_col,
        "rows": float(len(y)),
        "classes": float(y.nunique()),
        "min_class_count": float(class_counts.min()),
        "synthetic_ratio": synthetic_ratio,
        "cv_splits": float(effective_splits),
        "cv_repeats": float(cv_repeats),
        "accuracy": float(fm["accuracy"].mean()),
        "accuracy_std": float(fm["accuracy"].std(ddof=0)),
        "balanced_accuracy": float(fm["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(fm["balanced_accuracy"].std(ddof=0)),
        "f1_macro": float(fm["f1_macro"].mean()),
        "f1_macro_std": float(fm["f1_macro"].std(ddof=0)),
        "f1_weighted": float(fm["f1_weighted"].mean()),
        "f1_weighted_std": float(fm["f1_weighted"].std(ddof=0)),
    }


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {path}")
    return pd.read_csv(path)


def resolve_target_column(df: pd.DataFrame, preferred: str) -> str:
    candidates = [preferred, "rating_class", "rating_detail", "binary_rating", "y"]
    seen = set()
    ordered_candidates = []
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        ordered_candidates.append(col)

    for col in ordered_candidates:
        if col in df.columns and df[col].notna().any():
            return col

    raise ValueError(
        "Cannot resolve target column. Tried: "
        + ", ".join(ordered_candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark augmented train datasets")
    parser.add_argument("--ctgan", type=str, default=str(DEFAULT_CTGan_PATH), help="Path to CTGAN train CSV")
    parser.add_argument("--tabddpm", type=str, default=str(DEFAULT_TABDDPM_PATH), help="Path to TabDDPM train CSV")
    parser.add_argument("--smote", type=str, default=str(DEFAULT_SMOTE_PATH), help="Path to SMOTE train CSV")
    parser.add_argument("--timegan", type=str, default=str(DEFAULT_TIMEGAN_PATH), help="Path to TimeGAN train CSV")
    parser.add_argument("--target", type=str, default="rating_class", help="Common target column for comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help="Output CSV report path")
    args = parser.parse_args()

    datasets = []
    for name, path_str in [
        ("ctgan", args.ctgan),
        ("tabddpm", args.tabddpm),
        ("smote", args.smote),
        ("timegan", getattr(args, "timegan", None)),
    ]:
        if path_str:
            p = Path(path_str)
            if p.exists():
                datasets.append((name, p))

    if not datasets:
        print("No existing datasets found to benchmark!")
        return

    results: List[Dict[str, float]] = []
    for name, path in datasets:
        df = load_dataset(path)
        target_col = resolve_target_column(df, args.target)
        if target_col != args.target:
            print(f"[INFO] [{name}] Fallback target: {target_col} (requested: {args.target})")

        metrics = run_dataset_benchmark(
            df=df,
            dataset_name=name,
            target_col=target_col,
            seed=args.seed,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
        )
        results.append(metrics)

    report = pd.DataFrame(results)
    report = report.sort_values(by=["f1_macro", "balanced_accuracy"], ascending=[False, False])

    print("\n=== Benchmark 3 Augmented Train Datasets ===")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
