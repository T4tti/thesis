"""
scripts/export_tlstm_meta.py — Extract metadata từ training data để phục vụ backend inference.

Chạy 1 lần từ repo root:
    python scripts/export_tlstm_meta.py

Output:
    app/backend/model/tlstm_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "merged_credit_rating_common_3groups.csv"
OUTPUT_PATH = ROOT / "app" / "backend" / "model" / "tlstm_meta.json"

# ── Feature lists (phải khớp đúng thứ tự trong notebook) ─────────────────────
FINANCIAL_FEATURES: list[str] = [
    "current_ratio", "debt_equity_ratio",
    "gross_profit_margin", "operating_profit_margin",
    "ebit_margin", "pretax_profit_margin",
    "net_profit_margin", "asset_turnover",
    "roe", "roa",
    "operating_cashflow_ps", "free_cashflow_ps",
]
DELTA_FEATURES: list[str] = [f"{c}_delta" for c in FINANCIAL_FEATURES]
MODEL_FEATURES: list[str] = FINANCIAL_FEATURES + DELTA_FEATURES

TARGET_COL = "rating_detail"
TARGET_ORDERED_LABELS = ["Distressed", "HY", "IG"]
SECTOR_COL = "sector"
SECTOR_UNKNOWN = "UNKNOWN"

# ── Model hyperparameters (phải khớp checkpoint) ──────────────────────────────
MODEL_HPARAMS = {
    "n_channels":           len(MODEL_FEATURES),   # 24
    "n_classes":            3,
    "hidden_size":          128,
    "dropout":              0.10,
    "n_mfs":                5,
    "d_model":              128,
    "n_heads":              4,
    "n_layers":             3,
    "sector_emb_dim":       16,
    "max_relative_position": 32,
    "input_size":           1,           # sliding window length used at training
}


def main() -> None:
    print(f"[export_meta] Loading train data from {TRAIN_PATH} …")
    assert TRAIN_PATH.exists(), f"File not found: {TRAIN_PATH}"
    df = pd.read_csv(TRAIN_PATH)
    print(f"[export_meta] Loaded {len(df):,} rows.")

    # ── 1. Drop rows with no target ───────────────────────────────────────────
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df = df[df[TARGET_COL].isin(TARGET_ORDERED_LABELS)].reset_index(drop=True)
    print(f"[export_meta] Rows after label filter: {len(df):,}")

    # ── 2. Sector encoding ────────────────────────────────────────────────────
    if SECTOR_COL not in df.columns:
        df[SECTOR_COL] = SECTOR_UNKNOWN
    df[SECTOR_COL] = df[SECTOR_COL].fillna(SECTOR_UNKNOWN).astype(str).str.strip()
    df.loc[df[SECTOR_COL] == "", SECTOR_COL] = SECTOR_UNKNOWN

    sector_enc = LabelEncoder()
    sector_enc.fit(df[SECTOR_COL])
    sector_classes: list[str] = sector_enc.classes_.tolist()
    n_sectors = len(sector_classes)
    print(f"[export_meta] Sectors ({n_sectors}): {sector_classes}")

    # ── 3. Label mapping ──────────────────────────────────────────────────────
    observed = sorted(df[TARGET_COL].unique().tolist())
    ordered_present = [c for c in TARGET_ORDERED_LABELS if c in observed]
    label_to_id = {raw: idx for idx, raw in enumerate(ordered_present)}
    id_to_label = {idx: raw for raw, idx in label_to_id.items()}
    label_classes: list[str] = [id_to_label[i] for i in range(len(ordered_present))]
    print(f"[export_meta] Labels: {label_classes}")

    # ── 4. Imputation statistics (train only, leakage-safe) ───────────────────
    for col in FINANCIAL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Compute per-feature median + clip bounds on FULL train file
    impute_medians: dict[str, float] = {}
    clip_lowers: dict[str, float] = {}
    clip_uppers: dict[str, float] = {}

    for col in FINANCIAL_FEATURES:
        ser = df[col].dropna()
        med = float(ser.median()) if len(ser) > 0 else 0.0
        lo  = float(ser.quantile(0.01)) if len(ser) > 0 else -1e9
        hi  = float(ser.quantile(0.99)) if len(ser) > 0 else  1e9
        impute_medians[col] = med
        clip_lowers[col]    = lo
        clip_uppers[col]    = hi

    # ── 5. Apply impute + clip before fitting scaler ───────────────────────────
    for col in FINANCIAL_FEATURES:
        df[col] = df[col].fillna(impute_medians[col])
        df[col] = df[col].clip(clip_lowers[col], clip_uppers[col])

    # Delta features: 0 for single-point inference (no previous row)
    for col in FINANCIAL_FEATURES:
        dcol = f"{col}_delta"
        df[dcol] = 0.0

    # ── 6. Fit RobustScaler on MODEL_FEATURES ─────────────────────────────────
    scaler = RobustScaler()
    scaler.fit(df[MODEL_FEATURES].values)
    scaler_center: list[float] = scaler.center_.tolist()
    scaler_scale: list[float]  = scaler.scale_.tolist()
    print(f"[export_meta] Scaler fitted on {len(df):,} rows × {len(MODEL_FEATURES)} features.")

    # ── 7. Update MODEL_HPARAMS with runtime n_sectors ────────────────────────
    MODEL_HPARAMS["n_sectors"] = n_sectors

    # ── 8. Save JSON ──────────────────────────────────────────────────────────
    meta = {
        "financial_features":  FINANCIAL_FEATURES,
        "delta_features":      DELTA_FEATURES,
        "model_features":      MODEL_FEATURES,
        "label_classes":       label_classes,          # ["Distressed", "HY", "IG"]
        "label_to_id":         label_to_id,
        "sector_classes":      sector_classes,
        "n_sectors":           n_sectors,
        "impute_medians":      impute_medians,
        "clip_lowers":         clip_lowers,
        "clip_uppers":         clip_uppers,
        "scaler_center":       scaler_center,
        "scaler_scale":        scaler_scale,
        "model_hparams":       MODEL_HPARAMS,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[export_meta] ✅ Metadata saved to: {OUTPUT_PATH}")
    print(f"  label_classes : {label_classes}")
    print(f"  sector_classes: {sector_classes}")
    print(f"  scaler_center[:3]: {[round(v, 4) for v in scaler_center[:3]]}")
    print(f"  scaler_scale[:3]:  {[round(v, 4) for v in scaler_scale[:3]]}")


if __name__ == "__main__":
    main()
