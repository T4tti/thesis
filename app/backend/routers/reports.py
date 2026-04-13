"""
routers/reports.py — Data endpoints for companies, stats, sectors.

Endpoints:
    GET /api/companies  — paginated list with filter + search
    GET /api/stats      — aggregate statistics
    GET /api/sectors    — unique sector list
    GET /api/benchmark  — model benchmark results
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from state import MODEL_STATE

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["reports"])


def _get_df() -> pd.DataFrame:
    df = MODEL_STATE.get("data_df")
    if df is None:
        raise HTTPException(status_code=503, detail="Data not yet loaded.")
    return df


# ---------------------------------------------------------------------------
# /api/companies
# ---------------------------------------------------------------------------

@router.get("/companies")
async def get_companies(
    page:     int            = Query(1, ge=1),
    per_page: int            = Query(20, ge=1, le=100),
    sector:   Optional[str] = Query(None),
    rating:   Optional[str] = Query(None),
    search:   Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Return paginated company list, optionally filtered."""
    df = _get_df().copy()

    # Filters
    if sector and sector != "all":
        df = df[df["sector"].str.lower() == sector.lower()]
    if rating and rating != "all":
        df = df[df["rating_detail"].str.upper() == rating.upper()]
    if search:
        q = search.lower().strip()
        mask = (
            df["company_name"].str.lower().str.contains(q, na=False) |
            df["ticker"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    total = len(df)
    pages = max(1, math.ceil(total / per_page))
    page  = min(page, pages)
    start = (page - 1) * per_page
    end   = start + per_page

    subset = df.iloc[start:end]

    display_cols = [
        "company_name", "ticker", "sector",
        "rating_detail", "rating_date", "rating_agency", "source",
    ]
    keep = [c for c in display_cols if c in subset.columns]
    records = subset[keep].fillna("").to_dict(orient="records")

    return {
        "data":     records,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    pages,
    }


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Return aggregate statistics for the dashboard."""
    df = _get_df()

    rating_dist = df["rating_detail"].value_counts().to_dict()
    sector_dist = df["sector"].value_counts().head(15).to_dict()

    year_col = df["rating_date"].astype(str).str[:4]
    years    = pd.to_numeric(year_col, errors="coerce").dropna().astype(int)
    
    # Model CV metrics (from training)
    metrics = MODEL_STATE.get("metrics") or {}

    return {
        "total_records":        int(len(df)),
        "unique_tickers":       int(df["ticker"].nunique()),
        "unique_sectors":       int(df["sector"].nunique()),
        "rating_distribution":  {k: int(v) for k, v in rating_dist.items()},
        "sector_distribution":  {k: int(v) for k, v in sector_dist.items()},
        "year_range": {
            "min": int(years.min()) if len(years) > 0 else None,
            "max": int(years.max()) if len(years) > 0 else None,
        },
        "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
        "model_cv_f1_weighted": round(metrics.get("cv_f1_weighted_mean", 0), 4),
    }


# ---------------------------------------------------------------------------
# /api/sectors
# ---------------------------------------------------------------------------

@router.get("/sectors")
async def get_sectors() -> List[str]:
    """Return sorted list of unique sector names."""
    df = _get_df()
    return sorted(df["sector"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# /api/benchmark
# ---------------------------------------------------------------------------

_BENCHMARK_SUMMARY = {
    "model": "LightGBM (5-Fold Stratified CV)",
    "features": 12,
    "target_classes": ["IG", "HY", "Distressed"],
    "augmentation": "TimeGAN (25.9% synthetic ratio)",
    "walk_forward_folds": 3,
    "walk_forward": {
        "val_accuracy":  0.3353,
        "val_balanced_accuracy": 0.3321,
        "val_macro_f1":  0.2795,
        "test_accuracy": 0.2395,
        "test_macro_f1": 0.1921,
    },
    "augmented_cv": {
        "cv_accuracy_mean":           0.4089,
        "cv_balanced_accuracy_mean":  0.3737,
        "cv_f1_macro_mean":           0.2292,
        "cv_f1_weighted_mean":        0.2725,
    },
}


@router.get("/benchmark")
async def get_benchmark() -> Dict[str, Any]:
    """Return benchmark summary for the methodology page."""
    metrics = MODEL_STATE.get("metrics") or {}
    return {
        **_BENCHMARK_SUMMARY,
        "production_model_cv": {
            "cv_accuracy":             round(metrics.get("cv_accuracy_mean", 0), 4),
            "cv_balanced_accuracy":    round(metrics.get("cv_balanced_accuracy_mean", 0), 4),
            "cv_f1_weighted":          round(metrics.get("cv_f1_weighted_mean", 0), 4),
            "cv_f1_macro":             round(metrics.get("cv_f1_macro_mean", 0), 4),
            "n_samples":               metrics.get("n_samples", 0),
        },
    }
