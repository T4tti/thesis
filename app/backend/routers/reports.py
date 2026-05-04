"""
routers/reports.py — Data endpoints for companies, stats, sectors.

Endpoints:
    GET /api/companies  — paginated list with filter + search
    GET /api/stats      — aggregate statistics
    GET /api/sectors    — unique sector list
    GET /api/benchmark  — model benchmark results
    POST /api/history   — append one analyze result to reports history
"""
from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session
from sqlalchemy.exc import OperationalError

from i18n import Lang, resolve_lang
from i18n_messages import msg
from state import MODEL_STATE
from database import get_session, save_rating_history, RatingHistory

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["reports"])

# Mapping sector ID (int) to sector name (str)
SECTOR_MAPPING = {
    0: "Basic Industries", 1: "Capital Goods", 2: "Consumer Durables",
    3: "Consumer Non-Durables", 4: "Consumer Services", 5: "Energy",
    6: "Finance", 7: "Health Care", 8: "Miscellaneous",
    9: "Public Utilities", 10: "Technology", 11: "Transportation", 12: "__MISSING__"
}

SECTOR_NAMES_LOCALIZED = {
    "Basic Industries": {"en": "Basic Industries", "vi": "Công nghiệp Cơ bản"},
    "Capital Goods": {"en": "Capital Goods", "vi": "Tư liệu Sản xuất"},
    "Consumer Durables": {"en": "Consumer Durables", "vi": "Hàng Tiêu dùng Bền vững"},
    "Consumer Non-Durables": {"en": "Consumer Non-Durables", "vi": "Hàng Tiêu dùng Không bền"},
    "Consumer Services": {"en": "Consumer Services", "vi": "Dịch vụ Tiêu dùng"},
    "Energy": {"en": "Energy", "vi": "Năng lượng"},
    "Finance": {"en": "Finance", "vi": "Tài chính"},
    "Health Care": {"en": "Health Care", "vi": "Y tế"},
    "Miscellaneous": {"en": "Khác", "vi": "Khác"},
    "Public Utilities": {"en": "Tiện ích Công cộng", "vi": "Tiện ích Công cộng"},
    "Technology": {"en": "Công nghệ", "vi": "Công nghệ"},
    "Transportation": {"en": "Giao thông Vận tải", "vi": "Giao thông Vận tải"},
}

def _get_localized_sector(name: str, lang: Lang) -> str:
    mapping = SECTOR_NAMES_LOCALIZED.get(name)
    if not mapping:
        return name
    return mapping.get(lang, mapping["en"])


def _get_df(lang: Lang = "en") -> pd.DataFrame:
    df = MODEL_STATE.get("data_df")
    if df is None:
        raise HTTPException(status_code=503, detail=msg("data_not_loaded", lang))
    return df


def _ensure_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = (
        "company_name", "ticker", "sector", "rating_detail",
        "rating_date", "rating_agency", "source",
    )
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    return df


class SaveHistoryRequest(BaseModel):
    """Persist one analyze result so it appears in Reports & Data."""

    prediction: Dict[str, Any] = Field(..., description="Response payload returned by /api/predict")
    features: Dict[str, Optional[float]] = Field(..., description="Input financial features used for prediction")
    ticker: Optional[str] = Field(None, description="Optional ticker symbol from user input/CSV")
    company_name: Optional[str] = Field(None, description="Optional company name from user input/CSV")
    sector: Optional[str] = Field(None, description="Optional sector label")
    source: Optional[str] = Field("VN-Rating Analyze", description="Origin tag shown in reports")


def _build_history_record(payload: SaveHistoryRequest) -> Dict[str, Any]:
    now_utc = datetime.datetime.utcnow()
    stamp = now_utc.strftime("%Y-%m-%d")
    ticker = (payload.ticker or "").strip() or f"ANL-{now_utc.strftime('%Y%m%d%H%M%S')}"
    company_name = (payload.company_name or "").strip() or f"Analysis {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Resolve sector using mapping if it's an integer ID
    raw_sector = (payload.sector or "").strip()
    sector = "Unknown"
    if raw_sector:
        try:
            # If it's a numeric string, map it
            sector_id = int(raw_sector)
            sector = SECTOR_MAPPING.get(sector_id, raw_sector)
        except ValueError:
            # Otherwise use the string as-is
            sector = raw_sector
            
    source = (payload.source or "").strip() or "VN-Rating Analyze"

    pred = payload.prediction or {}
    rating_detail = str(pred.get("rating") or "")
    confidence = float(pred.get("confidence", 0.0) or 0.0)
    risk_score = float(pred.get("risk_score", 0.0) or 0.0)
    risk_level = str(pred.get("risk_level") or "")

    record: Dict[str, Any] = {
        "company_name": company_name,
        "ticker": ticker,
        "sector": sector,
        "rating_detail": rating_detail,
        "rating_date": stamp,
        "rating_agency": "VN-Rating AI",
        "source": source,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "created_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for feature_name, feature_value in payload.features.items():
        record[feature_name] = feature_value

    return record


def _persist_history_record(record: Dict[str, Any]) -> None:
    history_path = MODEL_STATE.get("history_csv_path")
    if not history_path:
        return

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([record])
    if history_path.exists():
        old_df = pd.read_csv(history_path, encoding="utf-8")
        merged_df = pd.concat([old_df, row_df], ignore_index=True, sort=False)
    else:
        merged_df = row_df

    merged_df.to_csv(history_path, index=False, encoding="utf-8")


@router.post("/history", summary="Save one analyze result into Reports & Data")
async def save_history(
    body: SaveHistoryRequest,
    session: Session = Depends(get_session),
    lang: Lang = Depends(resolve_lang),
) -> Dict[str, Any]:
    """Persist analysis history to Database and update in-memory reports dataset."""
    lock = MODEL_STATE.get("history_lock")
    if lock is None:
        raise HTTPException(status_code=503, detail=msg("history_not_ready", lang))

    try:
        with lock:
            record = _build_history_record(body)
            
            # 1. Save to Database (fallback to CSV if Postgres is down)
            db_saved = True
            csv_saved = False
            db_record: RatingHistory | None = None
            try:
                db_record = save_rating_history(record, session=session)
            except OperationalError as exc:
                db_saved = False
                log.warning("Database unavailable; falling back to CSV history. Error: %s", exc)
                _persist_history_record(record)
                csv_saved = True
            
            # 2. Update in-memory DataFrame (for Reports & Data page)
            df = _ensure_report_columns(_get_df().copy())
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True, sort=False)
            MODEL_STATE["data_df"] = df
            
            # 3. Optional: also save to CSV as backup (old behavior) - removed to avoid duplicate logs
            # _persist_history_record(record)

        return {
            "saved": True,
            "db_saved": db_saved,
            "csv_saved": csv_saved,
            "id": getattr(db_record, "id", None),
            "record": {k: record.get(k) for k in ("ticker", "company_name", "rating_detail", "rating_date")},
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Failed to save analysis history: %s", exc)
        raise HTTPException(status_code=500, detail=msg("history_save_failed", lang)) from exc


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
    sort_order: str          = Query("desc", pattern="^(asc|desc)$"),
    lang: Lang = Depends(resolve_lang),
) -> Dict[str, Any]:
    """Return paginated company list, optionally filtered."""
    df = _ensure_report_columns(_get_df(lang).copy())

    # Filters
    if sector and sector != "all":
        # Search in both localized and original names
        def matches_sector(row_sector: str) -> bool:
            orig = str(row_sector).lower()
            loc_en = _get_localized_sector(row_sector, "en").lower()
            loc_vi = _get_localized_sector(row_sector, "vi").lower()
            target = sector.lower()
            return target in (orig, loc_en, loc_vi)
        
        mask = df["sector"].apply(matches_sector)
        df = df[mask]
    
    if rating and rating != "all":
        df = df[df["rating_detail"].str.upper() == rating.upper()]
    if search:
        q = search.lower().strip()
        mask = (
            df["company_name"].str.lower().str.contains(q, na=False) |
            df["ticker"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    # Show newest ratings first in Reports & Data (including newly saved analyze history).
    sort_dt = pd.to_datetime(df["rating_date"], errors="coerce")
    is_asc = (sort_order == "asc")
    df = df.assign(_sort_date=sort_dt).sort_values("_sort_date", ascending=is_asc, na_position="last" if not is_asc else "first")

    total = len(df)
    pages = max(1, math.ceil(total / per_page))
    page  = min(page, pages)
    start = (page - 1) * per_page
    end   = start + per_page

    subset = df.iloc[start:end].drop(columns=["_sort_date"], errors="ignore")

    display_cols = [
        "company_name", "ticker", "sector",
        "rating_detail", "rating_date", "rating_agency", "source",
    ]
    keep = [c for c in display_cols if c in subset.columns]
    records = subset[keep].fillna("").to_dict(orient="records")
    
    # Translate sectors in the output
    for r in records:
        if "sector" in r:
            r["sector"] = _get_localized_sector(r["sector"], lang)

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
async def get_stats(lang: Lang = Depends(resolve_lang)) -> Dict[str, Any]:
    """Return aggregate statistics for the dashboard."""
    df = _get_df(lang)

    rating_dist = df["rating_detail"].value_counts().to_dict()
    sector_dist = df["sector"].value_counts().head(15).to_dict()

    year_col = df["rating_date"].astype(str).str[:4]
    years    = pd.to_numeric(year_col, errors="coerce").dropna().astype(int)
    
    tlstm_meta = MODEL_STATE.get("tlstm_meta") or {}
    tlstm_benchmark = tlstm_meta.get("benchmark") or {}
    model_f1w = float(tlstm_benchmark.get("cv_f1_weighted", 0.0))

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
        "model_cv_f1_weighted": round(model_f1w, 4),
    }


# ---------------------------------------------------------------------------
# /api/sectors
# ---------------------------------------------------------------------------

@router.get("/sectors")
async def get_sectors(lang: Lang = Depends(resolve_lang)) -> List[str]:
    """Return sorted list of unique sector names."""
    df = _get_df(lang)
    unique_sectors = df["sector"].dropna().unique().tolist()
    localized = sorted([_get_localized_sector(s, lang) for s in unique_sectors])
    return localized


# ---------------------------------------------------------------------------
# /api/benchmark
# ---------------------------------------------------------------------------

_BENCHMARK_SUMMARY = {
    "model": "TLSTMFuzzy (pre-trained)",
    "features": 12,
    "target_classes": ["IG", "HY", "Distressed"],
    "augmentation": "N/A",
    "walk_forward_folds": 0,
    "walk_forward": None,
    "augmented_cv": None,
}


@router.get("/benchmark")
async def get_benchmark() -> Dict[str, Any]:
    """Return benchmark summary for the methodology page."""
    tlstm_meta = MODEL_STATE.get("tlstm_meta") or {}
    tlstm_benchmark = tlstm_meta.get("benchmark") or {}
    n_samples = int(tlstm_benchmark.get("n_samples", len(_get_df())))

    return {
        **_BENCHMARK_SUMMARY,
        "production_model_cv": {
            "cv_accuracy":             round(float(tlstm_benchmark.get("cv_accuracy", 0.0)), 4),
            "cv_balanced_accuracy":    round(float(tlstm_benchmark.get("cv_balanced_accuracy", 0.0)), 4),
            "cv_f1_weighted":          round(float(tlstm_benchmark.get("cv_f1_weighted", 0.0)), 4),
            "cv_f1_macro":             round(float(tlstm_benchmark.get("cv_f1_macro", 0.0)), 4),
            "n_samples":               n_samples,
        },
    }
