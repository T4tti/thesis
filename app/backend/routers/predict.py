"""
routers/predict.py — Credit rating inference endpoints.

Endpoints:
    POST /api/predict         — DMF/DCS T-LSTM + GraphSAGE (primary)
    POST /api/predict/tlstm   — Compatibility alias using the primary runtime

How to Run:
        # Primary endpoint
    curl -X POST http://localhost:8000/api/predict \\
         -H "Content-Type: application/json" \\
         -d '{"current_ratio": 1.5, "debt_equity_ratio": 0.8}'

        # Compatibility alias
    curl -X POST http://localhost:8000/api/predict/tlstm \\
         -H "Content-Type: application/json" \\
         -d '{"current_ratio": 1.5, "roe": 0.15, "sector": "Finance", "previous_rating": "IG"}'

Expected Output (both):
    {"rating": "IG", "probabilities": {...}, "confidence": 0.75, ...}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from i18n import Lang, resolve_lang
from i18n_messages import msg
from state import MODEL_STATE

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["predict"])

# ---------------------------------------------------------------------------
# Shared example + field definitions
# ---------------------------------------------------------------------------

_EXAMPLE_FEATURES = {
    "current_ratio": 1.5,
    "debt_equity_ratio": 0.8,
    "gross_profit_margin": 0.35,
    "operating_profit_margin": 0.15,
    "ebit_margin": 0.12,
    "pretax_profit_margin": 0.10,
    "net_profit_margin": 0.08,
    "asset_turnover": 0.9,
    "roe": 0.15,
    "roa": 0.08,
    "operating_cashflow_ps": 2.5,
    "free_cashflow_ps": 1.8,
}

_FIN_FEATURE_KEYS = set(_EXAMPLE_FEATURES.keys())
_CONTEXT_KEYS = {
    "row_id",
    "ticker",
    "company_name",
    "rating_date",
    "sector",
    "previous_rating",
}
_SECTOR_ID_TO_NAME = {
    "0": "Basic Industries",
    "1": "Capital Goods",
    "2": "Consumer Durables",
    "3": "Consumer Non-Durables",
    "4": "Consumer Services",
    "5": "Energy",
    "6": "Finance",
    "7": "Health Care",
    "8": "Miscellaneous",
    "9": "Public Utilities",
    "10": "Technology",
    "11": "Transportation",
    "12": "__MISSING__",
}


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sector_label(raw_sector: Any, resolved_sector: Any) -> Optional[str]:
    raw = _clean_text(raw_sector)
    if raw and not raw.isdigit():
        return raw
    resolved = _clean_text(resolved_sector)
    if resolved and resolved in _SECTOR_ID_TO_NAME:
        return _SECTOR_ID_TO_NAME[resolved]
    return resolved or raw


def _enrich_prediction_context(result: dict, body: BaseModel) -> dict:
    payload = body.model_dump()
    input_context = {
        key: _clean_text(payload.get(key))
        for key in _CONTEXT_KEYS
        if _clean_text(payload.get(key)) is not None
    }
    sector_resolved = _sector_label(payload.get("sector"), result.get("sector_resolved"))
    if sector_resolved:
        input_context["sector_resolved"] = sector_resolved
    if result.get("previous_rating"):
        input_context["previous_rating_resolved"] = _clean_text(result.get("previous_rating"))

    decision_context = {
        "selected_model": result.get("selected_model"),
        "dcs_case": result.get("dcs_case"),
        "tlstm_score": result.get("tlstm_score"),
        "graphsage_score": result.get("graphsage_score"),
        "graphsage_runtime": result.get("graphsage_runtime"),
        "graphsage_proxy_distance": result.get("graphsage_proxy_distance"),
    }
    decision_context = {k: v for k, v in decision_context.items() if v is not None}

    enriched = dict(result)
    if sector_resolved:
        enriched["sector_resolved"] = sector_resolved
    enriched["input_context"] = input_context
    enriched["decision_context"] = decision_context
    enriched["xai_match_hint"] = {
        "row_id_available": bool(input_context.get("row_id")),
        "ticker_date_available": bool(input_context.get("ticker") and input_context.get("rating_date")),
        "preferred_match_order": ["row_id", "ticker+rating_date"],
    }
    return enriched


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """12 financial features; missing values are imputed by the T-LSTM base preprocessing."""

    current_ratio:           Optional[float] = Field(None, description="Current Assets / Current Liabilities")
    debt_equity_ratio:       Optional[float] = Field(None, description="Total Debt / Shareholders' Equity")
    gross_profit_margin:     Optional[float] = Field(None, description="Gross Profit / Net Revenue (decimal)")
    operating_profit_margin: Optional[float] = Field(None, description="Operating Profit / Net Revenue (decimal)")
    ebit_margin:             Optional[float] = Field(None, description="EBIT / Net Revenue (decimal)")
    pretax_profit_margin:    Optional[float] = Field(None, description="Pre-Tax Profit / Net Revenue (decimal)")
    net_profit_margin:       Optional[float] = Field(None, description="Net Profit / Net Revenue (decimal)")
    asset_turnover:          Optional[float] = Field(None, description="Net Revenue / Average Total Assets")
    roe:                     Optional[float] = Field(None, description="Net Profit / Average Equity (decimal)")
    roa:                     Optional[float] = Field(None, description="Net Profit / Average Total Assets (decimal)")
    operating_cashflow_ps:   Optional[float] = Field(None, description="Operating Cash Flow per Share")
    free_cashflow_ps:        Optional[float] = Field(None, description="Free Cash Flow per Share")

    sector: Optional[str] = Field(None, description="Optional industry sector context")
    previous_rating: Optional[str] = Field(None, description="Optional previous rating context")
    row_id: Optional[str] = Field(None, description="Optional source row id for artifact-backed xAI")
    ticker: Optional[str] = Field(None, description="Optional ticker symbol")
    company_name: Optional[str] = Field(None, description="Optional company name")
    rating_date: Optional[str] = Field(None, description="Optional rating date, preferably YYYY-MM-DD")

    model_config = {"json_schema_extra": {"example": {
        **_EXAMPLE_FEATURES,
        "sector": "Finance",
        "previous_rating": "IG",
        "ticker": "AAPL",
        "rating_date": "2016-06-03",
    }}}


class PredictTLSTMRequest(BaseModel):
    """12 financial features + sector + previous_rating for the rating runtime."""

    current_ratio:           Optional[float] = Field(None, description="Current Assets / Current Liabilities")
    debt_equity_ratio:       Optional[float] = Field(None, description="Total Debt / Shareholders' Equity")
    gross_profit_margin:     Optional[float] = Field(None, description="Gross Profit / Net Revenue (decimal)")
    operating_profit_margin: Optional[float] = Field(None, description="Operating Profit / Net Revenue (decimal)")
    ebit_margin:             Optional[float] = Field(None, description="EBIT / Net Revenue (decimal)")
    pretax_profit_margin:    Optional[float] = Field(None, description="Pre-Tax Profit / Net Revenue (decimal)")
    net_profit_margin:       Optional[float] = Field(None, description="Net Profit / Net Revenue (decimal)")
    asset_turnover:          Optional[float] = Field(None, description="Net Revenue / Average Total Assets")
    roe:                     Optional[float] = Field(None, description="Net Profit / Average Equity (decimal)")
    roa:                     Optional[float] = Field(None, description="Net Profit / Average Total Assets (decimal)")
    operating_cashflow_ps:   Optional[float] = Field(None, description="Operating Cash Flow per Share")
    free_cashflow_ps:        Optional[float] = Field(None, description="Free Cash Flow per Share")

    sector: Optional[str] = Field(
        None,
        description=(
            "Industry sector. One of: Basic Industries, Capital Goods, Consumer Durables, "
            "Consumer Non-Durables, Consumer Services, Energy, Finance, Health Care, "
            "Miscellaneous, Public Utilities, Technology, Transportation. "
            "Defaults to 'Miscellaneous' if omitted or unrecognised."
        ),
    )
    previous_rating: Optional[str] = Field(
        None,
        description=(
            "Last known credit rating class: 'IG', 'HY', or 'Distressed'. "
            "Used as temporal context by the model. Defaults to 'IG' if omitted."
        ),
    )
    row_id: Optional[str] = Field(None, description="Optional source row id for artifact-backed xAI")
    ticker: Optional[str] = Field(None, description="Optional ticker symbol")
    company_name: Optional[str] = Field(None, description="Optional company name")
    rating_date: Optional[str] = Field(None, description="Optional rating date, preferably YYYY-MM-DD")

    model_config = {"json_schema_extra": {"example": {
        **_EXAMPLE_FEATURES,
        "sector": "Finance",
        "previous_rating": "IG",
    }}}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _predict_with_rating_model(
    features: dict,
    sector: Optional[str],
    previous_rating: Optional[str],
    lang: Lang,
) -> dict:
    from model.predictor import predict

    return predict(
        features=features,
        sector=sector,
        previous_rating=previous_rating,
        model=MODEL_STATE["rating_model"],
        meta=MODEL_STATE["rating_meta"],
        lang=lang,
    )


def _require_rating_runtime(lang: Lang) -> Dict[str, Any]:
    if not MODEL_STATE.get("ready"):
        raise HTTPException(
            status_code=503,
            detail=msg("model_not_ready", lang),
        )
    if MODEL_STATE.get("rating_model") is None:
        raise HTTPException(
            status_code=503,
            detail=msg("model_not_ready", lang),
        )
    return {
        "model": MODEL_STATE["rating_model"],
        "meta": MODEL_STATE["rating_meta"],
    }


@router.post("/predict", summary="DMF/DCS T-LSTM + GraphSAGE Credit Rating (Primary)")
async def predict_rating(
    body: PredictRequest,
    lang: Lang = Depends(resolve_lang),
) -> dict:
    """
    Predict credit rating (IG / HY / Distressed) using the pre-trained
    **DMF/DCS T-LSTM + GraphSAGE** decision-combination runtime.

    This endpoint keeps backward compatibility for existing frontend clients.
    - Uses the same 12 financial fields as before
    - Sector defaults to "Miscellaneous"
    - previous_rating defaults to "IG"
    """
    _require_rating_runtime(lang)

    try:
        features = {k: v for k, v in body.model_dump().items() if k in _FIN_FEATURE_KEYS}
        result = _predict_with_rating_model(
            features=features,
            sector=body.sector,
            previous_rating=body.previous_rating,
            lang=lang,
        )
        return _enrich_prediction_context(result, body)
    except Exception as exc:
        log.exception("DMF/DCS prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=msg("prediction_error", lang)) from exc


@router.post("/predict/tlstm", summary="Credit Rating Compatibility Alias")
async def predict_rating_tlstm(
    body: PredictTLSTMRequest,
    lang: Lang = Depends(resolve_lang),
) -> dict:
    """
    Predict credit rating (IG / HY / Distressed) using the primary
    **DMF/DCS T-LSTM + GraphSAGE** runtime.

    ### Inputs
    - **12 financial ratios** (all optional, imputed by training-set statistics)
    - **sector** — industry sector string (optional, defaults to *Miscellaneous*)
    - **previous_rating** — last known rating class (optional, defaults to *IG*)

    ### Returns
    - `rating`: predicted class (`IG`, `HY`, or `Distressed`)
    - `probabilities`: per-class softmax probabilities
    - `confidence`: probability of the predicted class
    - `risk_level`, `risk_score`, colored label, and Vietnamese/English interpretations
    - `sector_resolved`, `previous_rating`: inputs resolved by the model
    """
    _require_rating_runtime(lang)

    try:
        features = {
            k: v for k, v in body.model_dump().items()
            if k in _FIN_FEATURE_KEYS
        }

        result = _predict_with_rating_model(
            features=features,
            sector=body.sector,
            previous_rating=body.previous_rating,
            lang=lang,
        )
        return _enrich_prediction_context(result, body)
    except Exception as exc:
        log.exception("DMF/DCS prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=msg("prediction_error", lang)) from exc
