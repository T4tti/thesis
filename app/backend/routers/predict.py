"""
routers/predict.py — Credit rating inference endpoints.

Endpoints:
    POST /api/predict         — TLSTMFuzzyClassifier (primary)
    POST /api/predict/tlstm   — TLSTMFuzzyClassifier (compatibility alias)

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


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """12 financial features — all optional, missing values imputed by TLSTM preprocessing medians."""

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

    model_config = {"json_schema_extra": {"example": _EXAMPLE_FEATURES}}


class PredictTLSTMRequest(BaseModel):
    """12 financial features + sector + previous_rating for TLSTMFuzzy model."""

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

    model_config = {"json_schema_extra": {"example": {
        **_EXAMPLE_FEATURES,
        "sector": "Finance",
        "previous_rating": "IG",
    }}}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _predict_with_tlstm(
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
        model=MODEL_STATE["tlstm_model"],
        meta=MODEL_STATE["tlstm_meta"],
        lang=lang,
    )


def _require_tlstm_runtime(lang: Lang) -> Dict[str, Any]:
    if not MODEL_STATE.get("ready"):
        raise HTTPException(
            status_code=503,
            detail=msg("model_not_ready", lang),
        )
    if MODEL_STATE.get("tlstm_model") is None:
        raise HTTPException(
            status_code=503,
            detail=msg("model_not_ready", lang),
        )
    return {
        "model": MODEL_STATE["tlstm_model"],
        "meta": MODEL_STATE["tlstm_meta"],
    }


@router.post("/predict", summary="TLSTMFuzzy Credit Rating (Primary)")
async def predict_rating(
    body: PredictRequest,
    lang: Lang = Depends(resolve_lang),
) -> dict:
    """
    Predict credit rating (IG / HY / Distressed) using the pre-trained
    **TLSTMFuzzy** deep learning model.

    This endpoint keeps backward compatibility for existing frontend clients.
    - Uses the same 12 financial fields as before
    - Sector defaults to "Miscellaneous"
    - previous_rating defaults to "IG"
    """
    _require_tlstm_runtime(lang)

    try:
        features = {k: v for k, v in body.model_dump().items() if k in _FIN_FEATURE_KEYS}
        result = _predict_with_tlstm(features=features, sector=None, previous_rating=None, lang=lang)
        return result
    except Exception as exc:
        log.exception("TLSTMFuzzy prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=msg("prediction_error", lang)) from exc


@router.post("/predict/tlstm", summary="TLSTMFuzzy Credit Rating (Compatibility Alias)")
async def predict_rating_tlstm(
    body: PredictTLSTMRequest,
    lang: Lang = Depends(resolve_lang),
) -> dict:
    """
    Predict credit rating (IG / HY / Distressed) using the pre-trained
    **Transformer-BiLSTM + Fuzzy** deep learning model.

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
    _require_tlstm_runtime(lang)

    try:
        features = {
            k: v for k, v in body.model_dump().items()
            if k in _FIN_FEATURE_KEYS
        }

        result = _predict_with_tlstm(
            features=features,
            sector=body.sector,
            previous_rating=body.previous_rating,
            lang=lang,
        )
        return result
    except Exception as exc:
        log.exception("TLSTMFuzzy prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=msg("prediction_error", lang)) from exc
