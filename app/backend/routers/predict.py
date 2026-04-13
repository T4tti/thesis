"""
routers/predict.py — POST /api/predict (single-record credit rating inference).

How to Run:
    curl -X POST http://localhost:8000/api/predict \\
         -H "Content-Type: application/json" \\
         -d '{"current_ratio": 1.5, "debt_equity_ratio": 0.8, "gross_profit_margin": 0.35}'

Expected Output:
    {"rating": "IG", "probabilities": {...}, "confidence": 0.75, ...}
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from model.predictor import predict
from state import MODEL_STATE

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["predict"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """All 12 financial features — all optional, missing values imputed by model."""

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

    model_config = {"json_schema_extra": {"example": {
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
    }}}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/predict")
async def predict_rating(body: PredictRequest) -> dict:
    """
    Predict credit rating class (IG / HY / Distressed) from financial ratios.

    All fields are optional — missing values are imputed using the training
    set median. At least 3-4 features are recommended for meaningful results.
    """
    if not MODEL_STATE.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="Model is still being trained. Please try again in a few seconds.",
        )

    try:
        result = predict(
            features=body.model_dump(),
            pipeline=MODEL_STATE["pipeline"],
            label_encoder=MODEL_STATE["label_encoder"],
        )
        return result
    except Exception as exc:
        log.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
