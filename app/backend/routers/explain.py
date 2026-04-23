"""
routers/explain.py - AI explanation endpoint backed by Gemini API.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from model.gemini_explainer import generate_gemini_explanation, get_gemini_model_name
from model.features import FEATURES

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["explain"])


class ExplainRequest(BaseModel):
    """AI explanation input payload."""

    features: Dict[str, Optional[float]] = Field(
        ...,
        description="Financial input fields used for prediction.",
    )
    prediction: Dict[str, Any] = Field(
        ...,
        description="Prediction response from /api/predict or /api/predict/tlstm.",
    )
    tlstm_prediction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Explicit TLSTM prediction payload to strengthen explain grounding.",
    )
    lang: Literal["vi", "en"] = Field(
        "vi",
        description="Output language for explanation.",
    )
    xai_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured xAI context to be synthesized by the LLM.",
    )


@router.post("/explain", summary="Generate AI explanation with Gemini")
async def explain_rating(body: ExplainRequest) -> dict:
    """
    Generate a natural-language explanation for a predicted rating using Gemini.

    Requires:
    - `GEMINI_API_KEY` in backend environment.
    Optional:
    - `GEMINI_MODEL` (default: gemini-3.1-flash-lite-preview)
    """
    try:
        normalized_features = {feat: body.features.get(feat) for feat in FEATURES}
        explanation = generate_gemini_explanation(
            features=normalized_features,
            prediction=body.prediction,
            lang=body.lang,
            xai_context=body.xai_context,
            tlstm_prediction=body.tlstm_prediction,
        )
        return {
            "provider": "gemini",
            "model": get_gemini_model_name(),
            "explanation": explanation,
            "xai_used": bool(body.xai_context),
        }
    except RuntimeError as exc:
        msg = str(exc)
        msg_l = msg.lower()
        if "api key is missing" in msg_l:
            status = 503
        elif "quota exceeded" in msg_l or "resource_exhausted" in msg_l:
            status = 429
        elif "model" in msg_l and ("not found" in msg_l or "not available" in msg_l):
            status = 400
        else:
            status = 502
        raise HTTPException(status_code=status, detail=msg) from exc
    except Exception as exc:
        log.exception("Gemini explain failed: %s", exc)
        raise HTTPException(status_code=500, detail="Explain failed due to an internal error.") from exc
