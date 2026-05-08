"""
routers/explain.py - AI explanation endpoint backed by Gemini API.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model.gemini_explainer import (
    build_explanation_prompt,
    generate_gemini_explanation,
    get_gemini_client,
    get_gemini_model_name,
)
from model.features import FEATURES
from model.sp_notch import build_sp_notch_context

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
    stream: bool = Field(
        default=True,
        description="When true, stream explanation tokens over SSE.",
    )


async def _token_stream(
    prompt: str,
    client: Any,
    model_name: str,
    sp_context: Dict[str, Any],
):
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise corporate credit-rating analyst. "
                        "Write concisely. Never invent data. "
                        "When responding in Vietnamese, always use proper Unicode Vietnamese "
                        "with full diacritical marks (dấu tiếng Việt đầy đủ)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=420,
            top_p=0.9,
            stream=True,
        )

        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if choices is None and isinstance(chunk, dict):
                choices = chunk.get("choices")
            choices = choices or []
            for choice in choices:
                delta = getattr(choice, "delta", None)
                if delta is None and isinstance(choice, dict):
                    delta = choice.get("delta")

                token = None
                if isinstance(delta, dict):
                    token = delta.get("content")
                elif delta is not None:
                    token = getattr(delta, "content", None)

                if not token:
                    continue
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'done': True, 'sp_context': sp_context}, ensure_ascii=False)}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"


@router.post("/explain", summary="Generate AI explanation with Gemini")
async def explain_rating(body: ExplainRequest):
    """
    Generate a natural-language explanation for a predicted rating using Gemini.

    Requires:
    - `GEMINI_API_KEY` in backend environment.
    Optional:
    - `GEMINI_MODEL` (default: gemini-3.1-flash-lite-preview)
    """
    try:
        normalized_features = {feat: body.features.get(feat) for feat in FEATURES}
        sp_context = build_sp_notch_context(body.prediction)
        prompt = build_explanation_prompt(
            features=normalized_features,
            prediction=body.prediction,
            lang=body.lang,
            xai_context=body.xai_context,
            tlstm_prediction=body.tlstm_prediction,
            sp_context=sp_context,
        )

        if body.stream:
            client = get_gemini_client()
            model_name = get_gemini_model_name()
            return StreamingResponse(
                _token_stream(prompt, client, model_name, sp_context),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

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
            "sp_context": sp_context,
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
