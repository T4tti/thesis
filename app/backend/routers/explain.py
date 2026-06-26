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
    build_fallback_explanation,
    build_explanation_prompt,
    generate_gemini_explanation,
    get_gemini_client,
    get_gemini_model_name,
)
from model.features import FEATURES
from model.sp_notch import build_sp_notch_context
from model.xai_artifacts import resolve_xai_context
from i18n_messages import msg

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


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _token_stream(
    prompt: str,
    client: Any,
    model_name: str,
    sp_context: Dict[str, Any],
    xai_context: Dict[str, Any],
    fallback_text: str,
    lang: str,
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
                yield _sse_event({"token": token})

        yield _sse_event({
            'done': True,
            'provider': 'gemini',
            'model': model_name,
            'fallback_used': False,
            'xai_source': xai_context.get('xai_source') or xai_context.get('source'),
            'xai_match_status': xai_context.get('xai_match_status'),
            'sp_context': sp_context,
        })
    except Exception as exc:
        log.warning("Gemini stream failed; using deterministic fallback: %s", exc, exc_info=True)
        yield _sse_event({"token": fallback_text})
        yield _sse_event({
            'done': True,
            'provider': 'local-fallback',
            'model': 'deterministic-financial-xai',
            'fallback_used': True,
            'warning': _safe_explain_message(exc, lang),
            'xai_source': xai_context.get('xai_source') or xai_context.get('source'),
            'xai_match_status': xai_context.get('xai_match_status'),
            'sp_context': sp_context,
        })


async def _fallback_stream(
    fallback_text: str,
    sp_context: Dict[str, Any],
    xai_context: Dict[str, Any],
    warning: str,
):
    yield _sse_event({"token": fallback_text})
    yield _sse_event({
        'done': True,
        'provider': 'local-fallback',
        'model': 'deterministic-financial-xai',
        'fallback_used': True,
        'warning': warning,
        'xai_source': xai_context.get('xai_source') or xai_context.get('source'),
        'xai_match_status': xai_context.get('xai_match_status'),
        'sp_context': sp_context,
    })


def _safe_explain_message(exc: Exception, lang: str) -> str:
    text = str(exc).lower()
    safe_lang = "vi" if str(lang).lower().startswith("vi") else "en"
    if "api key is missing" in text or "google api key" in text:
        return msg("explain_gemini_key_missing", safe_lang)
    if "quota exceeded" in text or "resource_exhausted" in text:
        return msg("explain_gemini_quota", safe_lang)
    if "connection" in text or "network" in text or "timeout" in text:
        return msg("explain_gemini_network", safe_lang)
    return msg("explain_gemini_unavailable", safe_lang)


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
        tlstm_prediction = body.tlstm_prediction or body.prediction.get("tlstm_prediction")
        xai_context = resolve_xai_context(
            features=normalized_features,
            prediction=body.prediction,
            lang=body.lang,
            client_context=body.xai_context,
        )
        prompt = build_explanation_prompt(
            features=normalized_features,
            prediction=body.prediction,
            lang=body.lang,
            xai_context=xai_context,
            tlstm_prediction=tlstm_prediction,
            sp_context=sp_context,
        )
        fallback_text = build_fallback_explanation(
            features=normalized_features,
            prediction=body.prediction,
            lang=body.lang,
            xai_context=xai_context,
            tlstm_prediction=tlstm_prediction,
            sp_context=sp_context,
        )

        if body.stream:
            try:
                client = get_gemini_client()
                model_name = get_gemini_model_name()
                stream = _token_stream(
                    prompt,
                    client,
                    model_name,
                    sp_context,
                    xai_context,
                    fallback_text,
                    body.lang,
                )
            except RuntimeError as exc:
                log.warning("Gemini stream unavailable before request; using fallback: %s", exc)
                stream = _fallback_stream(
                    fallback_text,
                    sp_context,
                    xai_context,
                    _safe_explain_message(exc, body.lang),
                )
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        fallback_used = False
        provider = "gemini"
        model_name = get_gemini_model_name()
        warning = None
        try:
            explanation = generate_gemini_explanation(
                features=normalized_features,
                prediction=body.prediction,
                lang=body.lang,
                xai_context=xai_context,
                tlstm_prediction=tlstm_prediction,
            )
        except RuntimeError as exc:
            log.warning("Gemini explain failed; using deterministic fallback: %s", exc, exc_info=True)
            explanation = fallback_text
            provider = "local-fallback"
            model_name = "deterministic-financial-xai"
            fallback_used = True
            warning = _safe_explain_message(exc, body.lang)

        return {
            "provider": provider,
            "model": model_name,
            "explanation": explanation,
            "xai_used": True,
            "fallback_used": fallback_used,
            "warning": warning,
            "xai_source": xai_context.get("xai_source") or xai_context.get("source"),
            "xai_match_status": xai_context.get("xai_match_status"),
            "xai_context": xai_context,
            "sp_context": sp_context,
        }
    except RuntimeError as exc:
        log.warning("Explain fallback setup failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=_safe_explain_message(exc, body.lang)) from exc
    except Exception as exc:
        log.exception("Gemini explain failed: %s", exc)
        raise HTTPException(status_code=500, detail="Explain failed due to an internal error.") from exc
