"""
model/gemini_explainer.py - Generate textual credit-risk explanations via Gemini API.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from model.features import FEATURES

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MIN_EXPLANATION_CHARS = 120
RETRY_FINISH_REASONS = {"LENGTH", "MAX_TOKENS", "CONTENT_FILTER"}

SP_GLOBAL_SCALE = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "SD", "D",
]

FEATURE_RISK_RULES: Dict[str, Dict[str, Any]] = {
    "current_ratio": {"neutral": 1.5, "risky_when": "lower", "label": "Current ratio"},
    "debt_equity_ratio": {"neutral": 1.0, "risky_when": "higher", "label": "Debt to equity"},
    "gross_profit_margin": {"neutral": 0.25, "risky_when": "lower", "label": "Gross margin"},
    "operating_profit_margin": {"neutral": 0.12, "risky_when": "lower", "label": "Operating margin"},
    "ebit_margin": {"neutral": 0.10, "risky_when": "lower", "label": "EBIT margin"},
    "pretax_profit_margin": {"neutral": 0.08, "risky_when": "lower", "label": "Pre-tax margin"},
    "net_profit_margin": {"neutral": 0.06, "risky_when": "lower", "label": "Net margin"},
    "asset_turnover": {"neutral": 0.8, "risky_when": "lower", "label": "Asset turnover"},
    "roe": {"neutral": 0.12, "risky_when": "lower", "label": "ROE"},
    "roa": {"neutral": 0.05, "risky_when": "lower", "label": "ROA"},
    "operating_cashflow_ps": {"neutral": 0.0, "risky_when": "lower", "label": "Operating cashflow/share"},
    "free_cashflow_ps": {"neutral": 0.0, "risky_when": "lower", "label": "Free cashflow/share"},
}

log = logging.getLogger(__name__)


def _read_env_file_value(key: str) -> str:
    """Read a single KEY=value from app/.env as a fallback in local dev."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return ""

    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            name, value = text.split("=", 1)
            name = name.strip().lstrip("\ufeff")
            if name != key:
                continue
            return value.strip().strip('"').strip("'")
    except OSError:
        return ""

    return ""


def get_gemini_model_name() -> str:
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model:
        model = os.getenv("GOOGLE_MODEL", "").strip()
    if not model:
        model = _read_env_file_value("GEMINI_MODEL")
    return model or DEFAULT_GEMINI_MODEL


def get_openrouter_model_name() -> str:
    # Backward-compatible alias used by existing imports.
    return get_gemini_model_name()


def _create_gemini_client(api_key: str) -> OpenAI:
    base_url = os.getenv("GEMINI_API_BASE_URL", "").strip()
    if not base_url:
        base_url = _read_env_file_value("GEMINI_API_BASE_URL")
    if not base_url:
        base_url = DEFAULT_GEMINI_BASE_URL
    return OpenAI(base_url=base_url, api_key=api_key)



def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_feature_lines(features: Dict[str, Optional[float]]) -> str:
    lines = []
    for key in FEATURES:
        val = _to_float_or_none(features.get(key))
        if val is None:
            lines.append(f"- {key}: missing")
        else:
            lines.append(f"- {key}: {val:.6g}")
    return "\n".join(lines)


def _confidence_band(confidence: Optional[float]) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _build_shap_style_drivers(features: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    drivers: List[Dict[str, Any]] = []
    for feature in FEATURES:
        val = _to_float_or_none(features.get(feature))
        if val is None:
            continue

        rule = FEATURE_RISK_RULES.get(feature)
        if rule is None:
            continue

        neutral = float(rule["neutral"])
        denom = abs(neutral) if abs(neutral) > 1e-6 else 1.0
        deviation_ratio = abs(val - neutral) / denom

        risky_when = str(rule["risky_when"])
        increases_risk = (val > neutral) if risky_when == "higher" else (val < neutral)
        direction = "increases_risk" if increases_risk else "reduces_risk"
        signed_score = min(2.5, deviation_ratio) * (1.0 if increases_risk else -1.0)

        drivers.append(
            {
                "feature": feature,
                "label": rule.get("label") or feature,
                "value": round(val, 6),
                "neutral_reference": neutral,
                "risky_when": risky_when,
                "impact_direction": direction,
                "shap_proxy_score": round(signed_score, 4),
                "abs_impact_strength": round(abs(signed_score), 4),
            }
        )

    drivers.sort(key=lambda item: float(item["abs_impact_strength"]), reverse=True)
    return drivers[:6]


def _build_sp_rating_context(prediction: Dict[str, Any]) -> Dict[str, Any]:
    rating = str(prediction.get("rating") or "").strip()
    confidence = _to_float_or_none(prediction.get("confidence")) or 0.0
    risk_score = _to_float_or_none(prediction.get("risk_score"))
    risk_score = 50.0 if risk_score is None else risk_score

    if rating == "IG":
        broad_bucket = "Investment grade"
        if confidence >= 0.75 and risk_score <= 30:
            indicative_range = "A- to AAA"
        elif risk_score <= 45:
            indicative_range = "BBB to A+"
        else:
            indicative_range = "BBB- to BBB+"
    elif rating == "HY":
        broad_bucket = "Speculative high-yield"
        if confidence >= 0.7 and risk_score <= 65:
            indicative_range = "BB- to B+"
        elif risk_score <= 78:
            indicative_range = "BB to B"
        else:
            indicative_range = "B- to CCC+"
    elif rating == "Distressed":
        broad_bucket = "Distressed / near-default"
        if confidence >= 0.75 and risk_score >= 88:
            indicative_range = "CC to D"
        elif risk_score >= 78:
            indicative_range = "CCC- to C"
        else:
            indicative_range = "CCC+ to CC"
    else:
        broad_bucket = "Unknown"
        indicative_range = "N/A"

    return {
        "scale": "S&P-style global long-term issuer scale",
        "full_scale_order": SP_GLOBAL_SCALE,
        "predicted_bucket": rating,
        "broad_bucket_description": broad_bucket,
        "indicative_notch_range": indicative_range,
        "disclaimer": (
            "Model output is 3-class (IG/HY/Distressed); mapped S&P notch range is indicative, "
            "not an official agency rating."
        ),
    }


def _build_default_xai_context(
    features: Dict[str, Optional[float]],
    prediction: Dict[str, Any],
    lang: str,
) -> Dict[str, Any]:
    probs = prediction.get("probabilities") or {}
    sorted_probs: List[Dict[str, Any]] = []
    for cls, prob in sorted(probs.items(), key=lambda item: float(item[1]), reverse=True):
        p = _to_float_or_none(prob)
        sorted_probs.append(
            {
                "class": cls,
                "probability": None if p is None else round(p, 4),
            }
        )

    missing = [feat for feat in FEATURES if _to_float_or_none(features.get(feat)) is None]
    provided = [feat for feat in FEATURES if feat not in missing]
    confidence = _to_float_or_none(prediction.get("confidence"))
    shap_style_top_drivers = _build_shap_style_drivers(features)

    return {
        "feature_coverage": {
            "provided_count": len(provided),
            "missing_count": len(missing),
            "missing_features": missing,
        },
        "probability_ranking": sorted_probs,
        "shap_style_top_drivers": shap_style_top_drivers,
        "xai_mode": "rule-based-shap-proxy",
        "xai_note": (
            "SHAP proxy derived from directional distance to neutral financial anchors. "
            "Use as explanatory support, not exact Shapley values."
        ),
        "confidence_band": _confidence_band(confidence),
        "risk_level": prediction.get("risk_level"),
        "risk_score": _to_float_or_none(prediction.get("risk_score")),
        "model_interpretation": (
            prediction.get("interpretation_vi")
            if lang == "vi"
            else prediction.get("interpretation_en")
        ),
    }


def _build_prediction_snapshot(prediction: Dict[str, Any]) -> Dict[str, Any]:
    probs = prediction.get("probabilities") or {}
    sorted_probs = sorted(
        ((str(label), _to_float_or_none(prob)) for label, prob in probs.items()),
        key=lambda item: item[1] if item[1] is not None else -1.0,
        reverse=True,
    )

    return {
        "model": prediction.get("model"),
        "rating": prediction.get("rating"),
        "confidence": _to_float_or_none(prediction.get("confidence")),
        "risk_level": prediction.get("risk_level"),
        "risk_score": _to_float_or_none(prediction.get("risk_score")),
        "sector_resolved": prediction.get("sector_resolved"),
        "previous_rating": prediction.get("previous_rating"),
        "top_probabilities": [
            {
                "class": label,
                "probability": prob,
            }
            for label, prob in sorted_probs[:3]
        ],
    }


def _build_prompt(
    features: Dict[str, Optional[float]],
    prediction: Dict[str, Any],
    lang: str,
    xai_context: Dict[str, Any],
    tlstm_prediction: Dict[str, Any],
) -> str:
    rating = str(prediction.get("rating", "Unknown"))
    confidence = _to_float_or_none(prediction.get("confidence"))
    risk_score = _to_float_or_none(prediction.get("risk_score"))
    probs = prediction.get("probabilities") or {}

    confidence_text = "N/A" if confidence is None else f"{confidence * 100:.2f}%"
    risk_score_text = "N/A" if risk_score is None else f"{risk_score:.1f}/100"
    probs_text = json.dumps(probs, ensure_ascii=False)
    xai_text = json.dumps(xai_context, ensure_ascii=False)
    tlstm_text = json.dumps(_build_prediction_snapshot(tlstm_prediction), ensure_ascii=False)
    primary_text = json.dumps(_build_prediction_snapshot(prediction), ensure_ascii=False)
    sp_context_text = json.dumps(_build_sp_rating_context(prediction), ensure_ascii=False)
    feature_lines = _build_feature_lines(features)

    if lang == "vi":
        return (
            "Ban la chuyen gia phan tich xep hang tin dung doanh nghiep theo phong cach to chuc xep hang quoc te.\n\n"
            "NHIEM VU: Chi tao DUY NHAT noi dung cho phan 'Risk Interpretation'.\n"
            "Rang buoc bat buoc:\n"
            "- Bat buoc dua tren CA HAI nguon: ket qua du doan TLSTM va bo xAI SHAP/co cau truc.\n"
            "- Khong tao section khac, khong lap lai tieu de, khong markdown phuc tap.\n"
            "- Viet 8-12 bullet co chieu sau, tong 220-360 tu.\n"
            "- Neu confidence thap hoac du lieu thieu, phai neu ro muc do bat dinh.\n"
            "- Bat buoc chi ra toi thieu 4 chi so tai chinh then chot va giai thich vi sao chung day hoac ha rui ro.\n"
            "- Bat buoc dien giai theo thang S&P AAA-D o dang dai kha nang dat duoc (khong duoc khang dinh xep hang notch chinh thuc).\n"
            "- Co 4 phan logic trong chuoi bullet: Tong quan rui ro, dong luc SHAP theo chi so, dai S&P AAA-D co the dat, hanh dong uu tien.\n"
            "- Coi ket qua la tham khao AI, khong thay the tham dinh chuyen sau.\n\n"
            f"Du doan chinh tu he thong:\n{primary_text}\n\n"
            f"Ket qua du doan TLSTM can bam sat:\n{tlstm_text}\n\n"
            f"Bo giai thich xAI (co cau truc):\n{xai_text}\n\n"
            f"Khung tham chieu S&P AAA-D de dien giai:\n{sp_context_text}\n\n"
            f"Tom tat ket qua nhanh:\n- rating: {rating}\n- confidence: {confidence_text}\n"
            f"- risk_score: {risk_score_text}\n- probabilities: {probs_text}\n\n"
            f"Du lieu tai chinh dau vao:\n{feature_lines}\n"
        )

    return (
        "You are a senior corporate credit-rating analyst using an S&P-style framework.\n\n"
        "TASK: Produce ONLY the content for a single 'Risk Interpretation' section.\n"
        "Hard constraints:\n"
        "- Ground the answer in BOTH TLSTM prediction output and structured xAI/SHAP context.\n"
        "- No extra sections, no repeated heading, no decorative formatting.\n"
        "- Return 8-12 analyst-style bullets, 220-360 words total.\n"
        "- Explicitly mention uncertainty if confidence is weak or inputs are missing.\n"
        "- Explain at least 4 key financial indicators and why each one increases or reduces credit risk.\n"
        "- Include an indicative S&P AAA-D notch range path (not an official agency notch assignment).\n"
        "- Cover 4 logical blocks: risk overview, SHAP driver analysis, plausible S&P AAA-D range, prioritized actions.\n"
        "- Clarify this is an AI reference interpretation, not full due diligence.\n\n"
        f"Primary system prediction:\n{primary_text}\n\n"
        f"TLSTM prediction to anchor on:\n{tlstm_text}\n\n"
        f"Structured xAI context:\n{xai_text}\n\n"
        f"S&P AAA-D reference context:\n{sp_context_text}\n\n"
        f"Quick numeric snapshot:\n- rating: {rating}\n- confidence: {confidence_text}\n"
        f"- risk_score: {risk_score_text}\n- probabilities: {probs_text}\n\n"
        f"Financial inputs:\n{feature_lines}\n"
    )


def _extract_gemini_text(response: Any) -> tuple[str, str]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("Gemini returned no choices.")

    best_text = ""
    best_finish_reason = ""
    for choice in choices:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        text = str(content or "").strip()
        if len(text) > len(best_text):
            best_text = text
            best_finish_reason = str(getattr(choice, "finish_reason", "") or "").upper()

    if not best_text:
        raise RuntimeError("Gemini returned an empty explanation.")

    return best_text, best_finish_reason


def _request_gemini_completion(
    prompt: str,
    client: OpenAI,
    model_name: str,
    *,
    temperature: float,
    max_output_tokens: int,
) -> tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are precise, evidence-grounded, and conservative. "
                        "Write like a professional corporate credit-rating analyst. "
                        "Never invent ratios, SHAP values, or model outputs that are not provided. "
                        "If notch detail is uncertain, provide a range and state uncertainty explicitly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            top_p=0.9,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
    return _extract_gemini_text(response)


def generate_gemini_explanation(
    features: Dict[str, Optional[float]],
    prediction: Dict[str, Any],
    lang: str = "vi",
    xai_context: Optional[Dict[str, Any]] = None,
    tlstm_prediction: Optional[Dict[str, Any]] = None,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        api_key = _read_env_file_value("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY in backend environment."
        )

    model_name = get_gemini_model_name()
    client = _create_gemini_client(api_key)
    safe_lang = "vi" if str(lang).lower() == "vi" else "en"
    resolved_xai = xai_context or _build_default_xai_context(
        features=features,
        prediction=prediction,
        lang=safe_lang,
    )

    prompt = _build_prompt(
        features=features,
        prediction=prediction,
        lang=safe_lang,
        xai_context=resolved_xai,
        tlstm_prediction=tlstm_prediction or prediction,
    )

    explanation, finish_reason = _request_gemini_completion(
        prompt=prompt,
        client=client,
        model_name=model_name,
        temperature=0.2,
        max_output_tokens=420,
    )

    if len(explanation) < MIN_EXPLANATION_CHARS or finish_reason in RETRY_FINISH_REASONS:
        retry_prompt = (
            f"{prompt}\n\n"
            "Output constraints: return complete final answer in plain text only (no markdown), "
            "with 8-12 concise bullet points and at least 220 words."
        )
        log.warning(
            "Gemini explanation looks incomplete (len=%s, finishReason=%s). Retrying once.",
            len(explanation),
            finish_reason or "UNKNOWN",
        )
        retry_text, retry_finish_reason = _request_gemini_completion(
            prompt=retry_prompt,
            client=client,
            model_name=model_name,
            temperature=0.1,
            max_output_tokens=900,
        )
        if len(retry_text) > len(explanation):
            explanation = retry_text
            finish_reason = retry_finish_reason

    if len(explanation) < 40:
        raise RuntimeError(
            "Gemini returned an incomplete explanation after retry."
        )

    return explanation


def generate_openai_explanation(
    features: Dict[str, Optional[float]],
    prediction: Dict[str, Any],
    lang: str = "vi",
    xai_context: Optional[Dict[str, Any]] = None,
    tlstm_prediction: Optional[Dict[str, Any]] = None,
) -> str:
    # Backward-compatible wrapper with clearer naming for new callers.
    return generate_gemini_explanation(
        features=features,
        prediction=prediction,
        lang=lang,
        xai_context=xai_context,
        tlstm_prediction=tlstm_prediction,
    )
