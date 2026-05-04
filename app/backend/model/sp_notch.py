"""
model/sp_notch.py - Indicative S&P-style AAA→D notch mapping utilities.

Pure-Python helper to map the platform's 3-class rating output (IG/HY/Distressed)
into a more granular, *indicative* 23-notch S&P-style scale using confidence × risk
heuristics. This is not an official agency methodology.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

SP_SCALE: List[str] = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
    "SD",
    "D",
]

IG_NOTCHES: List[str] = SP_SCALE[:10]
HY_NOTCHES: List[str] = ["BB+", "BB", "BB-", "B+", "B", "B-"]
DISTRESSED_NOTCHES: List[str] = ["CCC+", "CCC", "CCC-", "CC", "C", "SD", "D"]


def _confidence_band(confidence: float) -> Literal["high", "medium", "low"]:
    if confidence >= 0.82:
        return "high"
    if confidence >= 0.62:
        return "medium"
    return "low"


def _risk_band(risk_score: float) -> Literal["low", "medium", "high", "very_high"]:
    if risk_score <= 28:
        return "low"
    if risk_score <= 52:
        return "medium"
    if risk_score <= 75:
        return "high"
    return "very_high"


def _map_ig(confidence: float, risk_score: float) -> Tuple[str, str, str]:
    c = _confidence_band(confidence)
    r = _risk_band(risk_score)

    if r == "high":
        return ("BBB-", "BBB-", "BBB")
    if c == "low":
        return ("BBB", "BBB-", "BBB+")
    if c == "high" and r == "low":
        return ("AA", "AA-", "AAA")
    if c == "high" and r == "medium":
        return ("A+", "A-", "AA-")
    if c == "medium" and r == "low":
        return ("A-", "BBB+", "A+")
    if c == "medium" and r == "medium":
        return ("BBB+", "BBB", "A-")
    return ("BBB", "BBB-", "BBB+")


def _map_hy(confidence: float, risk_score: float) -> Tuple[str, str, str]:
    c = _confidence_band(confidence)
    r = _risk_band(risk_score)

    if c == "high" and r == "medium":
        return ("BB", "BB-", "BB+")
    if c == "high" and r == "high":
        return ("BB-", "B+", "BB")
    if c == "medium" and r == "medium":
        return ("B+", "B", "BB-")
    if c == "medium" and r == "high":
        return ("B", "B-", "B+")
    return ("B-", "B-", "B")


def _map_distressed(confidence: float, risk_score: float) -> Tuple[str, str, str]:
    c = _confidence_band(confidence)
    r = _risk_band(risk_score)

    if c == "high" and r == "very_high":
        return ("C", "D", "CC")
    if c == "high" and r == "high":
        return ("CCC-", "CC", "CCC")
    if c == "medium" and r == "very_high":
        return ("CC", "D", "CCC-")
    if c == "medium" and r == "high":
        return ("CCC", "CCC-", "CCC+")
    return ("CCC+", "CCC", "CCC+")


_MIGRATION_NOTES: Dict[str, Dict[str, str]] = {
    "IG": {
        "high": "Strong fundamentals; upgrade pressure if margins improve.",
        "medium": "Stable outlook; watch leverage trend for potential re-rating.",
        "low": "Borderline IG; downgrade risk if operating conditions weaken.",
    },
    "HY": {
        "high": "Solid HY profile; BB upgrade candidate with FCF improvement.",
        "medium": "Speculative grade; refinancing risk at next maturity wall.",
        "low": "Weak HY; CCC migration risk if EBITDA deteriorates further.",
    },
    "Distressed": {
        "high": "Acute distress; selective default or restructuring likely.",
        "medium": "Near-default; covenant breach or liquidity event imminent.",
        "low": "Severe stress; recovery value analysis recommended.",
    },
}


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_sp_notch_context(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a structured S&P-style notch context from 3-class model output.

    Expected prediction keys: rating (str), confidence (float), risk_score (float).
    """
    raw_rating = str(prediction.get("rating") or "").strip()
    rating_class = raw_rating if raw_rating in {"IG", "HY", "Distressed"} else "Unknown"

    confidence = _to_float(prediction.get("confidence"), 0.0)
    risk_score = _to_float(prediction.get("risk_score"), 50.0)

    conf_band = _confidence_band(confidence)
    r_band = _risk_band(risk_score)

    if rating_class == "IG":
        indicative_notch, range_low, range_high = _map_ig(confidence, risk_score)
    elif rating_class == "HY":
        indicative_notch, range_low, range_high = _map_hy(confidence, risk_score)
    elif rating_class == "Distressed":
        indicative_notch, range_low, range_high = _map_distressed(confidence, risk_score)
    else:
        indicative_notch, range_low, range_high = ("BBB", "BBB", "BBB")

    if indicative_notch == range_low == range_high:
        indicative_range = indicative_notch
    else:
        indicative_range = f"{range_high} to {range_low}"

    migration_note = _MIGRATION_NOTES.get(rating_class, {}).get(conf_band, "")

    return {
        "rating_class": rating_class,
        "indicative_notch": indicative_notch,
        "indicative_range": indicative_range,
        "range_low": range_low,
        "range_high": range_high,
        "confidence_band": conf_band,
        "risk_band": r_band,
        "sp_scale": SP_SCALE,
        "disclaimer": (
            "Indicative S&P-style notch only. Not an official agency rating. "
            "Based on 3-class model output mapped via confidence × risk heuristics."
        ),
        "migration_note": migration_note,
    }

