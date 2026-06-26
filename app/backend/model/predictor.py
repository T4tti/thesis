"""
backend/model/predictor.py — Localized prediction wrapper.

This module adapts the DMF/DCS predictor output which returns bilingual keys
(`label_en`, `label_vi`, `interpretation_en`, `interpretation_vi`) into a smaller,
backend-localized payload (`label`, `interpretation`) based on request language.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from i18n import Lang, pick
from model.dmf_predictor import predict_dmf_dcs


def predict(
    *,
    features: dict,
    sector: Optional[str],
    previous_rating: Optional[str],
    model: Any,
    meta: dict,
    lang: Lang = "en",
) -> Dict[str, Any]:
    """
    Run DMF/DCS inference and return a localized response.
    """
    raw = predict_dmf_dcs(
        features=features,
        sector=sector,
        previous_rating=previous_rating,
        model=model,
        meta=meta,
    )

    result: Dict[str, Any] = dict(raw)
    result["label"] = pick(raw, "label", lang)
    result["interpretation"] = pick(raw, "interpretation", lang)

    result.pop("label_en", None)
    result.pop("label_vi", None)
    result.pop("interpretation_en", None)
    result.pop("interpretation_vi", None)

    return result
