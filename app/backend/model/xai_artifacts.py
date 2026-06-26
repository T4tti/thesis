"""
Resolve artifact-backed xAI context for prediction explanations.

Local artifacts are used only on exact identity matches. For unmatched inputs,
the resolver falls back to global class-level importance plus deterministic
financial-ratio drivers so the UI never presents proxy values as local SHAP/LIME.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from model.features import FEATURES

ARTIFACT_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "DMF" / "credit_rating_artifacts"
LOCAL_SHAP_PATH = ARTIFACT_ROOT / "dmf_dcs_financial_shap_local_decisions.csv"
LOCAL_LIME_PATH = ARTIFACT_ROOT / "dmf_dcs_financial_lime_local_decisions.csv"
GLOBAL_SHAP_PATH = ARTIFACT_ROOT / "dmf_dcs_financial_shap_importance_by_class.csv"

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


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_date(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text[:10]
    return parsed.strftime("%Y-%m-%d")


@lru_cache(maxsize=8)
def _read_artifact(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path, encoding="utf-8")


def _sort_by_rank(df: pd.DataFrame) -> pd.DataFrame:
    if "rank" not in df.columns:
        return df
    ranked = df.copy()
    ranked["_rank_num"] = pd.to_numeric(ranked["rank"], errors="coerce")
    return ranked.sort_values("_rank_num", na_position="last")


def _probability_ranking(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    probs = prediction.get("probabilities") or {}
    rows: List[Dict[str, Any]] = []
    for label, value in probs.items():
        prob = _to_float_or_none(value)
        rows.append({"class": str(label), "probability": None if prob is None else round(prob, 6)})
    rows.sort(key=lambda item: item["probability"] if item["probability"] is not None else -1.0, reverse=True)
    return rows


def _feature_coverage(features: Dict[str, Optional[float]]) -> Dict[str, Any]:
    missing = [feature for feature in FEATURES if _to_float_or_none(features.get(feature)) is None]
    return {
        "provided_count": len(FEATURES) - len(missing),
        "missing_count": len(missing),
        "missing_features": missing,
    }


def _finance_proxy_drivers(features: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    drivers: List[Dict[str, Any]] = []
    for feature in FEATURES:
        value = _to_float_or_none(features.get(feature))
        rule = FEATURE_RISK_RULES.get(feature)
        if value is None or rule is None:
            continue

        neutral = float(rule["neutral"])
        denom = abs(neutral) if abs(neutral) > 1e-6 else 1.0
        deviation_ratio = abs(value - neutral) / denom
        risky_when = str(rule["risky_when"])
        increases_risk = (value > neutral) if risky_when == "higher" else (value < neutral)
        signed_score = min(2.5, deviation_ratio) * (1.0 if increases_risk else -1.0)

        drivers.append(
            {
                "feature": feature,
                "label": rule.get("label") or feature,
                "value": round(value, 6),
                "neutral_reference": neutral,
                "risky_when": risky_when,
                "impact_direction": "increases_risk" if increases_risk else "reduces_risk",
                "shap_proxy_score": round(signed_score, 6),
                "abs_impact_strength": round(abs(signed_score), 6),
                "source": "finance_rule_proxy",
            }
        )

    drivers.sort(key=lambda item: float(item["abs_impact_strength"]), reverse=True)
    return drivers[:6]


def _input_context(prediction: Dict[str, Any], client_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    prediction_context = prediction.get("input_context")
    if isinstance(prediction_context, dict):
        context.update({k: v for k, v in prediction_context.items() if _clean(v)})
    if isinstance(client_context, dict):
        nested = client_context.get("input_context")
        if isinstance(nested, dict):
            context.update({k: v for k, v in nested.items() if _clean(v)})
        for key in ("row_id", "ticker", "company_name", "rating_date", "sector", "previous_rating"):
            if _clean(client_context.get(key)):
                context[key] = client_context[key]
    return context


def _match_local(df: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        return df, "local_artifact_missing"

    row_id = _clean(context.get("row_id"))
    if row_id and "row_id" in df.columns:
        matched = df[df["row_id"].astype(str).str.strip() == row_id]
        if not matched.empty:
            return matched, "exact_row_id"

    ticker = _clean(context.get("ticker")).upper()
    rating_date = _normalize_date(context.get("rating_date"))
    if ticker and rating_date and {"ticker", "rating_date"}.issubset(df.columns):
        ticker_mask = df["ticker"].astype(str).str.strip().str.upper() == ticker
        date_mask = df["rating_date"].map(_normalize_date) == rating_date
        matched = df[ticker_mask & date_mask]
        if not matched.empty:
            return matched, "exact_ticker_date"

    return df.iloc[0:0], "no_local_match"


def _rows_for_class(df: pd.DataFrame, rating: str) -> pd.DataFrame:
    if df.empty or "explained_class" not in df.columns:
        return df.iloc[0:0]
    return df[df["explained_class"].astype(str).str.strip().str.lower() == rating.lower()]


def _shap_rows(df: pd.DataFrame, limit: int = 6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows

    for _, row in _sort_by_rank(df).head(limit).iterrows():
        attribution = _to_float_or_none(row.get("gradientshap_attribution"))
        probability = _to_float_or_none(row.get("explained_class_model_probability"))
        rows.append(
            {
                "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
                "feature": _clean(row.get("feature")),
                "label": _clean(row.get("feature")),
                "value": _to_float_or_none(row.get("feature_value")),
                "attribution": None if attribution is None else round(attribution, 8),
                "abs_attribution": None if attribution is None else round(abs(attribution), 8),
                "impact_direction": _clean(row.get("direction")),
                "explained_class": _clean(row.get("explained_class")),
                "explained_probability": None if probability is None else round(probability, 6),
                "xai_method": _clean(row.get("xai_method")),
                "source": "artifact_local_shap",
            }
        )
    return rows


def _lime_rows(df: pd.DataFrame, limit: int = 6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows

    for _, row in _sort_by_rank(df).head(limit).iterrows():
        weight = _to_float_or_none(row.get("lime_weight_for_explained_class"))
        probability = _to_float_or_none(row.get("explained_class_lime_prediction_probability"))
        rows.append(
            {
                "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
                "lime_rule": _clean(row.get("lime_rule")),
                "weight": None if weight is None else round(weight, 8),
                "abs_weight": None if weight is None else round(abs(weight), 8),
                "impact_direction": _clean(row.get("direction")),
                "explained_class": _clean(row.get("explained_class")),
                "lime_probability": None if probability is None else round(probability, 6),
                "xai_method": "LIME",
                "source": "artifact_local_lime",
            }
        )
    return rows


def _global_rows(rating: str, limit: int = 6) -> List[Dict[str, Any]]:
    df = _read_artifact(str(GLOBAL_SHAP_PATH))
    if df.empty:
        return []
    if "class_name" in df.columns:
        df = df[df["class_name"].astype(str).str.strip().str.lower() == rating.lower()]
    rows: List[Dict[str, Any]] = []
    for _, row in _sort_by_rank(df).head(limit).iterrows():
        mean_abs = _to_float_or_none(row.get("mean_abs_gradientshap"))
        mean_signed = _to_float_or_none(row.get("mean_signed_gradientshap"))
        rows.append(
            {
                "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
                "feature": _clean(row.get("feature")),
                "class_name": _clean(row.get("class_name")),
                "mean_abs_gradientshap": None if mean_abs is None else round(mean_abs, 8),
                "mean_signed_gradientshap": None if mean_signed is None else round(mean_signed, 8),
                "xai_method": _clean(row.get("xai_method")),
                "source": "artifact_global_shap",
            }
        )
    return rows


def resolve_xai_context(
    features: Dict[str, Optional[float]],
    prediction: Dict[str, Any],
    lang: str = "vi",
    client_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return structured xAI context with explicit local/global provenance."""
    rating = _clean(prediction.get("rating")) or "Unknown"
    context = _input_context(prediction, client_context)
    shap_df = _read_artifact(str(LOCAL_SHAP_PATH))
    lime_df = _read_artifact(str(LOCAL_LIME_PATH))

    shap_match, match_status = _match_local(shap_df, context)
    lime_match, lime_match_status = _match_local(lime_df, context)
    if match_status == "no_local_match" and lime_match_status != "no_local_match":
        match_status = lime_match_status

    shap_class = _rows_for_class(shap_match, rating)
    lime_class = _rows_for_class(lime_match, rating)
    local_shap = _shap_rows(shap_class)
    local_lime = _lime_rows(lime_class)
    global_drivers = _global_rows(rating)
    proxy_drivers = _finance_proxy_drivers(features)

    local_available = bool(local_shap or local_lime)
    if local_available:
        xai_source = "artifact_local_shap_lime"
        note = (
            "Exact local SHAP/LIME artifact rows were matched by row_id or ticker+rating_date."
            if lang == "en"
            else "Da khop dung dong artifact SHAP/LIME local bang row_id hoac ticker+rating_date."
        )
        shap_style_top_drivers = local_shap or proxy_drivers
    else:
        if global_drivers:
            xai_source = "artifact_global_plus_finance_proxy"
            note = (
                "No exact local xAI row matched; using global class-level SHAP importance and finance-rule local proxy."
                if lang == "en"
                else "Khong khop dong xAI local; dung SHAP global theo lop va proxy tai chinh cuc bo."
            )
        else:
            xai_source = "finance_proxy_only"
            note = (
                "No xAI artifact was available; using deterministic finance-rule proxy only."
                if lang == "en"
                else "Khong co artifact xAI kha dung; chi dung proxy tai chinh xac dinh."
            )
        shap_style_top_drivers = proxy_drivers

    return {
        "source": xai_source,
        "xai_source": xai_source,
        "xai_match_status": match_status,
        "xai_mode": "artifact-backed" if local_available else "artifact-global-plus-finance-proxy",
        "xai_note": note,
        "input_context": context,
        "prediction_context": {
            "rating": prediction.get("rating"),
            "confidence": _to_float_or_none(prediction.get("confidence")),
            "risk_score": _to_float_or_none(prediction.get("risk_score")),
            "selected_model": prediction.get("selected_model"),
            "dcs_case": prediction.get("dcs_case"),
            "decision_context": prediction.get("decision_context") or {},
        },
        "feature_coverage": _feature_coverage(features),
        "probability_ranking": _probability_ranking(prediction),
        "artifact_shap_drivers": local_shap,
        "artifact_lime_rules": local_lime,
        "artifact_global_drivers": global_drivers,
        "finance_proxy_drivers": proxy_drivers,
        "shap_style_top_drivers": shap_style_top_drivers,
        "confidence_band": _confidence_band(_to_float_or_none(prediction.get("confidence"))),
        "risk_level": prediction.get("risk_level"),
        "risk_score": _to_float_or_none(prediction.get("risk_score")),
    }


def _confidence_band(confidence: Optional[float]) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"
