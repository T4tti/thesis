"""
model/predictor.py — Inference wrapper for the trained LightGBM pipeline.

How to Run:
    Called internally by routers/predict.py via predict().

Expected Output:
    {
        "rating": "IG",
        "probabilities": {"Distressed": 0.04, "HY": 0.21, "IG": 0.75},
        "confidence": 0.75,
        "risk_level": "low",
        "risk_score": 12.5,
        "interpretation": "..."
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from model.trainer import FEATURES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk-level mapping
# ---------------------------------------------------------------------------

RISK_META: Dict[str, Dict[str, Any]] = {
    "IG": {
        "risk_level": "low",
        "risk_score_base": 25,          # 0-100 composite risk score
        "color": "#10b981",
        "label_vi": "Đầu tư an toàn",
        "label_en": "Investment Grade",
        "interp_en": (
            "This entity demonstrates solid financial fundamentals consistent with "
            "investment-grade credit quality. Liquidity, leverage, and profitability "
            "metrics are within acceptable ranges for institutional investors."
        ),
        "interp_vi": (
            "Doanh nghiệp này có nền tảng tài chính vững chắc, phù hợp với tiêu chuẩn "
            "đầu tư an toàn. Các chỉ số thanh khoản, đòn bẩy và sinh lời nằm trong "
            "ngưỡng chấp nhận được với nhà đầu tư tổ chức."
        ),
    },
    "HY": {
        "risk_level": "medium",
        "risk_score_base": 55,
        "color": "#f59e0b",
        "label_vi": "Sinh lợi cao (Đầu cơ)",
        "label_en": "High Yield (Speculative)",
        "interp_en": (
            "This entity exhibits elevated credit risk characteristics typical of "
            "high-yield issuers. While generating returns, leverage and/or cash-flow "
            "metrics signal heightened sensitivity to economic downturns. Selective "
            "suitability for risk-tolerant investors."
        ),
        "interp_vi": (
            "Doanh nghiệp thể hiện rủi ro tín dụng cao hơn, điển hình cho nhóm đầu cơ. "
            "Mặc dù vẫn tạo ra lợi nhuận, các chỉ số đòn bẩy và/hoặc dòng tiền cho thấy "
            "độ nhạy cảm cao hơn trước biến động kinh tế. Phù hợp với nhà đầu tư chấp "
            "nhận rủi ro."
        ),
    },
    "Distressed": {
        "risk_level": "high",
        "risk_score_base": 82,
        "color": "#ef4444",
        "label_vi": "Căng thẳng tài chính",
        "label_en": "Distressed",
        "interp_en": (
            "This entity shows significant indicators of financial distress. "
            "Key metrics — including leverage, profitability, and/or cash-flow — "
            "indicate a materially elevated probability of default or credit "
            "deterioration. Enhanced due-diligence is strongly recommended."
        ),
        "interp_vi": (
            "Doanh nghiệp có dấu hiệu căng thẳng tài chính đáng kể. Các chỉ số "
            "then chốt — bao gồm đòn bẩy, lợi nhuận và/hoặc dòng tiền — cho thấy "
            "xác suất suy giảm tín dụng hoặc vỡ nợ cao. Khuyến nghị thẩm định kỹ "
            "trước khi đầu tư."
        ),
    },
}


def predict(
    features: Dict[str, Optional[float]],
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """Run single-record inference.

    Args:
        features:      Dict mapping feature name → float value (None = missing).
        pipeline:      Fitted Pipeline from trainer.train_model().
        label_encoder: Fitted LabelEncoder from trainer.train_model().

    Returns:
        Prediction dict with rating, probabilities, confidence, risk metadata.

    Raises:
        RuntimeError: If model is not yet trained.
    """
    if pipeline is None or label_encoder is None:
        raise RuntimeError("Model is not yet trained. Please wait for startup.")

    # Build single-row DataFrame with all features
    row: Dict[str, float] = {}
    for feat in FEATURES:
        val = features.get(feat)
        row[feat] = float(val) if val is not None else np.nan

    X_input = pd.DataFrame([row], columns=FEATURES)

    # Predict
    proba: np.ndarray = pipeline.predict_proba(X_input)[0]   # shape: (n_classes,)
    pred_idx: int = int(np.argmax(proba))
    pred_class: str = label_encoder.classes_[pred_idx]       # e.g. "IG"

    # Build probability dict  {class_name: probability}
    probabilities: Dict[str, float] = {
        cls: round(float(p), 4)
        for cls, p in zip(label_encoder.classes_, proba)
    }

    confidence: float = round(float(proba[pred_idx]), 4)
    meta = RISK_META[pred_class]

    # Risk score: blend base score with uncertainty (entropy-like)
    entropy = -float(np.sum(proba * np.log(proba + 1e-9)))
    max_entropy = float(np.log(len(proba)))
    uncertainty_factor = entropy / max(max_entropy, 1e-9)
    risk_score = round(
        float(meta["risk_score_base"]) * (1.0 + 0.2 * uncertainty_factor), 1
    )

    return {
        "rating":        pred_class,
        "probabilities": probabilities,
        "confidence":    confidence,
        "risk_level":    meta["risk_level"],
        "risk_score":    min(risk_score, 100.0),
        "color":         meta["color"],
        "label_en":      meta["label_en"],
        "label_vi":      meta["label_vi"],
        "interpretation_en": meta["interp_en"],
        "interpretation_vi": meta["interp_vi"],
    }
