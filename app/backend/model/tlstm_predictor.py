# -*- coding: utf-8 -*-
"""
model/tlstm_predictor.py — Inference wrapper for the pre-trained TLSTMFuzzyClassifier.

Responsibilities:
    - Load model weights (.pt) + metadata (.pt) from disk
    - Preprocess a single financial record (imputation -> clip -> scale -> delta=0)
    - Run forward pass and return a rich prediction dict

How to Run (standalone smoke test):
    python -m model.tlstm_predictor

Expected Output:
    {
        "model": "TLSTMFuzzy",
        "rating": "IG",
        "probabilities": {"Distressed": 0.05, "HY": 0.20, "IG": 0.75},
        "confidence": 0.75,
        "risk_level": "low",
        ...
    }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from model.tlstm_architecture import TLSTMFuzzyClassifier

log = logging.getLogger(__name__)

# Mapping sector ID (int) to sector name (str)
SECTOR_MAPPING = {
    0: "Basic Industries", 1: "Capital Goods", 2: "Consumer Durables",
    3: "Consumer Non-Durables", 4: "Consumer Services", 5: "Energy",
    6: "Finance", 7: "Health Care", 8: "Miscellaneous",
    9: "Public Utilities", 10: "Technology", 11: "Transportation", 12: "__MISSING__"
}

# ---------------------------------------------------------------------------
# Paths (relative to this file's directory)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

def _get_repo_root(p: Path) -> Path:
    """Safely find repo root (3 levels up) or stop at root."""
    try:
        return p.parents[2]
    except (IndexError, ValueError):
        # Fallback to current or parent if not deep enough (e.g. in Docker)
        return p.parent

_REPO_ROOT = _get_repo_root(_HERE)
_DEFAULT_META_PT = _HERE / "transformer_best_model_meta.pt"
_DEFAULT_META_JSON = _HERE / "tlstm_meta.json"
_DEFAULT_LABEL_CLASSES = ["Distressed", "HY", "IG"]


def _resolve_ckpt() -> Path:
    """Locate checkpoint: prefer local model dir, fallback to repo artifacts."""
    candidates = _candidate_ckpt_paths()
    if candidates:
        return candidates[0]

    local = _HERE / "transformer_best_model.pt"
    repo_candidates = [
        _REPO_ROOT / "artifacts" / "TLSTM" / "TLSTM" / "transformer_best_model.pt",
        _REPO_ROOT / "artifacts" / "TLSTM" / "transformer_best_model.pt",
    ]
    raise FileNotFoundError(
        "transformer_best_model.pt not found.\n"
        f"Checked: {local}\n"
        f"Checked: {repo_candidates[0]}\n"
        f"Checked: {repo_candidates[1]}"
    )


def _candidate_ckpt_paths() -> list[Path]:
    candidates = [
        _HERE / "transformer_best_model.pt",
        _REPO_ROOT / "artifacts" / "TLSTM" / "TLSTM" / "transformer_best_model.pt",
        _REPO_ROOT / "artifacts" / "TLSTM" / "transformer_best_model.pt",
    ]
    seen: set[str] = set()
    existing: list[Path] = []
    for p in candidates:
        p_key = str(p)
        if p_key in seen:
            continue
        seen.add(p_key)
        if p.exists():
            existing.append(p)
    return existing


def _resolve_meta_pt() -> Path:
    """Locate metadata .pt: prefer local model dir, fallback to repo artifacts."""
    local = _HERE / "transformer_best_model_meta.pt"
    if local.exists():
        return local

    repo_candidates = [
        _REPO_ROOT / "artifacts" / "TLSTM" / "TLSTM" / "transformer_best_model_meta.pt",
        _REPO_ROOT / "artifacts" / "TLSTM" / "transformer_best_model_meta.pt",
    ]
    for repo_artifact in repo_candidates:
        if repo_artifact.exists():
            return repo_artifact

    raise FileNotFoundError(
        "transformer_best_model_meta.pt not found.\n"
        f"Checked: {local}\n"
        f"Checked: {repo_candidates[0]}\n"
        f"Checked: {repo_candidates[1]}"
    )


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _labels_from_encoder(raw_classes: Any, n_classes: int) -> list[str]:
    if isinstance(raw_classes, list) and len(raw_classes) == n_classes:
        normalized = [str(c) for c in raw_classes]
        try:
            numeric = [int(x) for x in normalized]
            if n_classes == 3 and set(numeric) == {0, 1, 2}:
                return _DEFAULT_LABEL_CLASSES.copy()
            return [f"class_{i}" for i in numeric]
        except Exception:
            return normalized

    if n_classes == 3:
        return _DEFAULT_LABEL_CLASSES.copy()
    return [f"class_{i}" for i in range(n_classes)]


def _build_meta_from_pt(meta_pt: Dict[str, Any], legacy_json_meta: Dict[str, Any]) -> Dict[str, Any]:
    model_features = meta_pt.get("model_features") or legacy_json_meta.get("model_features")
    if not isinstance(model_features, list) or not model_features:
        raise ValueError("Invalid metadata: missing model_features in transformer_best_model_meta.pt")
    model_features = [str(f) for f in model_features]

    financial_features = [f for f in model_features if not f.endswith("_delta")]
    delta_features = [f for f in model_features if f.endswith("_delta")]

    n_classes = int(meta_pt["n_classes"])
    n_sectors = int(meta_pt["n_sectors"])
    label_classes = _labels_from_encoder(meta_pt.get("label_encoder_classes"), n_classes)
    label_to_id = {name: idx for idx, name in enumerate(label_classes)}

    sector_classes = meta_pt.get("sector_classes")
    if not isinstance(sector_classes, list) or not sector_classes:
        sector_classes = legacy_json_meta.get("sector_classes", [])
    sector_classes = [str(s) for s in sector_classes]
    if len(sector_classes) != n_sectors:
        if len(sector_classes) > n_sectors:
            sector_classes = sector_classes[:n_sectors]
        else:
            sector_classes.extend([f"Sector_{i}" for i in range(len(sector_classes), n_sectors)])

    def _dict_from_legacy(key: str, default_value: float) -> Dict[str, float]:
        raw = legacy_json_meta.get(key, {})
        if not isinstance(raw, dict):
            raw = {}
        return {col: float(raw.get(col, default_value)) for col in financial_features}

    scaler_center_raw = legacy_json_meta.get("scaler_center", [])
    scaler_scale_raw = legacy_json_meta.get("scaler_scale", [])
    if not isinstance(scaler_center_raw, list) or len(scaler_center_raw) != len(model_features):
        scaler_center = [0.0] * len(model_features)
    else:
        scaler_center = [float(x) for x in scaler_center_raw]

    if not isinstance(scaler_scale_raw, list) or len(scaler_scale_raw) != len(model_features):
        scaler_scale = [1.0] * len(model_features)
    else:
        scaler_scale = [float(x) if float(x) != 0.0 else 1.0 for x in scaler_scale_raw]

    legacy_hparams = legacy_json_meta.get("model_hparams", {})
    if not isinstance(legacy_hparams, dict):
        legacy_hparams = {}

    model_hparams = {
        "n_channels": int(meta_pt["n_channels"]),
        "n_classes": n_classes,
        "n_sectors": n_sectors,
        "hidden_size": int(meta_pt["lstm_hidden"]),
        "dropout": float(meta_pt["dropout"]),
        "n_mfs": int(meta_pt["fuzzy_mfs"]),
        "d_model": int(meta_pt["d_model"]),
        "n_heads": int(meta_pt["transformer_heads"]),
        "n_layers": int(meta_pt["transformer_layers"]),
        "sector_emb_dim": int(meta_pt["sector_emb_dim"]),
        "max_relative_position": int(legacy_hparams.get("max_relative_position", 32)),
        "input_size": int(meta_pt.get("input_size", legacy_hparams.get("input_size", 1))),
    }

    merged_meta: Dict[str, Any] = {
        "financial_features": financial_features,
        "delta_features": delta_features,
        "model_features": model_features,
        "label_classes": label_classes,
        "label_to_id": label_to_id,
        "sector_classes": sector_classes,
        "n_sectors": n_sectors,
        "impute_medians": _dict_from_legacy("impute_medians", 0.0),
        "clip_lowers": _dict_from_legacy("clip_lowers", -1e9),
        "clip_uppers": _dict_from_legacy("clip_uppers", 1e9),
        "scaler_center": scaler_center,
        "scaler_scale": scaler_scale,
        "model_hparams": model_hparams,
        "best_epoch": int(meta_pt.get("best_epoch", -1)),
        "best_metric_name": str(meta_pt.get("best_metric_name", "val_f1_weighted")),
        "best_metric_value": float(meta_pt.get("best_metric_value", 0.0)),
        "minority_threshold_enabled": bool(meta_pt.get("minority_threshold_enabled", False)),
        "minority_threshold_class": int(meta_pt.get("minority_threshold_class", 0)),
        "minority_threshold_value": float(meta_pt.get("minority_threshold_value", 0.5)),
        "minority_threshold_score": float(meta_pt.get("minority_threshold_score", 0.0)),
    }
    return merged_meta


def _load_runtime_meta(meta_path: Path, fallback_json_path: Path = _DEFAULT_META_JSON) -> Dict[str, Any]:
    """Load metadata from .pt (preferred) or .json (compatibility mode)."""
    if not meta_path.exists():
        raise FileNotFoundError(f"TLSTM metadata not found: {meta_path}")

    if meta_path.suffix.lower() == ".json":
        with open(meta_path, encoding="utf-8") as f:
            meta_json: Dict[str, Any] = json.load(f)
        return meta_json

    raw = torch.load(meta_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid metadata format in {meta_path}: expected dict, got {type(raw).__name__}")

    legacy_json_meta = _load_json_if_exists(fallback_json_path)
    if not legacy_json_meta:
        log.warning(
            "No fallback JSON metadata found at %s. "
            "Preprocessing stats will use safe defaults.",
            fallback_json_path,
        )

    return _build_meta_from_pt(raw, legacy_json_meta)

# ---------------------------------------------------------------------------
# Risk metadata per output class
# ---------------------------------------------------------------------------

RISK_META: Dict[str, Dict[str, Any]] = {
    "IG": {
        "risk_level":      "low",
        "risk_score_base": 25,
        "color":           "#10b981",
        "label_vi":        "Đầu tư an toàn",
        "label_en":        "Investment Grade",
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
        "risk_level":      "medium",
        "risk_score_base": 55,
        "color":           "#f59e0b",
        "label_vi":        "Sinh lợi cao (Đầu cơ)",
        "label_en":        "High Yield (Speculative)",
        "interp_en": (
            "This entity exhibits elevated credit risk characteristics typical of "
            "high-yield issuers. Leverage and/or cash-flow metrics signal heightened "
            "sensitivity to economic downturns."
        ),
        "interp_vi": (
            "Doanh nghiệp thể hiện rủi ro tín dụng cao hơn, điển hình cho nhóm đầu cơ. "
            "Các chỉ số đòn bẩy và/hoặc dòng tiền cho thấy độ nhạy cảm cao trước biến "
            "động kinh tế."
        ),
    },
    "Distressed": {
        "risk_level":      "high",
        "risk_score_base": 82,
        "color":           "#ef4444",
        "label_vi":        "Căng thẳng tài chính",
        "label_en":        "Distressed",
        "interp_en": (
            "This entity shows significant indicators of financial distress. "
            "Key metrics including leverage, profitability, and/or cash-flow indicate "
            "a materially elevated probability of default or credit deterioration."
        ),
        "interp_vi": (
            "Doanh nghiệp có dấu hiệu căng thẳng tài chính đáng kể. Các chỉ số then "
            "chốt cho thấy xác suất suy giảm tín dụng hoặc vỡ nợ cao. Khuyến nghị "
            "thẩm định kỹ trước khi đầu tư."
        ),
    },
}

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_tlstm(
    ckpt_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    device: str               = "cpu",
) -> Tuple[TLSTMFuzzyClassifier, Dict[str, Any]]:
    """Load model weights and metadata from disk.

    Args:
        ckpt_path : Path to ``transformer_best_model.pt`` (state_dict).
        meta_path : Path to metadata file (default: ``transformer_best_model_meta.pt``).
        device    : Torch device string ("cpu" or "cuda").

    Returns:
        (model, meta) — model is in eval() mode.
    """
    # ── 0. Resolve checkpoint path candidates ─────────────────────────────────
    if ckpt_path is None:
        ckpt_candidates = _candidate_ckpt_paths()
        if not ckpt_candidates:
            _ = _resolve_ckpt()  # raise with detailed message
    else:
        ckpt_candidates = [ckpt_path]

    if meta_path is None:
        meta_path = _resolve_meta_pt()

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    meta: Dict[str, Any] = _load_runtime_meta(meta_path)

    log.info("[TLSTMLoader] Metadata loaded: %d classes, %d sectors, %d features",
             meta["model_hparams"]["n_classes"],
             meta["model_hparams"]["n_sectors"],
             len(meta["model_features"]))

    # ── 2. Reconstruct model ──────────────────────────────────────────────────
    hp = meta["model_hparams"]
    def _build_model() -> TLSTMFuzzyClassifier:
        return TLSTMFuzzyClassifier(
            n_channels           = hp["n_channels"],
            n_classes            = hp["n_classes"],
            n_sectors            = hp["n_sectors"],
            hidden_size          = hp["hidden_size"],
            dropout              = hp["dropout"],
            n_mfs                = hp["n_mfs"],
            d_model              = hp["d_model"],
            n_heads              = hp["n_heads"],
            n_layers             = hp["n_layers"],
            sector_emb_dim       = hp["sector_emb_dim"],
            max_relative_position= hp["max_relative_position"],
        )

    # ── 3. Load weights (compatibility fallback across available checkpoints) ─
    model: Optional[TLSTMFuzzyClassifier] = None
    selected_ckpt: Optional[Path] = None
    load_errors: list[str] = []

    for candidate in ckpt_candidates:
        if not candidate.exists():
            continue
        trial_model = _build_model()
        state_dict = torch.load(candidate, map_location=device, weights_only=True)
        try:
            trial_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            load_errors.append(f"{candidate}: {exc}")
            continue
        model = trial_model
        selected_ckpt = candidate
        break

    if model is None or selected_ckpt is None:
        details = "\n\n".join(load_errors) if load_errors else "No compatible checkpoint candidates found."
        raise RuntimeError(
            "Failed to load TLSTM checkpoint with metadata from transformer_best_model_meta.pt.\n"
            f"Tried checkpoints: {[str(p) for p in ckpt_candidates]}\n\n{details}"
        )

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log.info("[TLSTMLoader] Model loaded: %s — %d params", selected_ckpt.name, n_params)
    return model, meta


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_features(
    features: Dict[str, Optional[float]],
    meta: Dict[str, Any],
) -> np.ndarray:
    """Impute → clip → scale → append zeros for delta features.

    Returns:
        X : np.ndarray shape (1, 1, n_channels)   — (batch=1, T=1, C)
    """
    fin_feats   : list[str]   = meta["financial_features"]
    model_feats : list[str]   = meta["model_features"]
    medians     : Dict[str, float] = meta["impute_medians"]
    clip_lo     : Dict[str, float] = meta["clip_lowers"]
    clip_hi     : Dict[str, float] = meta["clip_uppers"]
    center      : list[float]      = meta["scaler_center"]
    scale       : list[float]      = meta["scaler_scale"]

    # Build raw vector for all 24 features
    raw = np.zeros(len(model_feats), dtype=np.float32)

    for i, col in enumerate(fin_feats):
        val = features.get(col)
        raw[i] = float(val) if val is not None else float(medians.get(col, 0.0))
        raw[i] = float(np.clip(raw[i], clip_lo.get(col, -1e9), clip_hi.get(col, 1e9)))

    # Delta features = 0 (single time-step, no prior context)
    for i in range(len(fin_feats), len(model_feats)):
        raw[i] = 0.0

    # RobustScaler: (x - center) / scale
    c_arr = np.array(center, dtype=np.float32)
    s_arr = np.array(scale,  dtype=np.float32)
    s_arr = np.where(s_arr == 0, 1.0, s_arr)
    raw = (raw - c_arr) / s_arr

    return raw.reshape(1, 1, -1)  # (B=1, T=1, C)


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_tlstm(
    features       : Dict[str, Optional[float]],
    sector         : Optional[str],
    previous_rating: Optional[str],
    model          : TLSTMFuzzyClassifier,
    meta           : Dict[str, Any],
    device         : str = "cpu",
) -> Dict[str, Any]:
    """Run single-record inference with the TLSTMFuzzy model.

    Args:
        features        : Dict of financial feature name → float (None = imputed).
        sector          : Sector string (e.g. "Finance"). None → defaults to "Miscellaneous".
        previous_rating : Last known rating label ("IG"/"HY"/"Distressed"). None → "IG".
        model           : Loaded TLSTMFuzzyClassifier in eval() mode.
        meta            : Metadata dict from ``load_tlstm()``.
        device          : Torch device string.

    Returns:
        Prediction dict with rating, probabilities, confidence, and risk metadata.
    """
    sector_classes : list[str] = meta["sector_classes"]
    label_classes  : list[str] = meta["label_classes"]   # ["Distressed", "HY", "IG"]
    label_to_id    : Dict[str, int] = {k: int(v) for k, v in meta["label_to_id"].items()}

    # ── Resolve sector_id ─────────────────────────────────────────────────────
    raw_sector = (sector or "Miscellaneous").strip()
    resolved_sector = raw_sector

    # If it's a numeric ID, map it first
    try:
        sector_id_input = int(raw_sector)
        resolved_sector = SECTOR_MAPPING.get(sector_id_input, raw_sector)
    except ValueError:
        pass

    if resolved_sector not in sector_classes:
        # Fuzzy fallback: case-insensitive substring match
        resolved_sector = next(
            (s for s in sector_classes if resolved_sector.lower() in s.lower()),
            sector_classes[0],
        )
    
    try:
        sector_id = int(sector_classes.index(resolved_sector))
    except ValueError:
        # Fallback to Miscellaneous if not found in metadata classes
        if "Miscellaneous" in sector_classes:
            sector_id = sector_classes.index("Miscellaneous")
        else:
            sector_id = 0

    # ── Resolve last_y ────────────────────────────────────────────────────────
    resolved_prev = (previous_rating or "IG").strip()
    if resolved_prev not in label_to_id:
        resolved_prev = "IG"
    last_y_id = int(label_to_id[resolved_prev])

    # ── Preprocess features ───────────────────────────────────────────────────
    X_np = _preprocess_features(features, meta)             # (1, 1, 24)

    X_t        = torch.tensor(X_np, dtype=torch.float32).to(device)
    last_y_t   = torch.tensor([last_y_id], dtype=torch.long).to(device)
    sector_t   = torch.tensor([sector_id], dtype=torch.long).to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(X_t, last_y_t, sector_t)            # (1, n_classes)

    proba: np.ndarray = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (n_classes,)
    pred_idx = int(np.argmax(proba))
    pred_class: str = label_classes[pred_idx]

    probabilities: Dict[str, float] = {
        cls: round(float(p), 4) for cls, p in zip(label_classes, proba)
    }
    confidence = round(float(proba[pred_idx]), 4)

    meta_risk = RISK_META.get(pred_class, RISK_META["Distressed"])

    # Risk score: blend base with uncertainty
    entropy = -float(np.sum(proba * np.log(proba + 1e-9)))
    max_entropy = float(np.log(len(proba)))
    uncertainty = entropy / max(max_entropy, 1e-9)
    risk_score = round(float(meta_risk["risk_score_base"]) * (1.0 + 0.2 * uncertainty), 1)

    return {
        "model":              "TLSTMFuzzy",
        "rating":             pred_class,
        "probabilities":      probabilities,
        "confidence":         confidence,
        "risk_level":         meta_risk["risk_level"],
        "risk_score":         min(risk_score, 100.0),
        "color":              meta_risk["color"],
        "label_en":           meta_risk["label_en"],
        "label_vi":           meta_risk["label_vi"],
        "interpretation_en":  meta_risk["interp_en"],
        "interpretation_vi":  meta_risk["interp_vi"],
        "sector_resolved":    resolved_sector,
        "previous_rating":    resolved_prev,
    }


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    model, meta = load_tlstm()

    result = predict_tlstm(
        features={
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
        },
        sector="Finance",
        previous_rating="IG",
        model=model,
        meta=meta,
    )

    print("\n=== TLSTMFuzzy Prediction ===")
    pprint.pprint(result)
