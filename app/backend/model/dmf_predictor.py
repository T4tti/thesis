# -*- coding: utf-8 -*-
"""
DMF/DCS runtime adapter for the backend rating endpoint.

The saved DMF artifact is a decision-combination package, not a standalone
GraphSAGE checkpoint. It needs base predictions from T-LSTM and GraphSAGE.
This adapter runs the local T-LSTM model live, then estimates the GraphSAGE
base prediction from the saved DMF reference predictions using nearest
T-LSTM-probability profile matching.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from model.tlstm_predictor import RISK_META, load_tlstm, predict_tlstm

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parents[2]
_DEFAULT_DMF_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "DMF"
    / "credit_rating_artifacts"
    / "dmf_gat_tlstm"
    / "dmf_dcs_final_model.pkl"
)
_DEFAULT_REFERENCE_PATH = _DEFAULT_DMF_PATH.with_name("dmf_dcs_test_predictions.csv")


def _as_probability_vector(
    probabilities: Dict[str, float],
    class_names: list[str],
) -> np.ndarray:
    vec = np.array([float(probabilities.get(name, 0.0)) for name in class_names], dtype=np.float64)
    total = float(vec.sum())
    if total <= 0.0:
        return np.full(len(class_names), 1.0 / max(len(class_names), 1), dtype=np.float64)
    return vec / total


def _risk_payload(pred_class: str, proba: np.ndarray) -> Dict[str, Any]:
    meta_risk = RISK_META.get(pred_class, RISK_META["Distressed"])
    entropy = -float(np.sum(proba * np.log(proba + 1e-9)))
    max_entropy = float(np.log(len(proba)))
    uncertainty = entropy / max(max_entropy, 1e-9)
    risk_score = round(float(meta_risk["risk_score_base"]) * (1.0 + 0.2 * uncertainty), 1)
    return {
        "risk_level": meta_risk["risk_level"],
        "risk_score": min(risk_score, 100.0),
        "color": meta_risk["color"],
        "label_en": meta_risk["label_en"],
        "label_vi": meta_risk["label_vi"],
        "interpretation_en": meta_risk["interp_en"],
        "interpretation_vi": meta_risk["interp_vi"],
    }


def _compact_prediction(class_names: list[str], proba: np.ndarray) -> Dict[str, Any]:
    pred_idx = int(np.argmax(proba))
    pred_class = class_names[pred_idx]
    return {
        "rating": pred_class,
        "pred_label": pred_idx,
        "probabilities": {
            cls: round(float(prob), 4)
            for cls, prob in zip(class_names, proba)
        },
        "confidence": round(float(proba[pred_idx]), 4),
    }


def _load_dmf_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"DMF/DCS artifact not found: {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict):
        raise ValueError(f"Invalid DMF/DCS artifact: expected dict, got {type(artifact).__name__}")
    if artifact.get("artifact_type") != "dmf_dcs_final_model":
        raise ValueError(f"Unexpected DMF/DCS artifact type: {artifact.get('artifact_type')}")
    return artifact


@dataclass
class DmfDcsRuntime:
    tlstm_model: Any
    tlstm_meta: Dict[str, Any]
    artifact: Dict[str, Any]
    artifact_path: Path
    reference_path: Optional[Path]
    reference_tlstm_probs: Optional[np.ndarray]
    reference_gat_probs: Optional[np.ndarray]
    reference_rows: list[Dict[str, Any]]

    @property
    def class_names(self) -> list[str]:
        names = self.artifact.get("class_names") or self.tlstm_meta.get("label_classes") or []
        return [str(name) for name in names]

    @property
    def model_name(self) -> str:
        return "DMF/DCS T-LSTM+GraphSAGE"

    @classmethod
    def load(
        cls,
        artifact_path: Optional[Path] = None,
        reference_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> Tuple["DmfDcsRuntime", Dict[str, Any]]:
        selected_artifact_path = artifact_path or _DEFAULT_DMF_PATH
        artifact = _load_dmf_artifact(selected_artifact_path)
        tlstm_model, tlstm_meta = load_tlstm(device=device)
        selected_reference_path = reference_path or _DEFAULT_REFERENCE_PATH
        reference_tlstm_probs, reference_gat_probs, reference_rows = cls._load_reference(
            selected_reference_path,
            artifact,
        )
        runtime = cls(
            tlstm_model=tlstm_model,
            tlstm_meta=tlstm_meta,
            artifact=artifact,
            artifact_path=selected_artifact_path,
            reference_path=selected_reference_path if selected_reference_path.exists() else None,
            reference_tlstm_probs=reference_tlstm_probs,
            reference_gat_probs=reference_gat_probs,
            reference_rows=reference_rows,
        )
        meta = runtime._build_meta()
        return runtime, meta

    @staticmethod
    def _load_reference(
        reference_path: Path,
        artifact: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list[Dict[str, Any]]]:
        if not reference_path.exists():
            log.warning("[DMFDCSLoader] Reference predictions not found: %s", reference_path)
            return None, None, []

        class_ids = [int(x) for x in artifact.get("class_ids", [0, 1, 2])]
        tlstm_cols = [f"tlstm_prob_{class_id}" for class_id in class_ids]
        gat_cols = [f"gat_prob_{class_id}" for class_id in class_ids]
        meta_cols = ["row_id", "ticker", "company_name", "rating_date", "selected_model", "dcs_case"]
        use_cols = [col for col in [*meta_cols, *tlstm_cols, *gat_cols]]

        df = pd.read_csv(reference_path, encoding="utf-8", usecols=lambda col: col in use_cols)
        missing = [col for col in [*tlstm_cols, *gat_cols] if col not in df.columns]
        if missing:
            raise ValueError(f"DMF/DCS reference predictions missing columns: {missing}")

        tlstm_probs = df[tlstm_cols].to_numpy(dtype=np.float64)
        gat_probs = df[gat_cols].to_numpy(dtype=np.float64)
        row_cols = [col for col in meta_cols if col in df.columns]
        rows = df[row_cols].to_dict(orient="records") if row_cols else []
        log.info("[DMFDCSLoader] Loaded %d GraphSAGE reference rows", len(df))
        return tlstm_probs, gat_probs, rows

    def _build_meta(self) -> Dict[str, Any]:
        metrics = ((self.artifact.get("test_set_summary") or {}).get("final_metrics") or {})
        return {
            **self.tlstm_meta,
            "model_name": self.model_name,
            "artifact_type": self.artifact.get("artifact_type"),
            "artifact_version": self.artifact.get("artifact_version"),
            "artifact_path": str(self.artifact_path),
            "reference_path": str(self.reference_path) if self.reference_path else None,
            "base_models": ["tlstm", "graphsage"],
            "dmf_metrics": metrics,
            "graphsage_runtime": (
                "artifact_neighbor_proxy"
                if self.reference_tlstm_probs is not None and self.reference_gat_probs is not None
                else "tlstm_mirror_fallback"
            ),
        }

    def _competence(self, model_key: str, class_id: int) -> float:
        for row in self.artifact.get("validation_competence", []):
            if row.get("model_key") == model_key and int(row.get("class_id")) == int(class_id):
                return float(row.get("delta_topk_mean", 0.0))
        return 0.0

    def _model_weight(self, model_key: str) -> float:
        config = self.artifact.get("dcs_config") or {}
        weights = config.get("model_weights") or {}
        return float(weights.get(model_key, 1.0))

    def _graphsage_proxy(self, tlstm_proba: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.reference_tlstm_probs is None or self.reference_gat_probs is None:
            return tlstm_proba.copy(), {
                "graphsage_runtime": "tlstm_mirror_fallback",
                "graphsage_reference": None,
            }

        distances = np.sum((self.reference_tlstm_probs - tlstm_proba.reshape(1, -1)) ** 2, axis=1)
        idx = int(np.argmin(distances))
        gat_proba = self.reference_gat_probs[idx].astype(np.float64)
        gat_total = float(gat_proba.sum())
        if gat_total > 0.0:
            gat_proba = gat_proba / gat_total

        row = self.reference_rows[idx] if idx < len(self.reference_rows) else {}
        return gat_proba, {
            "graphsage_runtime": "artifact_neighbor_proxy",
            "graphsage_proxy_distance": round(float(distances[idx]), 8),
            "graphsage_reference": row,
        }

    def _combine(self, tlstm_proba: np.ndarray, gat_proba: np.ndarray) -> Dict[str, Any]:
        config = self.artifact.get("dcs_config") or {}
        tlstm_pred = int(np.argmax(tlstm_proba))
        gat_pred = int(np.argmax(gat_proba))

        if tlstm_pred == gat_pred:
            final_proba = (tlstm_proba + gat_proba) / 2.0
            final_pred = int(np.argmax(final_proba))
            return {
                "proba": final_proba,
                "pred_idx": final_pred,
                "selected_model": "agreement",
                "dcs_case": "agreement",
                "tlstm_score": None,
                "graphsage_score": None,
            }

        tlstm_score = self._competence("tlstm", tlstm_pred) * self._model_weight("tlstm") + float(tlstm_proba[tlstm_pred])
        gat_score = self._competence("gat", gat_pred) * self._model_weight("gat") + float(gat_proba[gat_pred])

        if gat_score > tlstm_score:
            selected = "graphsage"
            final_proba = gat_proba
            final_pred = gat_pred
        elif tlstm_score > gat_score:
            selected = "tlstm"
            final_proba = tlstm_proba
            final_pred = tlstm_pred
        else:
            selected = str(config.get("tie_breaker") or "tlstm")
            final_proba = gat_proba if selected == "gat" else tlstm_proba
            final_pred = gat_pred if selected == "gat" else tlstm_pred

        return {
            "proba": final_proba,
            "pred_idx": int(final_pred),
            "selected_model": "GraphSAGE" if selected in {"gat", "graphsage"} else "T-LSTM",
            "dcs_case": "disagreement",
            "tlstm_score": round(float(tlstm_score), 6),
            "graphsage_score": round(float(gat_score), 6),
        }

    def predict(
        self,
        *,
        features: Dict[str, Optional[float]],
        sector: Optional[str],
        previous_rating: Optional[str],
        device: str = "cpu",
    ) -> Dict[str, Any]:
        tlstm_raw = predict_tlstm(
            features=features,
            sector=sector,
            previous_rating=previous_rating,
            model=self.tlstm_model,
            meta=self.tlstm_meta,
            device=device,
        )
        class_names = self.class_names
        tlstm_proba = _as_probability_vector(tlstm_raw.get("probabilities", {}), class_names)
        gat_proba, proxy_meta = self._graphsage_proxy(tlstm_proba)
        combined = self._combine(tlstm_proba, gat_proba)

        final_proba = combined["proba"]
        pred_idx = int(combined["pred_idx"])
        pred_class = class_names[pred_idx]
        probabilities = {
            cls: round(float(prob), 4)
            for cls, prob in zip(class_names, final_proba)
        }
        confidence = round(float(final_proba[pred_idx]), 4)

        payload = {
            "model": self.model_name,
            "rating": pred_class,
            "pred_label": pred_idx,
            "probabilities": probabilities,
            "confidence": confidence,
            **_risk_payload(pred_class, final_proba),
            "sector_resolved": tlstm_raw.get("sector_resolved"),
            "previous_rating": tlstm_raw.get("previous_rating"),
            "selected_model": combined["selected_model"],
            "dcs_case": combined["dcs_case"],
            "base_models": ["T-LSTM", "GraphSAGE"],
            "tlstm_score": combined["tlstm_score"],
            "graphsage_score": combined["graphsage_score"],
            "tlstm_prediction": _compact_prediction(class_names, tlstm_proba),
            "graphsage_prediction": _compact_prediction(class_names, gat_proba),
            **proxy_meta,
        }
        return payload


def load_dmf_tlstm_graphsage(
    artifact_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    device: str = "cpu",
) -> Tuple[DmfDcsRuntime, Dict[str, Any]]:
    """Load the DMF/DCS T-LSTM+GraphSAGE runtime package."""
    return DmfDcsRuntime.load(
        artifact_path=artifact_path,
        reference_path=reference_path,
        device=device,
    )


def predict_dmf_dcs(
    features: Dict[str, Optional[float]],
    sector: Optional[str],
    previous_rating: Optional[str],
    model: DmfDcsRuntime,
    meta: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run backend inference with the loaded DMF/DCS runtime."""
    del meta
    return model.predict(
        features=features,
        sector=sector,
        previous_rating=previous_rating,
        device=device,
    )
