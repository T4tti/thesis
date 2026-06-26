"""
routers/health.py — GET /api/health endpoint.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from state import MODEL_STATE

router = APIRouter(prefix="/api", tags=["health"])


def _is_model_ready() -> bool:
    return bool(MODEL_STATE.get("ready", False) and MODEL_STATE.get("rating_model") is not None)


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Return service health and model training status."""
    ready = _is_model_ready()
    rating_meta = MODEL_STATE.get("rating_meta") or MODEL_STATE.get("tlstm_meta") or {}
    hparams = rating_meta.get("model_hparams") or {}

    return {
        "status": "ok",
        "model_ready": ready,
        "uptime_seconds": (
            (datetime.datetime.utcnow() - MODEL_STATE["startup_time"]).total_seconds()
            if MODEL_STATE.get("startup_time")
            else None
        ),
        "model_metrics": {
            "model": rating_meta.get("model_name", "DMF/DCS T-LSTM+GraphSAGE"),
            "n_classes": hparams.get("n_classes"),
            "n_sectors": hparams.get("n_sectors"),
            "n_features": len(rating_meta.get("financial_features", [])),
            "base_models": rating_meta.get("base_models"),
            "graphsage_runtime": rating_meta.get("graphsage_runtime"),
            "dmf_metrics": rating_meta.get("dmf_metrics"),
        } if ready else None,
    }


@router.get("/health/ready")
async def readiness() -> JSONResponse:
    """Readiness probe for orchestrators. Returns 200 only when model is ready."""
    ready = _is_model_ready()
    status_code = 200 if ready else 503
    payload = {
        "status": "ready" if ready else "starting",
        "model_ready": ready,
    }
    return JSONResponse(status_code=status_code, content=payload)
