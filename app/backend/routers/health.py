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


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Return service health and model training status."""
    ready = MODEL_STATE.get("ready", False)
    metrics = MODEL_STATE.get("metrics") or {}
    return {
        "status": "ok",
        "model_ready": ready,
        "uptime_seconds": (
            (datetime.datetime.utcnow() - MODEL_STATE["startup_time"]).total_seconds()
            if MODEL_STATE.get("startup_time")
            else None
        ),
        "model_metrics": {
            "cv_f1_weighted": round(metrics.get("cv_f1_weighted_mean", 0), 4),
            "cv_f1_macro":    round(metrics.get("cv_f1_macro_mean", 0), 4),
            "cv_accuracy":    round(metrics.get("cv_accuracy_mean", 0), 4),
            "n_samples":      metrics.get("n_samples", 0),
        } if ready else None,
    }
