"""
main.py — FastAPI application entry point for VN-Rate backend.

How to Run (dev):
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Expected Output on startup:
    [Trainer] Loaded 8680 rows …
    [Trainer] CV f1_weighted=0.xxxx …
    [Trainer] Training complete.
    INFO: Application startup complete.
"""
from __future__ import annotations

import datetime
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.trainer import train_model
from routers import health, predict, reports
from state import MODEL_STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent / "data"
CSV_3GRP  = DATA_DIR / "merged_credit_rating_common_3groups.csv"
CSV_CMN   = DATA_DIR / "merged_credit_rating_common.csv"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Train model and load data on startup."""
    MODEL_STATE["startup_time"] = datetime.datetime.utcnow()

    # ── Train model ───────────────────────────────────────────────────────
    log.info("=== VN-Rate Backend Startup ===")
    log.info("Training LightGBM model from %s …", CSV_3GRP.name)
    pipeline, le, metrics = train_model(CSV_3GRP)
    MODEL_STATE["pipeline"]      = pipeline
    MODEL_STATE["label_encoder"] = le
    MODEL_STATE["metrics"]       = metrics

    # ── Load reports data ─────────────────────────────────────────────────
    # Prefer the 3-groups file (cleaner labels); fall back to common file
    src_csv = CSV_3GRP if CSV_3GRP.exists() else CSV_CMN
    log.info("Loading reports data from %s …", src_csv.name)
    df = pd.read_csv(src_csv)
    # Normalise column presence
    for col in ("company_name", "ticker", "sector", "rating_detail", "rating_date",
                "rating_agency", "source"):
        if col not in df.columns:
            df[col] = ""
    df["rating_date"] = pd.to_datetime(df["rating_date"], errors="coerce")
    df["rating_date"] = df["rating_date"].dt.strftime("%Y-%m-%d").fillna("")
    MODEL_STATE["data_df"] = df

    MODEL_STATE["ready"] = True
    log.info(
        "=== Startup complete — %d records, CV F1w=%.4f ===",
        len(df),
        metrics.get("cv_f1_weighted_mean", 0),
    )

    yield   # ── app runs here ──

    log.info("Shutting down VN-Rate backend.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VN-Rate API",
    description="Corporate Credit Rating Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow Next.js dev server and production origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(reports.router)


@app.get("/")
async def root():
    return {"name": "VN-Rate API", "version": "1.0.0", "docs": "/api/docs"}
