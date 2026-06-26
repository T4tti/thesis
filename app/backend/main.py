# -*- coding: utf-8 -*-
"""
main.py — FastAPI application entry point for VN-Rating backend.

How to Run (dev):
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Expected Output on startup:
    TLSTMFuzzy model loaded.
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

from routers import explain, health, predict, reports
from state import MODEL_STATE
from database import create_db_and_tables, get_all_rating_history
from migrate_csv_to_db import migrate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent / "data"
CSV_3GRP  = DATA_DIR / "merged_credit_rating_common_3groups.csv"
CSV_CMN   = DATA_DIR / "merged_credit_rating_common.csv"
HISTORY_CSV = DATA_DIR / "prediction_history.csv"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Load models and data on startup."""
    MODEL_STATE["startup_time"] = datetime.datetime.utcnow()
    MODEL_STATE["ready"] = False
    MODEL_STATE["history_csv_path"] = HISTORY_CSV

    # ── 1. Load pre-trained DMF/DCS model package ─────────────────────────
    log.info("=== VN-Rating Backend Startup ===")
    log.info("Loading DMF/DCS T-LSTM+GraphSAGE model …")
    try:
        from model.dmf_predictor import load_dmf_tlstm_graphsage

        rating_model, rating_meta = load_dmf_tlstm_graphsage()
        MODEL_STATE["rating_model"] = rating_model
        MODEL_STATE["rating_meta"] = rating_meta
        MODEL_STATE["tlstm_model"] = rating_model
        MODEL_STATE["tlstm_meta"] = rating_meta
        log.info(
            "DMF/DCS ready — %d classes, %d sectors, GraphSAGE=%s",
            rating_meta["model_hparams"]["n_classes"],
            rating_meta["model_hparams"]["n_sectors"],
            rating_meta.get("graphsage_runtime"),
        )
    except Exception as exc:
        log.error("DMF/DCS load failed: %s", exc)

    # ── 2. Load reports data ───────────────────────────────────────────────
    # Initialize database and migrate data if needed
    try:
        log.info("Initializing database and checking for migrations …")
        create_db_and_tables()
        migrate()
    except Exception as exc:
        log.error("Database initialization failed: %s", exc)

    src_csv = CSV_3GRP if CSV_3GRP.exists() else CSV_CMN
    log.info("Loading base reports data from %s …", src_csv.name)
    df = pd.read_csv(src_csv, encoding="utf-8")
    for col in ("company_name", "ticker", "sector", "rating_detail", "rating_date",
                "rating_agency", "source"):
        if col not in df.columns:
            df[col] = ""

    # Load history from Database instead of CSV
    try:
        log.info("Loading prediction history from database …")
        history_records = get_all_rating_history()
        if history_records:
            history_df = pd.DataFrame([r.model_dump() for r in history_records])
            # Drop the 'id' column if it exists to avoid confusion with CSV data
            if "id" in history_df.columns:
                history_df = history_df.drop(columns=["id"])
            
            df = pd.concat([df, history_df], ignore_index=True, sort=False)
            log.info("Loaded %d records from history database.", len(history_records))
    except Exception as exc:
        log.warning("Could not load prediction history from database: %s", exc)

    df["rating_date"] = pd.to_datetime(df["rating_date"], errors="coerce")
    df["rating_date"] = df["rating_date"].dt.strftime("%Y-%m-%d").fillna("")
    MODEL_STATE["data_df"] = df

    MODEL_STATE["ready"] = MODEL_STATE.get("rating_model") is not None
    log.info(
        "=== Startup complete — %d records, DMF_DCS=%s, READY=%s ===",
        len(df),
        "OK" if MODEL_STATE["rating_model"] is not None else "FAIL",
        "YES" if MODEL_STATE["ready"] else "NO",
    )

    yield   # ── app runs here ──

    log.info("Shutting down VN-Rating backend.")



# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VN-Rating API",
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
app.include_router(explain.router)
app.include_router(reports.router)


@app.get("/")
async def root():
    return {"name": "VN-Rating API", "version": "1.0.0", "docs": "/api/docs"}
