# AGENTS.md

## Project Snapshot
- Domain: Corporate credit rating data ingestion, normalization, and modeling experiments (Transformer-BiLSTM-Fuzzy).
- Main language: Python.
- Focus: Reproducible scripts for ETL, data optimization, and advanced temporal modeling.

## Current Workspace Layout
- Source code:
  - `src/scrapers/`: Scrapers for financial data.
  - `src/pipelines/`: ETL, data merging, optimization, and benchmarking.
  - `src/models/`: TLSTM-Fuzzy model variants and training utilities.
- Data:
  - Raw inputs: `data/raw/`
  - Processed outputs: `data/processed/` (includes train/val/test splits and optimized versions)
  - Reports: `data/reports/` (benchmarks, augmentation, walk-forward validation)
  - External scraper output: `data/external/fiinratings_output/`
- Research notebooks: `notebooks/`
- Model artifacts and plots: `artifacts/models/` (or `credit_rating_artifacts/`)
- Archives: `archive/`

## Script Responsibilities
- `src/scrapers/fiinratings_scraper.py`:
  - Scrapes fiinratings.vn via AJAX, exports CSV, and downloads PDFs.
- `src/pipelines/merge_dataset.py`:
  - Normalizes and merges raw datasets into processed CSVs.
- `src/pipelines/optimize_dataset.py`:
  - Applies sector-relative normalization, temporal deltas, and robust outlier capping.
- `src/pipelines/benchmark_targets.py` & `src/pipelines/benchmark_augmented_trains.py`:
  - Evaluates target choices and augmentation strategies.
- `src/pipelines/smote_augment.py` & `src/pipelines/ctgan_constraints.py`:
  - Minority-focused and synthetic data augmentation.
- `src/pipelines/generate_overview_diagram.py`:
  - Generates pipeline architecture diagrams for reports.
- `src/models/tlstm_fuzzy_v3.py`:
  - Advanced Transformer-BiLSTM-Fuzzy model with RoPE, Multi-Scale Pooling, and CORN loss.
- `src/models/training_utils.py`:
  - Implements "Feature-First Curriculum" training with aggressive context masking to combat persistence bias.

## Run Commands (from repo root)
- Install deps: `pip install -r requirements.txt`
- Run scraper: `python src/scrapers/fiinratings_scraper.py`
- Run merge pipeline: `python src/pipelines/merge_dataset.py`
- Run optimization: `python src/pipelines/optimize_dataset.py`
- Run benchmark: `python src/pipelines/benchmark_targets.py`
- Run SMOTE augmentation: `python src/pipelines/smote_augment.py`
- Run augmented-train benchmark: `python src/pipelines/benchmark_augmented_trains.py`
- Generate diagram: `python src/pipelines/generate_overview_diagram.py`

## Conventions and Guardrails
- Keep Vietnamese logs/comments unless explicitly asked to translate.
- Use `pathlib.Path` and explicit encodings (UTF-8) for file I/O.
- Model Training:
  - Prefer RoPE (Rotary Positional Embeddings) for temporal sequence modeling.
  - Use `ContextScheduler` to force the model to learn financial features before relying on the last known rating (anti-persistence).
  - Use CORN (Conditional Ordinal Regression) for ordinal rating classification.
- Data:
  - Maintain consistent rating normalization (D → AAA) and sector-based z-score normalization.
  - Keep augmentation artifacts/reports stable for reproducibility.

## Validation Expectations
- For script changes, verify syntax and run relevant entrypoints.
- Ensure outputs are correctly written to `data/processed/` or `data/reports/`.
- For model changes, verify that ChgAcc (Change Accuracy) is monitored alongside F1-weighted to detect persistence bias.
