# AGENTS.md

> Repository-native instruction layer for AI coding agents.

---

## 📋 Project Snapshot
- **Domain:** Corporate credit rating data ingestion, normalization, and modeling experiments (Transformer-BiLSTM-Fuzzy).
- **Core Stack:** Python (3.10+), PyTorch, Pandas, Scikit-learn.
- **Language Policy:** Keep Vietnamese logs/comments unless explicitly asked to translate. Use `pathlib.Path` and UTF-8 encoding for all file I/O.

---

## 🛠️ Execution & Commands
AI agents must use the following commands for development, testing, and validation:

### 1. Environment Setup
- Install dependencies:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Installs all required Python dependencies
  pip install -r requirements.txt
  ```

### 2. ETL & Operational Pipeline
- **Run Scraper:** Scrapes fiinratings.vn via AJAX and exports CSVs/PDFs:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: External data saved in data/external/fiinratings_output/
  python src/scrapers/fiinratings_scraper.py
  ```
- **Run Merge Pipeline:** Normalizes and merges raw datasets:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Processed dataset saved to data/processed/
  python src/pipelines/merge_dataset.py
  ```
- **Run Dataset Optimization:** Applies sector-relative normalization, temporal deltas, and robust outlier capping:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Optimized features saved to data/processed/
  python src/pipelines/optimize_dataset.py
  ```
- **Run SMOTE Augmentation:** Minority-focused data augmentation:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Augmented datasets written to data/processed/
  python src/pipelines/smote_augment.py
  ```
- **Run Target Benchmarking:** Evaluates target choices and augmentation strategies:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Metrics report saved to data/reports/
  python src/pipelines/benchmark_targets.py
  ```
- **Run Augmented-Train Benchmarking:** Evaluates augmented dataset training:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Metrics report saved to data/reports/
  python src/pipelines/benchmark_augmented_trains.py
  ```
- **Generate Pipeline Diagram:** Generates pipeline architecture diagrams:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Diagram PNG saved to reports/
  python src/pipelines/generate_overview_diagram.py
  ```

### 3. Model Training & Architecture
- Core model is defined in `src/models/tlstm_fuzzy_v3.py` (Transformer-BiLSTM-Fuzzy).
- Model training utilities are defined in `src/models/training_utils.py`.

---

## 🏗️ Workspace Layout
- `src/scrapers/` - Scrapers for external financial data.
- `src/pipelines/` - ETL, data merging, optimization, and benchmarking.
- `src/models/` - TLSTM-Fuzzy model variants and training utilities.
- `data/raw/` - Raw inputs.
- `data/processed/` - Normalization and train/val/test splits.
- `data/reports/` - Benchmarks, augmentation, walk-forward validation.
- `data/external/fiinratings_output/` - External scraper outputs.
- `notebooks/` - Research and inspection notebooks.
- `artifacts/models/` or `credit_rating_artifacts/` - Model weights and plots.
- `archive/` - Archive folders.

---

## 🛑 Guardrails & Conventions

### 1. Data Invariants
- Maintain consistent rating normalization (from lowest `D` to highest `AAA`).
- Apply sector-based z-score normalization for financial ratios to ensure cross-sector comparability.
- Keep data augmentation artifacts and reports stable for reproducibility.

### 2. Modeling Invariants
- **Sequence Embeddings:** Prefer Rotary Positional Embeddings (RoPE) for temporal sequence modeling.
- **Persistence Bias Control:** Use `ContextScheduler` to force the model to learn financial features before relying on the last known rating.
- **Loss Protocol:** Use plain multiclass CE/NLL for the primary benchmark and `CE + 0.10 * CDF-EMD²` only for the ordinal ablation. Do not use CORAL/CORN, focal loss, class weights, or weighted samplers in benchmark comparisons.

### 3. Code Quality & Standards
- Always check for class imbalance and feature scaling requirements.
- Use Cross-Validation (K-Fold/Stratified) by default for small-to-medium datasets.
- Prefer vectorized operations (NumPy/Pandas) over iterative loops.
- Use explicit error handling and Type Hinting for all Python functions.

---

## 🏁 Verification & Validation Harness
AI agents must verify code changes prior to completing tasks:

### 1. Verification Scripts
- Run the core checklist to verify security, linting, and basic tests:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Success or failure status of linting, security and syntax checks
  python .agent/scripts/checklist.py .
  ```
- Run full verification suite:
  ```bash
  # How to Run: Execute from repository root
  # Expected Output: Comprehensive verification report across all checklist items
  python .agent/scripts/verify_all.py .
  ```

### 2. Evaluation Metric Monitoring
- For script changes, verify syntax and run relevant entrypoints.
- Ensure outputs are correctly written to `data/processed/` or `data/reports/`.
- For model changes, verify that **ChgAcc** (Change Accuracy) is monitored alongside **F1-weighted** to detect and combat persistence bias.
