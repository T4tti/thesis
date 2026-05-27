"""
Shared application state — holds trained models and loaded data.
"""
from typing import Any, Dict
import threading

MODEL_STATE: Dict[str, Any] = {
    # ── TLSTMFuzzy (PyTorch) ───────────────────────────────────────────────
    "tlstm_model":   None,   # TLSTMFuzzyClassifier instance (eval mode)
    "tlstm_meta":    None,   # Metadata dict from tlstm_meta.json

    # ── Shared ─────────────────────────────────────────────────────────────
    "data_df":       None,   # Full companies DataFrame (for reports endpoints)
    "history_csv_path": None, # Optional CSV path for persisted analyze-history rows
    "history_lock":  threading.Lock(),
    "ready":         False,  # True once TLSTM model is ready
    "startup_time":  None,   # Server start timestamp
}
