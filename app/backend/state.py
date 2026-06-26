"""
Shared application state — holds trained models and loaded data.
"""
from typing import Any, Dict
import threading

MODEL_STATE: Dict[str, Any] = {
    # DMF/DCS T-LSTM + GraphSAGE rating runtime
    "rating_model": None,
    "rating_meta":  None,

    # ── TLSTMFuzzy (PyTorch) ───────────────────────────────────────────────
    "tlstm_model":   None,   # Backward-compatible alias for rating_model
    "tlstm_meta":    None,   # Backward-compatible alias for rating_meta

    # ── Shared ─────────────────────────────────────────────────────────────
    "data_df":       None,   # Full companies DataFrame (for reports endpoints)
    "history_csv_path": None, # Optional CSV path for persisted analyze-history rows
    "history_lock":  threading.Lock(),
    "ready":         False,  # True once the rating model is ready
    "startup_time":  None,   # Server start timestamp
}
