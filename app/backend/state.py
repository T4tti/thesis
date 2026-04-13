"""
Shared application state — holds trained model and loaded data.
"""
from typing import Any, Dict, Optional
import pandas as pd

MODEL_STATE: Dict[str, Any] = {
    "pipeline": None,        # Trained sklearn Pipeline (Imputer + LGBMClassifier)
    "label_encoder": None,   # sklearn LabelEncoder for IG/HY/Distressed
    "metrics": None,         # CV metrics dict from training
    "data_df": None,         # Full companies DataFrame (for reports endpoints)
    "ready": False,          # True once model is trained and data loaded
    "startup_time": None,    # Server start timestamp
}
