"""
Thêm cell lưu OOF probabilities vào cuối mỗi notebook baseline.
"""
import json
from pathlib import Path

NB_DIR = Path("e:/thesis/notebooks")

# Mapping: notebook_file -> artifact_key (dùng trong file .npy)
BASELINES = {
    'Transformer-BiLSTM.ipynb' : 'transformer_bilstm',
    'tcn-baseline.ipynb'       : 'tcn',
    'xgboost-baseline.ipynb'   : 'xgboost',
    'lstm-baseline.ipynb'      : 'lstm',
    'patchtst-baseline.ipynb'  : 'patchtst',
    'lightgbm-baseline.ipynb'  : 'lightgbm',
}

MARKER = '# [AUTO] EXPORT OOF PROBABILITIES'

for nb_file, model_key in BASELINES.items():
    path = NB_DIR / nb_file
    if not path.exists():
        print(f"SKIP (not found): {nb_file}")
        continue

    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    # Check if already added
    already_added = any(
        MARKER in ''.join(c['source'])
        for c in nb['cells']
        if c['cell_type'] == 'code'
    )
    if already_added:
        print(f"SKIP (already has export cell): {nb_file}")
        continue

    export_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Export OOF Probabilities\n\n",
            "Lưu xác suất dự đoán (val + test) để dùng trong Fuzzy Choquet Ensemble (KB7/KB8/KB9).\n"
        ]
    }

    export_code_src = f"""\
{MARKER}
# Lưu xác suất val và test để dùng trong Ensemble (KB7/KB8/KB9)
# Chạy cell này SAU KHI notebook đã huấn luyện xong.

import numpy as np
from pathlib import Path

_MODEL_KEY = '{model_key}'

# ─ Kiểm tra các biến cần thiết ─────────────────────────────────────────────
def _check_var(name):
    if name not in globals():
        raise RuntimeError(f"Biến '{{name}}' chưa được định nghĩa. Hãy chạy toàn bộ notebook trước.")

_check_var('ARTIFACT_DIR')
_check_var('y_val')
_check_var('y_test')

# ─ Lấy probabilities ────────────────────────────────────────────────────────
# Notebook deep learning (T-BiLSTM, TCN, LSTM, PatchTST): dùng biến val_proba/test_proba
# Notebook tree-based (XGBoost, LightGBM): dùng model.predict_proba()

if 'val_proba' not in globals() or 'test_proba' not in globals():
    # Tree-based: tính lại từ model và X_val / X_test
    if 'model' in globals() and hasattr(model, 'predict_proba'):
        if 'X_val' not in globals() or 'X_test' not in globals():
            raise RuntimeError("Không tìm thấy X_val / X_test. Hãy chạy lại pipeline chuẩn bị dữ liệu.")
        val_proba  = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)
    else:
        raise RuntimeError(
            "Không tìm thấy val_proba/test_proba. \\n"
            "Với deep learning: thêm `val_proba, test_proba = ...` vào cell training.\\n"
            "Với tree-based: đảm bảo `model` đã được huấn luyện."
        )

val_proba  = np.asarray(val_proba,  dtype=np.float32)
test_proba = np.asarray(test_proba, dtype=np.float32)

# ─ Validate shape ───────────────────────────────────────────────────────────
assert val_proba.shape[0] == len(y_val), (
    f"Shape mismatch: val_proba {{val_proba.shape}} vs y_val {{y_val.shape}}"
)
assert test_proba.shape[0] == len(y_test), (
    f"Shape mismatch: test_proba {{test_proba.shape}} vs y_test {{y_test.shape}}"
)

# ─ Lưu file ─────────────────────────────────────────────────────────────────
_save_val  = ARTIFACT_DIR / f'{{_MODEL_KEY}}_val_proba.npy'
_save_test = ARTIFACT_DIR / f'{{_MODEL_KEY}}_test_proba.npy'

np.save(str(_save_val),  val_proba)
np.save(str(_save_test), test_proba)

print(f"[EXPORT OOF] Model: {{_MODEL_KEY}}")
print(f"  Val  proba: {{val_proba.shape}} -> {{_save_val}}")
print(f"  Test proba: {{test_proba.shape}} -> {{_save_test}}")
print("Sẵn sàng dùng trong KB7 (FI-TTX), KB8 (FI-PLL), KB9 (FI-TTLPXL).")
"""

    export_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": export_code_src.splitlines(keepends=True)
    }

    nb['cells'].append(export_md)
    nb['cells'].append(export_code_cell)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"OK: {nb_file} (+2 cells, model_key='{model_key}')")

print("\nDone!")
