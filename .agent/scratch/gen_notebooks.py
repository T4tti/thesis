"""
Script sinh 3 notebook Fuzzy Choquet Integral Ensemble:
  - KB7_FI-TTX.ipynb   (T-BiLSTM + TCN + XGBoost)
  - KB8_FI-PLL.ipynb   (PatchTST + LSTM + LightGBM)
  - KB9_FI-TTLPXL.ipynb (tất cả 6 mô hình)
"""
import json
from pathlib import Path

OUT_DIR = Path("e:/thesis/notebooks")

# ── helpers ──────────────────────────────────────────────────────────────────
def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def nb(cells: list) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}},
        "cells": cells,
    }

# ── shared cell templates ─────────────────────────────────────────────────────
CELL_ENV = code("""\
import os, sys, platform, random, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize

SEED = 42
random.seed(SEED); np.random.seed(SEED)

def detect_kaggle():
    return bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','').strip()) or (
        Path('/kaggle/input').exists() and Path('/kaggle/working').exists())

IN_KAGGLE = detect_kaggle()

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p/'data').exists() and (p/'src').exists():
            return p
    return start

CURRENT_DIR   = Path.cwd().resolve()
PROJECT_ROOT  = Path('/kaggle/working') if IN_KAGGLE else find_root(CURRENT_DIR)
ARTIFACT_DIR  = PROJECT_ROOT / 'credit_rating_artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print('Python:', platform.python_version())
print('Kaggle:', IN_KAGGLE)
print('ROOT  :', PROJECT_ROOT)
print('ARTIFACTS:', ARTIFACT_DIR)
""")

CELL_FUZZY_MOD = code("""\
# ── Tải module Fuzzy Choquet Ensemble (đã tạo trong src/models/) ──────────────
import importlib.util, sys as _sys

def _load_module(path: str, name: str = 'fuzzy_choquet_ensemble'):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Tìm file module
_candidates = [
    PROJECT_ROOT / 'src' / 'models' / 'fuzzy_choquet_ensemble.py',
    Path('src/models/fuzzy_choquet_ensemble.py'),
    Path('../src/models/fuzzy_choquet_ensemble.py'),
]
_fci_path = next((p for p in _candidates if p.exists()), None)
if _fci_path is None:
    raise FileNotFoundError(
        f"Không tìm thấy fuzzy_choquet_ensemble.py. Đã thử: {_candidates}")

fci_mod = _load_module(str(_fci_path))
FuzzyChoquetEnsemble = fci_mod.FuzzyChoquetEnsemble
load_oof_probas      = fci_mod.load_oof_probas
load_oof_labels      = fci_mod.load_oof_labels
plot_shapley_bar     = fci_mod.plot_shapley_bar
plot_fuzzy_measure_heatmap = fci_mod.plot_fuzzy_measure_heatmap
print(f"[OK] Module tải từ: {_fci_path}")
""")

CELL_LOAD_DATA = code("""\
# ── Tải dữ liệu train / val / test gốc ─────────────────────────────────────
def resolve(default, fallbacks=None):
    for p in [Path(default)] + [Path(f) for f in (fallbacks or [])]:
        full = PROJECT_ROOT / p if not p.is_absolute() else p
        if full.exists(): return full
    raise FileNotFoundError(f"Không tìm thấy: {default}")

TRAIN_PATH = resolve('/kaggle/input/datasets/tailength/corporate-credit-rating/test/train.csv',
    ['data/processed/test/train.csv','data/processed/train_augmented_timegan.csv'])
VAL_PATH   = resolve('/kaggle/input/datasets/tailength/corporate-credit-rating/test/val.csv',
    ['data/processed/test/val.csv'])
TEST_PATH  = resolve('/kaggle/input/datasets/tailength/corporate-credit-rating/test/test.csv',
    ['data/processed/test/test.csv'])

df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)
df_test  = pd.read_csv(TEST_PATH)
print(f"Train: {df_train.shape}  Val: {df_val.shape}  Test: {df_test.shape}")

TARGET_COL = 'rating_detail'
FINANCIAL_FEATURES = [
    'current_ratio','debt_equity_ratio',
    'gross_profit_margin','operating_profit_margin',
    'ebit_margin','pretax_profit_margin',
    'net_profit_margin','asset_turnover',
    'roe','roa','operating_cashflow_ps','free_cashflow_ps',
]

# Encode labels sang 0..n-1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_train[TARGET_COL].astype(str))
y_train = le.transform(df_train[TARGET_COL].astype(str))
y_val   = le.transform(df_val[TARGET_COL].astype(str))
y_test  = le.transform(df_test[TARGET_COL].astype(str))
n_classes   = len(le.classes_)
class_names = list(le.classes_)
print(f"Classes: {class_names}  (n={n_classes})")
print(f"Train y: {np.bincount(y_train)}  Val y: {np.bincount(y_val)}  Test y: {np.bincount(y_test)}")
""")

CELL_SAVE_LABELS = code("""\
# Lưu labels để dùng lại trong notebook khác
np.save(str(ARTIFACT_DIR / 'y_val.npy'),  y_val)
np.save(str(ARTIFACT_DIR / 'y_test.npy'), y_test)
print("Đã lưu y_val.npy và y_test.npy vào", ARTIFACT_DIR)
""")

def cell_load_models(model_keys, model_display_names):
    """Tạo cell tải OOF probabilities từ các mô hình baseline."""
    lines = [
        "# ── Tải OOF probabilities từ các mô hình baseline ─────────────────────\n",
        f"MODEL_KEYS         = {model_keys!r}\n",
        f"MODEL_DISPLAY_NAMES = {model_display_names!r}\n\n",
        "# Mapping từ key sang tên file artifact\n",
        "KEY_TO_ARTIFACT = {\n",
        "    'tBiLSTM' : 'transformer_bilstm',\n",
        "    'tcn'     : 'tcn',\n",
        "    'xgboost' : 'xgboost',\n",
        "    'patchtst': 'patchtst',\n",
        "    'lstm'    : 'lstm',\n",
        "    'lgbm'    : 'lightgbm',\n",
        "}\n\n",
        "val_probas_list  = []\n",
        "test_probas_list = []\n",
        "missing_models   = []\n\n",
        "for key in MODEL_KEYS:\n",
        "    art_name = KEY_TO_ARTIFACT.get(key, key)\n",
        "    vp = load_oof_probas(ARTIFACT_DIR, art_name, 'val')\n",
        "    tp = load_oof_probas(ARTIFACT_DIR, art_name, 'test')\n",
        "    if vp is None or tp is None:\n",
        "        missing_models.append(key)\n",
        "        # Fallback: tạo uniform proba nếu thiếu file\n",
        "        vp = np.full((len(y_val), n_classes), 1.0/n_classes)\n",
        "        tp = np.full((len(y_test), n_classes), 1.0/n_classes)\n",
        "        print(f'[WARN] Không tìm thấy OOF cho {key}, dùng uniform proba.')\n",
        "    val_probas_list.append(vp)\n",
        "    test_probas_list.append(tp)\n\n",
        "if missing_models:\n",
        "    print(f'[WARN] Thiếu OOF probas: {missing_models}')\n",
        "    print('  --> Hãy chạy các notebook baseline trước và lưu *_val_proba.npy và *_test_proba.npy')\n",
        "print(f'\\nĐã tải xác suất cho {len(MODEL_KEYS)} mô hình.')\n",
        "for key, vp, tp in zip(MODEL_KEYS, val_probas_list, test_probas_list):\n",
        "    print(f'  {key}: val={vp.shape}, test={tp.shape}')\n",
    ]
    return code("".join(lines))

def cell_save_oof_guide(model_keys):
    """Hướng dẫn lưu OOF từ baseline notebook."""
    names = ', '.join(model_keys)
    return md(f"""\
## Lưu ý: Cách lưu OOF Probabilities từ notebook baseline

Trước khi chạy notebook này, cần đảm bảo các notebook baseline ({names}) đã chạy và lưu file xác suất.

Thêm đoạn code sau vào **cuối** mỗi notebook baseline:

```python
# Thêm vào cuối notebook baseline (ví dụ tcn-baseline.ipynb)
MODEL_KEY = 'tcn'  # Thay bằng 'transformer_bilstm', 'xgboost', 'patchtst', 'lstm', 'lightgbm'

# Lưu val proba
val_proba = model.predict_proba(X_val)   # hoặc tương đương
np.save(str(ARTIFACT_DIR / f'{{MODEL_KEY}}_val_proba.npy'), val_proba)

# Lưu test proba  
test_proba = model.predict_proba(X_test)
np.save(str(ARTIFACT_DIR / f'{{MODEL_KEY}}_test_proba.npy'), test_proba)
print(f"Đã lưu OOF probas: {{ARTIFACT_DIR}}")
```
""")

def cell_train_ensemble(scenario_name, model_display_names):
    return code(f"""\
# ── Học Fuzzy Measure trên tập Validation ──────────────────────────────────
print("=" * 60)
print(f"Kịch bản: {scenario_name}")
print(f"Mô hình: {model_display_names!r}")
print("=" * 60)

ensemble = FuzzyChoquetEnsemble(
    model_names   = MODEL_DISPLAY_NAMES,
    n_classes     = n_classes,
    max_iter      = 500,
    lambda_mono   = 0.1,
)

ensemble.fit(
    val_probas = val_probas_list,
    y_val      = y_val,
    verbose    = True,
)

shapley = ensemble.get_shapley_values()
print("\\nShapley Values (Model Importance):")
for name, sv in sorted(shapley.items(), key=lambda x: -x[1]):
    print(f"  {{name:30s}}: {{sv:.4f}}")
""")

def cell_evaluate(scenario_name, artifact_prefix):
    return code(f"""\
# ── Đánh giá trên tập Validation và Test ───────────────────────────────────
val_metrics  = ensemble.evaluate(val_probas_list,  y_val,  split_name='Val')
test_metrics = ensemble.evaluate(test_probas_list, y_test, split_name='Test')

# Confusion Matrix - Test
fused_test_proba = ensemble.predict_proba(test_probas_list)
y_pred_test = np.argmax(fused_test_proba, axis=1)

cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(6, 5), dpi=150)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'{scenario_name}\\nFuzzy Choquet Integral - Confusion Matrix (Test)', fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
cm_path = ARTIFACT_DIR / '{artifact_prefix}_confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
print('Saved:', cm_path)

print("\\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, target_names=class_names, digits=4, zero_division=0))

# Lưu metrics
metrics_df = pd.DataFrame([val_metrics, test_metrics])
metrics_path = ARTIFACT_DIR / '{artifact_prefix}_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print('Metrics saved:', metrics_path)
""")

def cell_visualize(scenario_name, artifact_prefix):
    return code(f"""\
# ── Visualize Shapley Values và Fuzzy Measure ──────────────────────────────
plot_shapley_bar(
    ensemble.get_shapley_values(),
    ensemble_name = '{scenario_name}',
    save_path = ARTIFACT_DIR / '{artifact_prefix}_shapley.png',
)

plot_fuzzy_measure_heatmap(
    mu           = ensemble.mu,
    model_names  = MODEL_DISPLAY_NAMES,
    ensemble_name= '{scenario_name}',
    save_path    = ARTIFACT_DIR / '{artifact_prefix}_fuzzy_measure.png',
)
""")

def cell_save_ensemble(artifact_prefix):
    return code(f"""\
# ── Lưu Fuzzy Ensemble Model ────────────────────────────────────────────────
ensemble_path = ARTIFACT_DIR / '{artifact_prefix}_fuzzy_ensemble.pkl'
ensemble.save(ensemble_path)

# Lưu predictions
np.save(str(ARTIFACT_DIR / '{artifact_prefix}_test_proba.npy'), fused_test_proba)
np.save(str(ARTIFACT_DIR / '{artifact_prefix}_test_pred.npy'),  y_pred_test)
print(f"Đã lưu ensemble model và predictions.")

# Summary table
print("\\n" + "="*60)
print(f"KẾT QUẢ CUỐI CÙNG - {artifact_prefix.upper()}")
print("="*60)
summary = pd.DataFrame([val_metrics, test_metrics])
print(summary.to_string(index=False))
""")

def cell_benchmark_comparison(scenario_name, artifact_prefix, model_keys, all_baselines):
    baseline_metrics = ', '.join([f"('{k}', ARTIFACT_DIR / '{k}_metrics.csv')" for k in all_baselines])
    return code(f"""\
# ── So sánh với baseline models ─────────────────────────────────────────────
_sources = [
    ('{artifact_prefix.upper()}', ARTIFACT_DIR / '{artifact_prefix}_metrics.csv'),
    {baseline_metrics}
]

_frames = []
for _name, _path in _sources:
    if _path.exists():
        _df = pd.read_csv(_path)
        _df['Model'] = _name
        _frames.append(_df)
    else:
        print(f'[SKIP] {{_path.name}} chưa tồn tại')

if _frames:
    _cmp = pd.concat(_frames, ignore_index=True)
    _cols = [c for c in ['Model','Split','Accuracy','Macro_F1','Weighted_F1','AUC','QWK'] if c in _cmp.columns]
    _cmp = _cmp[_cols].sort_values(['Split','Macro_F1'], ascending=[True, False]).reset_index(drop=True)
    print("\\nBenchmark Comparison:")
    from IPython.display import display
    display(_cmp)
    _cmp.to_csv(ARTIFACT_DIR / '{artifact_prefix}_benchmark.csv', index=False)
""")

# ──────────────────────────────────────────────────────────────────────────────
# Scenario 7: FI-TTX  (T-BiLSTM + TCN + XGBoost)
# ──────────────────────────────────────────────────────────────────────────────
SN7_MODELS = ['tBiLSTM', 'tcn', 'xgboost']
SN7_NAMES  = ['Transformer-BiLSTM', 'TCN', 'XGBoost']
SN7_NAME   = 'Kịch bản 7: FI-TTX (Fuzzy Choquet Integral)'
SN7_PREFIX = 'fi_ttx'

cells_7 = [
    md(f"# {SN7_NAME}\n\n"
       "Ensemble **Fuzzy Choquet Integral** kết hợp:\n"
       "- **T-BiLSTM** (Transformer + Bidirectional LSTM)\n"
       "- **TCN** (Temporal Convolutional Network)\n"
       "- **XGBoost** (Gradient Boosted Trees)\n\n"
       "Học trọng số **Fuzzy Measure μ(A)** trực tiếp từ dữ liệu validation,\n"
       "tôn trọng ràng buộc đơn điệu và tính toán Shapley values.\n"),
    md("## 1. Environment Setup"),
    CELL_ENV,
    CELL_FUZZY_MOD,
    md("## 2. Tải Dữ liệu"),
    CELL_LOAD_DATA,
    CELL_SAVE_LABELS,
    md("## 3. Tải OOF Probabilities từ Baseline Models"),
    cell_save_oof_guide(SN7_NAMES),
    cell_load_models(SN7_MODELS, SN7_NAMES),
    md("## 4. Học Fuzzy Measure (Fit Ensemble)"),
    cell_train_ensemble(SN7_NAME, SN7_NAMES),
    md("## 5. Đánh giá trên Val & Test"),
    cell_evaluate(SN7_NAME, SN7_PREFIX),
    md("## 6. Visualize Shapley Values & Fuzzy Measure"),
    cell_visualize(SN7_NAME, SN7_PREFIX),
    md("## 7. Lưu Model và So sánh Baseline"),
    cell_save_ensemble(SN7_PREFIX),
    cell_benchmark_comparison(SN7_NAME, SN7_PREFIX, SN7_MODELS,
        ['transformer_bilstm', 'tcn', 'xgboost', 'lstm', 'lightgbm', 'patchtst']),
]

# ──────────────────────────────────────────────────────────────────────────────
# Scenario 8: FI-PLL  (PatchTST + LSTM + LightGBM)
# ──────────────────────────────────────────────────────────────────────────────
SN8_MODELS = ['patchtst', 'lstm', 'lgbm']
SN8_NAMES  = ['PatchTST', 'LSTM', 'LightGBM']
SN8_NAME   = 'Kịch bản 8: FI-PLL (Fuzzy Choquet Integral)'
SN8_PREFIX = 'fi_pll'

cells_8 = [
    md(f"# {SN8_NAME}\n\n"
       "Ensemble **Fuzzy Choquet Integral** kết hợp:\n"
       "- **PatchTST** (Patch-based Time-Series Transformer - SOTA)\n"
       "- **LSTM** (Long Short-Term Memory)\n"
       "- **LightGBM** (Light Gradient Boosting Machine)\n\n"
       "Bổ sung cho nhóm TTX (Kịch bản 7) - khai thác các kiến trúc khác.\n"),
    md("## 1. Environment Setup"),
    CELL_ENV,
    CELL_FUZZY_MOD,
    md("## 2. Tải Dữ liệu"),
    CELL_LOAD_DATA,
    CELL_SAVE_LABELS,
    md("## 3. Tải OOF Probabilities từ Baseline Models"),
    cell_save_oof_guide(SN8_NAMES),
    cell_load_models(SN8_MODELS, SN8_NAMES),
    md("## 4. Học Fuzzy Measure (Fit Ensemble)"),
    cell_train_ensemble(SN8_NAME, SN8_NAMES),
    md("## 5. Đánh giá trên Val & Test"),
    cell_evaluate(SN8_NAME, SN8_PREFIX),
    md("## 6. Visualize Shapley Values & Fuzzy Measure"),
    cell_visualize(SN8_NAME, SN8_PREFIX),
    md("## 7. Lưu Model và So sánh Baseline"),
    cell_save_ensemble(SN8_PREFIX),
    cell_benchmark_comparison(SN8_NAME, SN8_PREFIX, SN8_MODELS,
        ['transformer_bilstm', 'tcn', 'xgboost', 'lstm', 'lightgbm', 'patchtst']),
]

# ──────────────────────────────────────────────────────────────────────────────
# Scenario 9: FI-TTLPXL  (All 6 models)
# ──────────────────────────────────────────────────────────────────────────────
SN9_MODELS = ['tBiLSTM', 'tcn', 'lstm', 'patchtst', 'xgboost', 'lgbm']
SN9_NAMES  = ['Transformer-BiLSTM', 'TCN', 'LSTM', 'PatchTST', 'XGBoost', 'LightGBM']
SN9_NAME   = 'Kịch bản 9: FI-TTLPXL (Fuzzy Choquet Integral - Full Ensemble)'
SN9_PREFIX = 'fi_ttlpxl'

cells_9 = [
    md(f"# {SN9_NAME}\n\n"
       "**Full Ensemble** với 6 mô hình:\n"
       "| Nhóm | Mô hình |\n"
       "|------|--------|\n"
       "| Recurrent | T-BiLSTM, LSTM |\n"
       "| Convolutional | TCN |\n"
       "| Transformer | PatchTST |\n"
       "| Tree-based | XGBoost, LightGBM |\n\n"
       "> ⚠️ Cần đảm bảo `y_val.npy` và `y_test.npy` đã được lưu từ KB7 hoặc KB8.\n"),
    md("## 1. Environment Setup"),
    CELL_ENV,
    CELL_FUZZY_MOD,
    md("## 2. Tải Dữ liệu"),
    CELL_LOAD_DATA,
    CELL_SAVE_LABELS,
    md("## 3. Tải OOF Probabilities từ TẤT CẢ 6 Mô hình Baseline"),
    cell_save_oof_guide(SN9_NAMES),
    cell_load_models(SN9_MODELS, SN9_NAMES),
    md("## 4. Học Fuzzy Measure (Fit Full Ensemble - 6 Sources)"),
    cell_train_ensemble(SN9_NAME, SN9_NAMES),
    md("## 5. Đánh giá trên Val & Test"),
    cell_evaluate(SN9_NAME, SN9_PREFIX),
    md("## 6. Visualize Shapley Values & Fuzzy Measure (6 Models)"),
    cell_visualize(SN9_NAME, SN9_PREFIX),
    md("## 7. Lưu Model và So sánh Toàn diện"),
    cell_save_ensemble(SN9_PREFIX),
    cell_benchmark_comparison(SN9_NAME, SN9_PREFIX, SN9_MODELS,
        ['transformer_bilstm', 'tcn', 'xgboost', 'lstm', 'lightgbm', 'patchtst',
         'fi_ttx', 'fi_pll']),
]

# ── Ghi file ──────────────────────────────────────────────────────────────────
scenarios = [
    ('KB7_FI-TTX.ipynb',     cells_7),
    ('KB8_FI-PLL.ipynb',     cells_8),
    ('KB9_FI-TTLPXL.ipynb',  cells_9),
]

for fname, cells in scenarios:
    path = OUT_DIR / fname
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb(cells), f, ensure_ascii=False, indent=1)
    print(f"Created: {path}  ({len(cells)} cells)")

print("\nDone!")
