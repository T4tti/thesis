"""
Sinh 3 notebook tự chứa hoàn toàn cho KB7/KB8/KB9.
Logic Fuzzy Choquet Integral nhúng thẳng vào notebook,
chỉ dùng numpy + scipy (đã có sẵn).
"""
import json
from pathlib import Path

OUT_DIR = Path("e:/thesis/notebooks")

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def nb(cells):
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python","version":"3.10.0"}},
            "cells":cells}

# ─── SHARED CELLS ────────────────────────────────────────────────────────────

C_ENV = code("""\
import os, sys, platform, random, warnings, itertools
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

SEED = 42
random.seed(SEED); np.random.seed(SEED)

def _detect_kaggle():
    return bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','')) or (
        Path('/kaggle/input').exists() and Path('/kaggle/working').exists())

IN_KAGGLE = _detect_kaggle()

def _find_root(start):
    for p in [start, *start.parents]:
        if (p/'data').exists() and (p/'src').exists(): return p
    return start

PROJECT_ROOT = Path('/kaggle/working') if IN_KAGGLE else _find_root(Path.cwd().resolve())
ARTIFACT_DIR = PROJECT_ROOT / 'credit_rating_artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print('Python:', platform.python_version())
print('Kaggle:', IN_KAGGLE)
print('ROOT  :', PROJECT_ROOT)
print('ART   :', ARTIFACT_DIR)
""")

C_FCI_CORE = code("""\
# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY CHOQUET INTEGRAL — Self-contained implementation
# Chỉ dùng numpy + scipy.optimize (không cần pip install thêm)
#
# Lý thuyết:
#   C_mu(f) = sum_{i=1}^{n} [f_σ(i) - f_σ(i-1)] * μ(A_σ(i))
#   Trong đó σ là hoán vị sắp xếp f tăng dần,
#   A_σ(i) = {σ(i), σ(i+1), ..., σ(n-1)} là tập hợp con từ i đến cuối.
#
# Học μ bằng L-BFGS-B tối thiểu CrossEntropy + Monotonicity Penalty.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Fuzzy Measure storage ─────────────────────────────────────────────────
class FuzzyMeasure:
    \"\"\"Lưu và quản lý Fuzzy Measure μ(A) cho n nguồn.\"\"\"
    def __init__(self, n: int):
        self.n = n
        self._mu: Dict[Tuple, float] = {}
        # Khởi tạo symmetric: μ(A) = |A|/n
        for r in range(1, n+1):
            for s in itertools.combinations(range(n), r):
                self._mu[s] = len(s) / n

    def get(self, s: Tuple) -> float:
        return 0.0 if not s else self._mu.get(tuple(sorted(s)), len(s)/self.n)

    def set(self, s: Tuple, v: float):
        self._mu[tuple(sorted(s))] = float(np.clip(v, 0.0, 1.0))

    @property
    def subsets(self):
        return sorted(self._mu.keys(), key=lambda s: (len(s), s))

    def from_vec(self, v: np.ndarray):
        for k, s in enumerate(self.subsets):
            if k < len(v): self.set(s, float(v[k]))

    def to_vec(self) -> np.ndarray:
        return np.array([self.get(s) for s in self.subsets])

    def enforce_monotone(self):
        for s in sorted(self._mu, key=len):
            for e in range(self.n):
                if e not in s:
                    bigger = tuple(sorted(s + (e,)))
                    if bigger in self._mu:
                        self._mu[bigger] = max(self._mu[bigger], self.get(s))
        self._mu[tuple(range(self.n))] = 1.0


# ── 2. Choquet Integral computation ─────────────────────────────────────────
def _choquet_single(scores: np.ndarray, mu: FuzzyMeasure) -> float:
    sigma = np.argsort(scores)
    s_sorted = scores[sigma]
    result = 0.0
    for i in range(len(scores)):
        a = tuple(sorted(sigma[i:].tolist()))
        f_prev = 0.0 if i == 0 else float(s_sorted[i-1])
        result += (float(s_sorted[i]) - f_prev) * mu.get(a)
    return result

def choquet_batch(proba_3d: np.ndarray, mu: FuzzyMeasure) -> np.ndarray:
    \"\"\"proba_3d: (N, S, C) → (N, C)\"\"\"
    N, S, C = proba_3d.shape
    out = np.zeros((N, C))
    for c in range(C):
        for i in range(N):
            out[i, c] = _choquet_single(proba_3d[i, :, c], mu)
    rs = out.sum(1, keepdims=True)
    return out / np.where(rs == 0, 1.0, rs)


# ── 3. Learn fuzzy measure ───────────────────────────────────────────────────
def _loss_fn(params, proba_3d, y_true, mu, lam=0.1):
    sig = 1 / (1 + np.exp(-params))
    mu.from_vec(sig)
    mu.set(tuple(range(mu.n)), 1.0)
    fused = np.clip(choquet_batch(proba_3d, mu), 1e-9, 1.0)
    ce = -np.mean(np.log(fused[np.arange(len(y_true)), y_true.astype(int)]))
    pen = 0.0
    for s in mu.subsets:
        for e in range(mu.n):
            if e not in s:
                bigger = tuple(sorted(s + (e,)))
                diff = mu.get(s) - mu.get(bigger)
                if diff > 0: pen += diff**2
    return ce + lam * pen

def learn_measure(proba_3d, y_true, n_sources, max_iter=500, lam=0.1, verbose=True):
    mu = FuzzyMeasure(n_sources)
    v0 = mu.to_vec()
    p0 = np.log(v0 / np.clip(1 - v0, 1e-9, 1.0))
    if verbose: print(f"[FCI] Tối ưu {len(p0)} tham số, max_iter={max_iter}...")
    res = minimize(_loss_fn, p0, args=(proba_3d, y_true, mu, lam),
                   method='L-BFGS-B', options={'maxiter': max_iter,'ftol':1e-8,'gtol':1e-6})
    sig = 1 / (1 + np.exp(-res.x))
    mu.from_vec(sig)
    mu.set(tuple(range(n_sources)), 1.0)
    mu.enforce_monotone()
    if verbose: print(f"[FCI] Loss={res.fun:.6f}, converged={res.success}")
    return mu


# ── 4. Shapley values ────────────────────────────────────────────────────────
def shapley(mu: FuzzyMeasure, i: int) -> float:
    n = mu.n
    others = [j for j in range(n) if j != i]
    sv = 0.0
    for r in range(n):
        for combo in itertools.combinations(others, r):
            s = tuple(sorted(combo))
            s_i = tuple(sorted(combo + (i,)))
            coeff = (np.math.factorial(len(s)) * np.math.factorial(n - len(s) - 1)
                     / np.math.factorial(n))
            sv += coeff * (mu.get(s_i) - mu.get(s))
    return sv


# ── 5. Ensemble class ────────────────────────────────────────────────────────
class FuzzyChoquetEnsemble:
    def __init__(self, model_names: List[str], n_classes: int = 3,
                 max_iter: int = 500, lam: float = 0.1):
        self.model_names = model_names
        self.n = len(model_names)
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.lam = lam
        self.mu: Optional[FuzzyMeasure] = None

    def fit(self, val_probas: List[np.ndarray], y_val: np.ndarray):
        p3d = np.stack(val_probas, axis=1)   # (N, S, C)
        self.mu = learn_measure(p3d, y_val, self.n, self.max_iter, self.lam)
        self._shapley = {nm: shapley(self.mu, i) for i, nm in enumerate(self.model_names)}
        print("\\nShapley Values:")
        for nm, sv in sorted(self._shapley.items(), key=lambda x: -x[1]):
            print(f"  {nm:30s}: {sv:.4f}")
        return self

    def predict_proba(self, probas: List[np.ndarray]) -> np.ndarray:
        return choquet_batch(np.stack(probas, axis=1), self.mu)

    def predict(self, probas: List[np.ndarray]) -> np.ndarray:
        return np.argmax(self.predict_proba(probas), axis=1)

    def evaluate(self, probas, y_true, split='Test') -> dict:
        fp = self.predict_proba(probas)
        yp = np.argmax(fp, axis=1)
        acc = accuracy_score(y_true, yp)
        f1m = f1_score(y_true, yp, average='macro', zero_division=0)
        f1w = f1_score(y_true, yp, average='weighted', zero_division=0)
        qwk = cohen_kappa_score(y_true, yp, weights='quadratic')
        try:
            yb = label_binarize(y_true, classes=list(range(self.n_classes)))
            auc = roc_auc_score(yb, fp, average='weighted', multi_class='ovr')
        except Exception: auc = float('nan')
        print(f"\\n[{split}] Acc={acc:.4f}  MacroF1={f1m:.4f}  WtF1={f1w:.4f}  AUC={auc:.4f}  QWK={qwk:.4f}")
        return {'Split':split,'Accuracy':acc,'Macro_F1':f1m,'Weighted_F1':f1w,'AUC':auc,'QWK':qwk}

    def get_shapley(self): return self._shapley

print("[OK] Fuzzy Choquet Integral module loaded (numpy + scipy only).")
""")

C_DATA = code("""\
# ── Tải dữ liệu train / val / test ─────────────────────────────────────────
def _resolve(*candidates):
    for c in candidates:
        p = Path(c) if Path(c).is_absolute() else PROJECT_ROOT / c
        if p.exists(): return p
    raise FileNotFoundError(f"Không tìm thấy: {candidates[0]}")

TRAIN_PATH = _resolve(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/test/train.csv',
    'data/processed/test/train.csv',
    'data/processed/train_augmented_timegan.csv',
)
VAL_PATH = _resolve(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/test/val.csv',
    'data/processed/test/val.csv',
)
TEST_PATH = _resolve(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/test/test.csv',
    'data/processed/test/test.csv',
)

df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)
df_test  = pd.read_csv(TEST_PATH)
print(f"Train: {df_train.shape}  Val: {df_val.shape}  Test: {df_test.shape}")

TARGET_COL = 'rating_detail'
le = LabelEncoder().fit(df_train[TARGET_COL].astype(str))
y_train = le.transform(df_train[TARGET_COL].astype(str))
y_val   = le.transform(df_val[TARGET_COL].astype(str))
y_test  = le.transform(df_test[TARGET_COL].astype(str))
n_classes   = len(le.classes_)
class_names = list(le.classes_)

np.save(str(ARTIFACT_DIR/'y_val.npy'),  y_val)
np.save(str(ARTIFACT_DIR/'y_test.npy'), y_test)

print(f"Classes: {class_names}  n={n_classes}")
print(f"Train y: {np.bincount(y_train)}")
print(f"Val   y: {np.bincount(y_val)}")
print(f"Test  y: {np.bincount(y_test)}")
""")

C_OOF_GUIDE = md("""\
## Lưu ý: OOF Probabilities từ Baseline Notebooks

Notebook này đọc file `.npy` đã lưu từ 6 notebook baseline.
**Cách lưu**: Thêm cell sau vào **cuối** mỗi notebook baseline sau khi training xong:

```python
MODEL_KEY = 'transformer_bilstm'  # tcn | xgboost | lstm | patchtst | lightgbm
# Deep learning:  val_proba, test_proba đã có sẵn sau training loop
# Tree-based:     val_proba = model.predict_proba(X_val)
#                 test_proba = model.predict_proba(X_test)
np.save(str(ARTIFACT_DIR / f'{MODEL_KEY}_val_proba.npy'),  val_proba.astype(np.float32))
np.save(str(ARTIFACT_DIR / f'{MODEL_KEY}_test_proba.npy'), test_proba.astype(np.float32))
print("OOF saved to", ARTIFACT_DIR)
```
""")

def c_load_oof(model_keys, model_names):
    art_map = str({
        'tBiLSTM':'transformer_bilstm','tcn':'tcn','xgboost':'xgboost',
        'lstm':'lstm','patchtst':'patchtst','lgbm':'lightgbm',
    })
    return code(f"""\
MODEL_KEYS   = {model_keys!r}
MODEL_NAMES  = {model_names!r}
ART_MAP = {{'tBiLSTM':'transformer_bilstm','tcn':'tcn','xgboost':'xgboost',
           'lstm':'lstm','patchtst':'patchtst','lgbm':'lightgbm'}}

val_probas, test_probas, missing = [], [], []

for key in MODEL_KEYS:
    art = ART_MAP.get(key, key)
    vp_path = ARTIFACT_DIR / f'{{art}}_val_proba.npy'
    tp_path = ARTIFACT_DIR / f'{{art}}_test_proba.npy'
    if vp_path.exists() and tp_path.exists():
        vp = np.load(str(vp_path)).astype(np.float64)
        tp = np.load(str(tp_path)).astype(np.float64)
        print(f"  [OK] {{key}}: val={{vp.shape}}, test={{tp.shape}}")
    else:
        missing.append(key)
        vp = np.full((len(y_val),  n_classes), 1.0/n_classes)
        tp = np.full((len(y_test), n_classes), 1.0/n_classes)
        print(f"  [WARN] {{key}}: file không tìm thấy, dùng uniform proba fallback")
    val_probas.append(vp)
    test_probas.append(tp)

if missing:
    print(f"\\n[WARN] Thiếu OOF cho: {{missing}}")
    print("  --> Chạy và export OOF từ notebook baseline trước.")
else:
    print(f"\\n[OK] Đã tải đủ {{len(MODEL_KEYS)}} mô hình.")
""")

def c_fit(scenario_name, model_names):
    return code(f"""\
print("=" * 65)
print("Kịch bản: {scenario_name}")
print("Mô hình :", MODEL_NAMES)
print("=" * 65)

ens = FuzzyChoquetEnsemble(
    model_names = MODEL_NAMES,
    n_classes   = n_classes,
    max_iter    = 600,
    lam         = 0.1,
)
ens.fit(val_probas, y_val)
""")

def c_eval(scenario_name, prefix):
    return code(f"""\
val_metrics  = ens.evaluate(val_probas,  y_val,  split='Val')
test_metrics = ens.evaluate(test_probas, y_test, split='Test')

# ── Confusion Matrix ────────────────────────────────────────────────────────
y_pred = ens.predict(test_probas)
fp     = ens.predict_proba(test_probas)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

fig, ax = plt.subplots(figsize=(6,5), dpi=150)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_title('{scenario_name}\\nFuzzy Choquet Integral — Confusion Matrix (Test)', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(ARTIFACT_DIR/'{prefix}_cm.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\nClassification Report (Test):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))

# ── Save metrics ─────────────────────────────────────────────────────────────
mdf = pd.DataFrame([val_metrics, test_metrics])
mdf.to_csv(ARTIFACT_DIR/'{prefix}_metrics.csv', index=False)
np.save(str(ARTIFACT_DIR/'{prefix}_test_proba.npy'), fp.astype(np.float32))
np.save(str(ARTIFACT_DIR/'{prefix}_test_pred.npy'),  y_pred.astype(np.int32))
print("Saved metrics + predictions to", ARTIFACT_DIR)
""")

def c_viz(scenario_name, prefix):
    return code(f"""\
# ── Shapley Value Bar Chart ─────────────────────────────────────────────────
sv = ens.get_shapley()
names_sv = list(sv.keys()); vals_sv = [sv[n] for n in names_sv]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# Bar: Shapley
colors = sns.color_palette("viridis", len(names_sv))
axes[0].barh(names_sv, vals_sv, color=colors)
for i,(nm,v) in enumerate(zip(names_sv,vals_sv)):
    axes[0].text(v+0.001, i, f"{{v:.4f}}", va='center', fontsize=9)
axes[0].set_xlabel("Shapley Value")
axes[0].set_title('{scenario_name}\\nModel Importance (Shapley)', fontweight='bold')

# Heatmap: pairwise fuzzy measure
n_src = ens.n
mat = np.zeros((n_src, n_src))
for i in range(n_src):
    mat[i,i] = ens.mu.get((i,))
    for j in range(i+1, n_src):
        v = ens.mu.get((i,j))
        mat[i,j] = mat[j,i] = v

hdf = pd.DataFrame(mat, index=MODEL_NAMES, columns=MODEL_NAMES)
sns.heatmap(hdf, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1],
            vmin=0, vmax=1, linewidths=0.5)
axes[1].set_title('Fuzzy Measure μ(A) — Pairwise', fontweight='bold')

plt.tight_layout()
plt.savefig(ARTIFACT_DIR/'{prefix}_shapley_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved:", ARTIFACT_DIR/'{prefix}_shapley_heatmap.png')
""")

def c_save_compare(prefix, all_baselines):
    sources = [(k, f"{k}_metrics.csv") for k in all_baselines]
    src_code = "\n".join([f"    ('{k}', '{v}')," for k,v in sources])
    return code(f"""\
import pickle

# ── Lưu ensemble model ───────────────────────────────────────────────────────
import pickle
_state = {{'model_names': ens.model_names, 'n_classes': ens.n_classes,
           'mu_vec': ens.mu.to_vec().tolist(), 'shapley': ens.get_shapley()}}
with open(ARTIFACT_DIR/'{prefix}_ensemble.pkl','wb') as _f: pickle.dump(_state, _f)
print("Ensemble saved:", ARTIFACT_DIR/'{prefix}_ensemble.pkl')

# ── Benchmark comparison ─────────────────────────────────────────────────────
_src = [
    ('{prefix.upper()}', ARTIFACT_DIR/'{prefix}_metrics.csv'),
{src_code}
]
_frames = []
for _nm, _p in _src:
    if _p.exists():
        _df = pd.read_csv(_p); _df['Model'] = _nm; _frames.append(_df)

if _frames:
    _cmp = pd.concat(_frames, ignore_index=True)
    _cols = [c for c in ['Model','Split','Accuracy','Macro_F1','Weighted_F1','AUC','QWK'] if c in _cmp.columns]
    _cmp = _cmp[_cols].sort_values(['Split','Macro_F1'],ascending=[True,False]).reset_index(drop=True)
    try: from IPython.display import display; display(_cmp)
    except: print(_cmp.to_string(index=False))
    _cmp.to_csv(ARTIFACT_DIR/'{prefix}_benchmark.csv', index=False)
    print("Benchmark saved.")
else:
    print("[INFO] Chưa có file metric baseline để so sánh.")

print("\\n" + "="*65)
print("HOÀN THÀNH: {prefix.upper()}")
print("="*65)
mdf2 = pd.DataFrame([val_metrics, test_metrics])
print(mdf2.to_string(index=False))
""")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD NOTEBOOKS
# ─────────────────────────────────────────────────────────────────────────────

def build_nb(title, intro, sn_models, sn_names, prefix, all_baselines):
    return nb([
        md(f"# {title}\n\n{intro}"),
        md("## 1. Môi trường & Thư viện"),
        C_ENV,
        md("## 2. Fuzzy Choquet Integral — Core Implementation\n\n"
           "Toàn bộ logic FCI được nhúng trực tiếp, chỉ dùng `numpy` + `scipy.optimize`."),
        C_FCI_CORE,
        md("## 3. Tải Dữ liệu"),
        C_DATA,
        md("## 4. OOF Probabilities từ Baseline Notebooks"),
        C_OOF_GUIDE,
        c_load_oof(sn_models, sn_names),
        md("## 5. Học Fuzzy Measure (Fit Ensemble)"),
        c_fit(title, sn_names),
        md("## 6. Đánh giá Val & Test"),
        c_eval(title, prefix),
        md("## 7. Visualize Shapley Values & Fuzzy Measure"),
        c_viz(title, prefix),
        md("## 8. Lưu Model & Benchmark"),
        c_save_compare(prefix, all_baselines),
    ])

ALL6 = ['transformer_bilstm','tcn','xgboost','lstm','patchtst','lightgbm']

NB7 = build_nb(
    title="Kịch bản 7: FI-TTX — Fuzzy Choquet Integral Ensemble",
    intro=("Kết hợp **3 mô hình** bằng Fuzzy Choquet Integral:\n"
           "- **T-BiLSTM** (Transformer + Bidirectional LSTM)\n"
           "- **TCN** (Temporal Convolutional Network)\n"
           "- **XGBoost** (Gradient Boosted Trees)\n\n"
           "> Học Fuzzy Measure μ(A) từ validation, tính Shapley values."),
    sn_models=['tBiLSTM','tcn','xgboost'],
    sn_names =['Transformer-BiLSTM','TCN','XGBoost'],
    prefix="fi_ttx",
    all_baselines=ALL6,
)

NB8 = build_nb(
    title="Kịch bản 8: FI-PLL — Fuzzy Choquet Integral Ensemble",
    intro=("Kết hợp **3 mô hình** bằng Fuzzy Choquet Integral:\n"
           "- **PatchTST** (Patch Time-Series Transformer)\n"
           "- **LSTM** (Long Short-Term Memory)\n"
           "- **LightGBM** (Light Gradient Boosting Machine)\n\n"
           "> Cặp với KB7 (TTX) — khai thác kiến trúc đa dạng hơn."),
    sn_models=['patchtst','lstm','lgbm'],
    sn_names =['PatchTST','LSTM','LightGBM'],
    prefix="fi_pll",
    all_baselines=ALL6,
)

NB9 = build_nb(
    title="Kịch bản 9: FI-TTLPXL — Full Ensemble (6 Models)",
    intro=("**Full ensemble** kết hợp tất cả 6 mô hình:\n\n"
           "| Kiến trúc | Mô hình |\n|---|---|\n"
           "| Recurrent | T-BiLSTM, LSTM |\n"
           "| Convolutional | TCN |\n"
           "| Transformer | PatchTST |\n"
           "| Tree-based | XGBoost, LightGBM |\n\n"
           "> Cần KB7 + KB8 chạy xong trước để so sánh benchmark."),
    sn_models=['tBiLSTM','tcn','lstm','patchtst','xgboost','lgbm'],
    sn_names =['Transformer-BiLSTM','TCN','LSTM','PatchTST','XGBoost','LightGBM'],
    prefix="fi_ttlpxl",
    all_baselines=ALL6 + ['fi_ttx','fi_pll'],
)

for fname, notebook in [
    ('KB7_FI-TTX.ipynb',    NB7),
    ('KB8_FI-PLL.ipynb',    NB8),
    ('KB9_FI-TTLPXL.ipynb', NB9),
]:
    p = OUT_DIR / fname
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Created: {p}  ({len(notebook['cells'])} cells)")

print("\nDone!")
