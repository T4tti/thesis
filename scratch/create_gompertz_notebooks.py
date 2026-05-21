"""
create_gompertz_notebooks.py
==============================
Tạo 3 notebook Jupyter cho Kịch bản 10, 11, 12 (Gompertz Fuzzy Ranking).

How to Run:
    python e:\\thesis\\scratch\\create_gompertz_notebooks.py

Expected Output:
    - e:\\thesis\\notebooks\\KB10_FR-TTX.ipynb
    - e:\\thesis\\notebooks\\KB11_FR-PLL.ipynb
    - e:\\thesis\\notebooks\\KB12_FR-TTLPXL.ipynb
"""
import json
from pathlib import Path
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Notebook cell helpers
# ─────────────────────────────────────────────────────────────────────────────

def md_cell(source: str) -> Dict[str, Any]:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")][:-1] + [source.split("\n")[-1]]
    }

def code_cell(source: str) -> Dict[str, Any]:
    """Create a code cell."""
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None
    }

def make_notebook(cells: List[Dict]) -> Dict:
    """Wrap cells into a notebook structure."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

# ─────────────────────────────────────────────────────────────────────────────
# Shared code sections (identical across KB10, KB11, KB12)
# ─────────────────────────────────────────────────────────────────────────────

CELL_ENV = '''import os, sys, platform, random, warnings, itertools, math
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
    precision_score, recall_score, roc_curve, auc,
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
print('ART   :', ARTIFACT_DIR)'''


CELL_GFR_CORE = r'''# ═══════════════════════════════════════════════════════════════════════════════
# GOMPERTZ FUZZY RANKING — Self-contained implementation
# Chỉ dùng numpy + scipy.optimize (không cần pip install thêm)
#
# Lý thuyết:
#   Hàm thành viên Gompertz (phi đối xứng sigmoid):
#       μ(x) = a · exp(-b · exp(-c · x))
#
#   Trong đó:
#       a = 1 (tiệm cận trên, cố định)
#       b > 0: kiểm soát vị trí điểm uốn
#       c > 0: kiểm soát tốc độ tăng trưởng
#
#   Quy trình tổng hợp Gompertz Fuzzy Ranking:
#       1. Với mỗi sample, sắp xếp xác suất của các mô hình theo thứ tự tăng dần.
#       2. Áp dụng hàm Gompertz lên thứ hạng đã chuẩn hoá: rank_i = i / (n-1)
#          để tạo trọng số fuzzy cho mỗi vị trí xếp hạng.
#       3. Nhân trọng số fuzzy với xác suất tương ứng, chuẩn hoá tổng.
#
#   Khác biệt với Choquet Integral:
#       - Choquet sử dụng 2^n - 2 tham số (capacity function trên tập con).
#       - Gompertz chỉ dùng 2 tham số toàn cục (b, c) → ít overfitting hơn.
#       - Gompertz tự nhiên mô hình hoá "bất đối xứng" trong xếp hạng tín nhiệm.
#
# Tối ưu b, c bằng L-BFGS-B tối thiểu CrossEntropy trên validation set.
# ═══════════════════════════════════════════════════════════════════════════════


# ── 1. Gompertz membership function ─────────────────────────────────────────
def gompertz(x: np.ndarray, b: float, c: float) -> np.ndarray:
    """
    Hàm Gompertz: μ(x) = exp(-b * exp(-c * x))
    
    Parameters
    ----------
    x : np.ndarray
        Giá trị đầu vào (thứ hạng chuẩn hoá trong [0, 1]).
    b : float > 0
        Kiểm soát vị trí điểm uốn (inflection point).
    c : float > 0
        Kiểm soát tốc độ tăng trưởng (growth rate).
    
    Returns
    -------
    np.ndarray : Giá trị fuzzy membership trong (0, 1].
    """
    return np.exp(-b * np.exp(-c * x))


# ── 2. Gompertz Fuzzy Ranking aggregation ────────────────────────────────────
def gompertz_rank_fuse_single(
    scores: np.ndarray,
    b: float,
    c: float,
) -> float:
    """
    Tổng hợp điểm số từ nhiều mô hình bằng Gompertz Fuzzy Ranking cho 1 class.
    
    Parameters
    ----------
    scores : np.ndarray, shape (n_sources,)
        Xác suất dự đoán của từng mô hình cho 1 class, 1 sample.
    b, c : float
        Tham số Gompertz đã học.
    
    Returns
    -------
    float : Điểm số tổng hợp.
    """
    n = len(scores)
    # Sắp xếp tăng dần
    sigma = np.argsort(scores)
    scores_sorted = scores[sigma]
    
    # Tạo thứ hạng chuẩn hoá: [0, 1/(n-1), 2/(n-1), ..., 1]
    if n > 1:
        ranks = np.arange(n, dtype=np.float64) / (n - 1)
    else:
        ranks = np.array([1.0])
    
    # Trọng số fuzzy từ hàm Gompertz
    weights = gompertz(ranks, b, c)
    
    # Tổng hợp có trọng số
    result = np.dot(weights, scores_sorted) / (np.sum(weights) + 1e-12)
    return result


def gompertz_rank_fuse_batch(
    proba_3d: np.ndarray,
    b: float,
    c: float,
) -> np.ndarray:
    """
    Tổng hợp batch xác suất đa lớp bằng Gompertz Fuzzy Ranking.
    
    Parameters
    ----------
    proba_3d : np.ndarray, shape (n_samples, n_sources, n_classes)
    b, c : float
        Tham số Gompertz.
    
    Returns
    -------
    np.ndarray, shape (n_samples, n_classes)
        Xác suất tổng hợp (đã chuẩn hoá tổng = 1).
    """
    N, S, C = proba_3d.shape
    out = np.zeros((N, C), dtype=np.float64)
    
    for cls in range(C):
        for i in range(N):
            out[i, cls] = gompertz_rank_fuse_single(proba_3d[i, :, cls], b, c)
    
    # Chuẩn hoá tổng xác suất = 1
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return out / row_sums


# ── 3. Learn Gompertz parameters from validation ────────────────────────────
def _gfr_loss(params, proba_3d, y_true):
    """
    Hàm loss cho tối ưu tham số Gompertz.
    Loss = CrossEntropy(GompertzFuse(proba), y_true)
    """
    # Params ở log-space để đảm bảo b, c > 0
    b = np.exp(params[0])
    c = np.exp(params[1])
    
    fused = np.clip(gompertz_rank_fuse_batch(proba_3d, b, c), 1e-9, 1.0)
    n_samples = len(y_true)
    ce = -np.mean(np.log(fused[np.arange(n_samples), y_true.astype(int)]))
    return ce


def learn_gompertz_params(
    proba_3d: np.ndarray,
    y_true: np.ndarray,
    max_iter: int = 500,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Học tham số (b, c) tối ưu từ tập validation.
    
    Parameters
    ----------
    proba_3d : np.ndarray, shape (n_val, n_sources, n_classes)
    y_true : np.ndarray, shape (n_val,)
    max_iter : int
    verbose : bool
    
    Returns
    -------
    Tuple[float, float] : (b_opt, c_opt)
    """
    # Khởi tạo: b=1.0, c=3.0 (log-space)
    init_params = np.array([np.log(1.0), np.log(3.0)])
    
    if verbose:
        print(f"[GFR] Tối ưu 2 tham số Gompertz (b, c), max_iter={max_iter}...")
    
    result = minimize(
        _gfr_loss,
        init_params,
        args=(proba_3d, y_true),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8},
    )
    
    b_opt = np.exp(result.x[0])
    c_opt = np.exp(result.x[1])
    
    if verbose:
        print(f"[GFR] Loss cuối = {result.fun:.6f}, converged = {result.success}")
        print(f"[GFR] b* = {b_opt:.6f}, c* = {c_opt:.6f}")
        # Tính điểm uốn (inflection point) của hàm Gompertz
        inflection = np.log(b_opt) / c_opt if c_opt > 0 else float('nan')
        print(f"[GFR] Điểm uốn (inflection point) = {inflection:.4f}")
    
    return b_opt, c_opt


# ── 4. Model Importance via Permutation ──────────────────────────────────────
def model_importance_permutation(
    proba_3d: np.ndarray,
    y_true: np.ndarray,
    b: float, c: float,
    model_names: List[str],
    n_repeats: int = 10,
) -> Dict[str, float]:
    """
    Đo tầm quan trọng mô hình bằng Permutation Importance.
    
    Ý tưởng: Với mỗi mô hình, xáo trộn xác suất của nó rồi đo mức suy giảm
    accuracy. Mô hình nào gây suy giảm nhiều → quan trọng hơn.
    """
    # Baseline accuracy
    fused_base = gompertz_rank_fuse_batch(proba_3d, b, c)
    y_pred_base = np.argmax(fused_base, axis=1)
    acc_base = accuracy_score(y_true, y_pred_base)
    
    importance = {}
    rng = np.random.default_rng(42)
    
    for src_idx, name in enumerate(model_names):
        drops = []
        for _ in range(n_repeats):
            p3d_perm = proba_3d.copy()
            perm_idx = rng.permutation(len(y_true))
            p3d_perm[:, src_idx, :] = p3d_perm[perm_idx, src_idx, :]
            fused_perm = gompertz_rank_fuse_batch(p3d_perm, b, c)
            y_pred_perm = np.argmax(fused_perm, axis=1)
            acc_perm = accuracy_score(y_true, y_pred_perm)
            drops.append(acc_base - acc_perm)
        importance[name] = float(np.mean(drops))
    
    return importance


# ── 5. Ensemble class ────────────────────────────────────────────────────────
class GompertzFuzzyRankingEnsemble:
    """
    Ensemble tổng hợp xác suất bằng Gompertz Fuzzy Ranking.
    
    Workflow:
    1. Nhận OOF predictions từ các mô hình baseline.
    2. Học tham số Gompertz (b, c) trên tập validation.
    3. Áp dụng lên tập test.
    4. Đo Model Importance bằng Permutation.
    """
    
    def __init__(self, model_names: List[str], n_classes: int = 3,
                 max_iter: int = 500):
        self.model_names = model_names
        self.n = len(model_names)
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.b: Optional[float] = None
        self.c: Optional[float] = None
        self._importance: Optional[Dict[str, float]] = None
    
    def fit(self, val_probas: List[np.ndarray], y_val: np.ndarray):
        """Học tham số Gompertz từ xác suất validation."""
        p3d = np.stack(val_probas, axis=1)   # (N, S, C)
        self.b, self.c = learn_gompertz_params(p3d, y_val, self.max_iter)
        
        # Tính Model Importance
        self._importance = model_importance_permutation(
            p3d, y_val, self.b, self.c, self.model_names)
        
        print("\nModel Importance (Permutation-based):")
        for nm, imp in sorted(self._importance.items(), key=lambda x: -x[1]):
            print(f"  {nm:30s}: {imp:+.4f}")
        return self
    
    def predict_proba(self, probas: List[np.ndarray]) -> np.ndarray:
        """Tổng hợp xác suất bằng Gompertz Fuzzy Ranking."""
        if self.b is None:
            raise RuntimeError("Chưa gọi .fit(). Hãy huấn luyện trước.")
        return gompertz_rank_fuse_batch(np.stack(probas, axis=1), self.b, self.c)
    
    def predict(self, probas: List[np.ndarray]) -> np.ndarray:
        return np.argmax(self.predict_proba(probas), axis=1)
    
    def evaluate(self, probas, y_true, split='Test') -> dict:
        fp = self.predict_proba(probas)
        yp = np.argmax(fp, axis=1)
        acc = accuracy_score(y_true, yp)
        f1m = f1_score(y_true, yp, average='macro', zero_division=0)
        f1w = f1_score(y_true, yp, average='weighted', zero_division=0)
        qwk = cohen_kappa_score(y_true, yp, weights='quadratic')
        prec_w = precision_score(y_true, yp, average='weighted', zero_division=0)
        rec_w  = recall_score(y_true, yp, average='weighted', zero_division=0)
        try:
            yb = label_binarize(y_true, classes=list(range(self.n_classes)))
            auc_val = roc_auc_score(yb, fp, average='weighted', multi_class='ovr')
        except Exception:
            auc_val = float('nan')
        print(f"\n[{split}] Acc={acc:.4f}  MacroF1={f1m:.4f}  WtF1={f1w:.4f}  AUC={auc_val:.4f}  QWK={qwk:.4f}")
        return {'Split': split, 'Accuracy': acc, 'Precision_Weighted': prec_w,
                'Recall_Weighted': rec_w, 'Macro_F1': f1m, 'Weighted_F1': f1w,
                'AUC': auc_val, 'QWK': qwk}
    
    def get_importance(self): return self._importance

print("[OK] Gompertz Fuzzy Ranking module loaded (numpy + scipy only).")'''


CELL_DATA_LOAD = '''# ── Tải dữ liệu train / val / test ─────────────────────────────────────────
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
print(f"Test  y: {np.bincount(y_test)}")'''


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-specific configuration
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "kb": 10,
        "name": "FR-TTX",
        "title": "Kịch bản 10: FR-TTX — Gompertz Fuzzy Ranking Ensemble",
        "desc": (
            "Kết hợp **3 mô hình** bằng Gompertz Fuzzy Ranking:\n"
            "- **T-BiLSTM** (Transformer + Bidirectional LSTM)\n"
            "- **TCN** (Temporal Convolutional Network)\n"
            "- **XGBoost** (Gradient Boosted Trees)\n\n"
            "> Học tham số Gompertz (b, c) từ validation. Đo Model Importance bằng Permutation."
        ),
        "model_keys": "['tBiLSTM', 'tcn', 'xgboost']",
        "model_names": "['Transformer-BiLSTM', 'TCN', 'XGBoost']",
        "prefix": "fr_ttx",
        "benchmark_extra": "",
    },
    {
        "kb": 11,
        "name": "FR-PLL",
        "title": "Kịch bản 11: FR-PLL — Gompertz Fuzzy Ranking Ensemble",
        "desc": (
            "Kết hợp **3 mô hình** bằng Gompertz Fuzzy Ranking:\n"
            "- **PatchTST** (Patch Time-Series Transformer)\n"
            "- **LSTM** (Long Short-Term Memory)\n"
            "- **LightGBM** (Light Gradient Boosting Machine)\n\n"
            "> Cặp với KB10 (TTX) — khai thác kiến trúc đa dạng hơn."
        ),
        "model_keys": "['patchtst', 'lstm', 'lgbm']",
        "model_names": "['PatchTST', 'LSTM', 'LightGBM']",
        "prefix": "fr_pll",
        "benchmark_extra": "",
    },
    {
        "kb": 12,
        "name": "FR-TTLPXL",
        "title": "Kịch bản 12: FR-TTLPXL — Gompertz Fuzzy Ranking Ensemble",
        "desc": (
            "Kết hợp **toàn bộ 6 mô hình** bằng Gompertz Fuzzy Ranking:\n"
            "- T-BiLSTM + TCN + XGBoost + PatchTST + LSTM + LightGBM\n\n"
            "> Kịch bản tổng hợp toàn diện — đối chứng với KB9 (FI-TTLPXL Choquet)."
        ),
        "model_keys": "['tBiLSTM', 'tcn', 'lstm', 'patchtst', 'xgboost', 'lgbm']",
        "model_names": "['Transformer-BiLSTM', 'TCN', 'LSTM', 'PatchTST', 'XGBoost', 'LightGBM']",
        "prefix": "fr_ttlpxl",
        "benchmark_extra": (
            "    ('FR_TTX',  ARTIFACT_DIR/'fr_ttx_metrics.csv'),\n"
            "    ('FR_PLL',  ARTIFACT_DIR/'fr_pll_metrics.csv'),\n"
        ),
    },
]


def build_oof_cell(scenario: dict) -> str:
    return f"""MODEL_KEYS   = {scenario['model_keys']}
MODEL_NAMES  = {scenario['model_names']}
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
    print(f"\\n[OK] Đã tải đủ {{len(MODEL_KEYS)}} mô hình.")"""


def build_fit_cell(scenario: dict) -> str:
    return f"""print("=" * 65)
print("Kịch bản: {scenario['title']}")
print("Mô hình :", MODEL_NAMES)
print("=" * 65)

ens = GompertzFuzzyRankingEnsemble(
    model_names = MODEL_NAMES,
    n_classes   = n_classes,
    max_iter    = 600,
)
ens.fit(val_probas, y_val)"""


def build_eval_cell(scenario: dict) -> str:
    prefix = scenario['prefix']
    title = scenario['title']
    return f"""import time

# ── Inference Time Measurement ───────────────────────────────────────────────
start_time = time.time()
y_pred = ens.predict(test_probas)
fp     = ens.predict_proba(test_probas)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Test Inference/Evaluation Time: {{elapsed_time:.6f}} seconds")

val_metrics  = ens.evaluate(val_probas,  y_val,  split='Val')
test_metrics = ens.evaluate(test_probas, y_test, split='Test')

# ── Confusion Matrix ────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

fig, ax = plt.subplots(figsize=(6,5), dpi=150)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_title('{title}\\nGompertz Fuzzy Ranking — Confusion Matrix (Test)', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(ARTIFACT_DIR/'{prefix}_cm.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\nClassification Report (Test):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))

# ── AUC-ROC Plot ────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

for i in range(n_classes):
    fpr_class, tpr_class, _ = roc_curve(y_test_bin[:, i], fp[:, i])
    roc_auc_val = auc(fpr_class, tpr_class)
    plt.plot(fpr_class, tpr_class, lw=2, label=f'ROC curve of class {{class_names[i]}} (area = {{roc_auc_val:0.2f}})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('{title} - Receiver Operating Characteristic (ROC) - Multiclass')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
roc_plot_path = ARTIFACT_DIR / '{prefix}_roc.png'
plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print('Saved ROC curves to:', roc_plot_path)

# ── Save metrics ─────────────────────────────────────────────────────────────
mdf = pd.DataFrame([val_metrics, test_metrics])
mdf.to_csv(ARTIFACT_DIR/'{prefix}_metrics.csv', index=False)
np.save(str(ARTIFACT_DIR/'{prefix}_test_proba.npy'), fp.astype(np.float32))
np.save(str(ARTIFACT_DIR/'{prefix}_test_pred.npy'),  y_pred.astype(np.int32))
print("Saved metrics + predictions to", ARTIFACT_DIR)"""


def build_viz_cell(scenario: dict) -> str:
    prefix = scenario['prefix']
    title = scenario['title']
    return f"""# ── Model Importance Bar Chart ───────────────────────────────────────────────
imp = ens.get_importance()
names_imp = list(imp.keys()); vals_imp = [imp[n] for n in names_imp]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# Bar: Importance
colors = sns.color_palette("viridis", len(names_imp))
axes[0].barh(names_imp, vals_imp, color=colors)
for i,(nm,v) in enumerate(zip(names_imp, vals_imp)):
    axes[0].text(max(v, 0)+0.001, i, f"{{v:+.4f}}", va='center', fontsize=9)
axes[0].set_xlabel("Accuracy Drop (Permutation Importance)")
axes[0].set_title('{title}\\nModel Importance (Permutation)', fontweight='bold')
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Gompertz Curve Visualization
x_range = np.linspace(0, 1, 200)
y_gompertz = gompertz(x_range, ens.b, ens.c)

axes[1].plot(x_range, y_gompertz, 'b-', lw=2.5, label=f'Gompertz(b={{ens.b:.3f}}, c={{ens.c:.3f}})')
# Mark inflection point
x_inflect = np.log(ens.b) / ens.c if ens.c > 0 else 0.5
if 0 <= x_inflect <= 1:
    y_inflect = gompertz(np.array([x_inflect]), ens.b, ens.c)[0]
    axes[1].plot(x_inflect, y_inflect, 'ro', markersize=10, label=f'Inflection ({{x_inflect:.3f}}, {{y_inflect:.3f}})')
# Sigmoid for comparison
y_sigmoid = 1 / (1 + np.exp(-10*(x_range - 0.5)))
axes[1].plot(x_range, y_sigmoid, 'g--', lw=1.5, alpha=0.6, label='Sigmoid (symmetric)')
axes[1].set_xlabel("Rank (normalized)")
axes[1].set_ylabel("Fuzzy Weight μ(x)")
axes[1].set_title("Learned Gompertz vs Sigmoid\\n(Asymmetric Fuzzy Membership)", fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(ARTIFACT_DIR/'{prefix}_importance_gompertz.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved:", ARTIFACT_DIR/'{prefix}_importance_gompertz.png')"""


def build_lime_cell(scenario: dict) -> str:
    prefix = scenario['prefix']
    return f"""# ==============================================================================
# xAI LIME Explainability — Gompertz Fuzzy Ranking Ensemble
# ==============================================================================
LIME_ENABLED = True

if LIME_ENABLED:
    import subprocess

    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        print("Installing lime package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lime", "-q"])
        import lime
        import lime.lime_tabular

    if 'ens' not in globals() or 'val_probas' not in globals() or 'test_probas' not in globals():
        raise RuntimeError('Không tìm thấy ens/val_probas/test_probas. Hãy chạy cell huấn luyện trước.')

    # Feature names: mỗi feature = xác suất của 1 class từ 1 model
    lime_class_names = class_names if 'class_names' in globals() else [str(i) for i in range(n_classes)]
    feature_names_lime = [
        f"{{model_name}}__{{cls}}"
        for model_name in ens.model_names
        for cls in lime_class_names
    ]

    # Stack & flatten xác suất
    val_probas_stacked = np.stack(val_probas, axis=1)   # (N_val, S, C)
    val_probas_flat = val_probas_stacked.reshape(len(y_val), -1)  # (N_val, S*C)

    test_probas_stacked = np.stack(test_probas, axis=1)
    test_probas_flat = test_probas_stacked.reshape(len(y_test), -1)

    # Custom predict_fn: flat → 3D → Gompertz fusion
    def ensemble_predict_fn_lime(x_flat_batch):
        x_flat_batch = np.asarray(x_flat_batch, dtype=np.float64)
        if x_flat_batch.ndim == 1:
            x_flat_batch = x_flat_batch.reshape(1, -1)
        N_batch = x_flat_batch.shape[0]
        p3d = x_flat_batch.reshape(N_batch, ens.n, ens.n_classes)
        return gompertz_rank_fuse_batch(p3d, ens.b, ens.c)

    # Khởi tạo LIME Explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=val_probas_flat,
        feature_names=feature_names_lime,
        class_names=lime_class_names,
        mode='classification',
        random_state=SEED
    )

    # Giải thích 3 mẫu ngẫu nhiên
    rng_lime = np.random.default_rng(SEED)
    selected_indices = rng_lime.choice(len(y_test), size=min(3, len(y_test)), replace=False)

    print(f"LIME explainability enabled. Explaining {{len(selected_indices)}} random ensemble predictions:")
    for idx in selected_indices:
        print(f"\\n--- Explaining ensemble prediction at test index {{idx}} ---")
        exp = explainer_lime.explain_instance(
            data_row=test_probas_flat[idx],
            predict_fn=ensemble_predict_fn_lime,
            num_features=10
        )
        try:
            exp.show_in_notebook(show_table=True)
        except Exception:
            pass
        
        html_path = ARTIFACT_DIR / f"{prefix}_lime_explanation_test_idx_{{idx}}.html"
        exp.save_to_file(str(html_path))
        print(f"Saved LIME explanation HTML to: {{html_path}}")
        print("LIME Local Explanation Details:")
        for feature, weight in exp.as_list():
            print(f"  {{feature:<30}} : {{weight:+.4f}}")"""


def build_shap_cell(scenario: dict) -> str:
    prefix = scenario['prefix']
    return f"""# ==============================================================================
# xAI SHAP Explainability — Gompertz Fuzzy Ranking Ensemble
# ==============================================================================
SHAP_ENABLED = True

if SHAP_ENABLED:
    import subprocess

    try:
        import shap
    except ImportError:
        print("Installing shap package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
        import shap

    if 'ens' not in globals() or 'val_probas' not in globals() or 'test_probas' not in globals():
        raise RuntimeError('Không tìm thấy ens/val_probas/test_probas. Hãy chạy cell huấn luyện trước.')

    # Feature names
    shap_class_names = class_names if 'class_names' in globals() else [str(i) for i in range(n_classes)]
    feature_names_shap = [
        f"{{model_name}}__{{cls}}"
        for model_name in ens.model_names
        for cls in shap_class_names
    ]

    # Stack & flatten
    val_probas_stacked = np.stack(val_probas, axis=1)
    val_probas_flat = val_probas_stacked.reshape(len(y_val), -1)
    test_probas_stacked = np.stack(test_probas, axis=1)
    test_probas_flat = test_probas_stacked.reshape(len(y_test), -1)

    # Custom predict_fn: flat → 3D → Gompertz fusion
    def ensemble_predict_fn_shap(x_flat_batch):
        x_flat_batch = np.asarray(x_flat_batch, dtype=np.float64)
        if x_flat_batch.ndim == 1:
            x_flat_batch = x_flat_batch.reshape(1, -1)
        N_batch = x_flat_batch.shape[0]
        p3d = x_flat_batch.reshape(N_batch, ens.n, ens.n_classes)
        return gompertz_rank_fuse_batch(p3d, ens.b, ens.c)

    # KernelExplainer with background data
    background_size = min(40, len(val_probas_flat))
    background_data = val_probas_flat[:background_size]

    print("Initializing SHAP KernelExplainer on Gompertz ensemble model...")
    explainer_shap = shap.KernelExplainer(ensemble_predict_fn_shap, background_data)

    # Explain random test samples
    rng_shap = np.random.default_rng(SEED)
    shap_indices = rng_shap.choice(len(test_probas_flat), size=min(10, len(test_probas_flat)), replace=False)
    
    print(f"SHAP explainer running on {{len(shap_indices)}} random test samples...")
    shap_values = explainer_shap.shap_values(test_probas_flat[shap_indices])

    print("SHAP explanation computed. Generating Summary Plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        test_probas_flat[shap_indices],
        feature_names=feature_names_shap,
        class_names=shap_class_names,
        show=False
    )
    plt.title(f"Gompertz Fuzzy Ranking Ensemble — SHAP Summary Plot", fontweight='bold')
    plt.tight_layout()
    shap_plot_path = ARTIFACT_DIR / '{prefix}_shap_summary.png'
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print('Saved SHAP summary plot to:', shap_plot_path)

    # ── SHAP Waterfall for first sample ──────────────────────────────────────
    try:
        print("\\nGenerating SHAP Waterfall Plot (sample 0)...")
        # Use class 0 for waterfall
        if isinstance(shap_values, list):
            sv_class0 = shap_values[0]
        else:
            sv_class0 = shap_values
        
        shap_exp = shap.Explanation(
            values=sv_class0[0],
            base_values=explainer_shap.expected_value[0] if isinstance(explainer_shap.expected_value, (list, np.ndarray)) else explainer_shap.expected_value,
            data=test_probas_flat[shap_indices[0]],
            feature_names=feature_names_shap,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, show=False)
        plt.title("SHAP Waterfall — Sample 0", fontweight='bold')
        plt.tight_layout()
        waterfall_path = ARTIFACT_DIR / '{prefix}_shap_waterfall.png'
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.show()
        print('Saved SHAP waterfall to:', waterfall_path)
    except Exception as e:
        print(f"[WARN] Waterfall plot skipped: {{e}}")"""


def build_save_benchmark_cell(scenario: dict) -> str:
    prefix = scenario['prefix']
    name_upper = scenario['name'].replace('-', '_').upper()
    extra = scenario['benchmark_extra']
    return f"""import pickle

# ── Lưu ensemble model ───────────────────────────────────────────────────────
_state = {{'model_names': ens.model_names, 'n_classes': ens.n_classes,
           'b': ens.b, 'c': ens.c, 'importance': ens.get_importance()}}
with open(ARTIFACT_DIR/'{prefix}_ensemble.pkl','wb') as _f: pickle.dump(_state, _f)
print("Ensemble saved:", ARTIFACT_DIR/'{prefix}_ensemble.pkl')

# ── Benchmark comparison ─────────────────────────────────────────────────────
_src = [
    ('{name_upper}', ARTIFACT_DIR/'{prefix}_metrics.csv'),
{extra}    ('transformer_bilstm', ARTIFACT_DIR/'transformer_bilstm_metrics.csv'),
    ('tcn',                ARTIFACT_DIR/'tcn_metrics.csv'),
    ('xgboost',            ARTIFACT_DIR/'xgboost_metrics.csv'),
    ('lstm',               ARTIFACT_DIR/'lstm_metrics.csv'),
    ('patchtst',           ARTIFACT_DIR/'patchtst_metrics.csv'),
    ('lightgbm',           ARTIFACT_DIR/'lightgbm_metrics.csv'),
]
_frames = []
for _nm, _p in _src:
    _p = Path(_p)
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
print("HOÀN THÀNH: {name_upper}")
print("="*65)
mdf2 = pd.DataFrame([val_metrics, test_metrics])
print(mdf2.to_string(index=False))"""


# ─────────────────────────────────────────────────────────────────────────────
# Main: Generate notebooks
# ─────────────────────────────────────────────────────────────────────────────

def build_notebook(scenario: dict) -> Dict:
    """Build a complete notebook for a Gompertz Fuzzy Ranking scenario."""
    cells = [
        md_cell(f"# {scenario['title']}\n\n{scenario['desc']}"),
        
        md_cell("## 1. Môi trường & Thư viện"),
        code_cell(CELL_ENV),
        
        md_cell(
            "## 2. Gompertz Fuzzy Ranking — Core Implementation\n\n"
            "Toàn bộ logic GFR được nhúng trực tiếp, chỉ dùng `numpy` + `scipy.optimize`.\n\n"
            "**Khác biệt với Fuzzy Choquet Integral (KB7-9):**\n"
            "- Choquet dùng $2^n - 2$ tham số (capacity function) → phức tạp, dễ overfitting.\n"
            "- Gompertz dùng **chỉ 2 tham số** $(b, c)$ → compact, robust.\n"
            "- Hàm Gompertz **phi đối xứng** (asymmetric sigmoid) phù hợp với ranh giới xếp hạng tín nhiệm.\n\n"
            "$$\\mu(x) = \\exp(-b \\cdot \\exp(-c \\cdot x))$$"
        ),
        code_cell(CELL_GFR_CORE),
        
        md_cell("## 3. Tải Dữ liệu"),
        code_cell(CELL_DATA_LOAD),
        
        md_cell(
            "## 4. OOF Probabilities từ Baseline Notebooks\n\n"
            "## Lưu ý: OOF Probabilities từ Baseline Notebooks\n\n"
            "Notebook này đọc file `.npy` đã lưu từ 6 notebook baseline.\n"
            "**Cách lưu**: Thêm cell sau vào **cuối** mỗi notebook baseline sau khi training xong:\n\n"
            "```python\n"
            "MODEL_KEY = 'transformer_bilstm'  # tcn | xgboost | lstm | patchtst | lightgbm\n"
            "# Deep learning:  val_proba, test_proba đã có sẵn sau training loop\n"
            "# Tree-based:     val_proba = model.predict_proba(X_val)\n"
            "#                 test_proba = model.predict_proba(X_test)\n"
            "np.save(str(ARTIFACT_DIR / f'{MODEL_KEY}_val_proba.npy'),  val_proba.astype(np.float32))\n"
            "np.save(str(ARTIFACT_DIR / f'{MODEL_KEY}_test_proba.npy'), test_proba.astype(np.float32))\n"
            "print('OOF saved to', ARTIFACT_DIR)\n"
            "```"
        ),
        code_cell(build_oof_cell(scenario)),
        
        md_cell("## 5. Học Tham số Gompertz (Fit Ensemble)"),
        code_cell(build_fit_cell(scenario)),
        
        md_cell("## 6. Đánh giá Val & Test"),
        code_cell(build_eval_cell(scenario)),
        
        md_cell("## 7. Visualize Model Importance & Gompertz Curve"),
        code_cell(build_viz_cell(scenario)),
        
        md_cell(
            "## xAI LIME Interpretation\n\n"
            "LIME (Local Interpretable Model-agnostic Explanations) giải thích từng dự đoán cục bộ\n"
            "bằng cách xây dựng mô hình tuyến tính xấp xỉ quanh sample đang xét."
        ),
        code_cell(build_lime_cell(scenario)),
        
        md_cell(
            "## xAI SHAP Interpretation\n\n"
            "SHAP (SHapley Additive exPlanations) cung cấp:\n"
            "- **Summary Plot**: Tổng quan đóng góp của mỗi input feature.\n"
            "- **Waterfall Plot**: Chi tiết đóng góp cho từng sample cụ thể."
        ),
        code_cell(build_shap_cell(scenario)),
        
        md_cell("## 8. Lưu Model & Benchmark"),
        code_cell(build_save_benchmark_cell(scenario)),
    ]
    return make_notebook(cells)


def main():
    output_dir = Path(r"e:\thesis\notebooks")
    
    for scenario in SCENARIOS:
        nb = build_notebook(scenario)
        filename = f"KB{scenario['kb']}_{scenario['name']}.ipynb"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        
        n_cells = len(nb['cells'])
        size_kb = filepath.stat().st_size / 1024
        print(f"[OK] Created {filename}: {n_cells} cells, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
