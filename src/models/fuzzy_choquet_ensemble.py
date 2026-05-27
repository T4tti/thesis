"""
fuzzy_choquet_ensemble.py
=========================
Module tổng hợp dự đoán từ nhiều mô hình bằng Fuzzy Choquet Integral.

Tác giả: Thesis Research
Kịch bản: 7 (FI-TTX), 8 (FI-PLL), 9 (FI-TTLPXL)

Tài liệu tham khảo:
- Grabisch, M. (1995). Fuzzy integral in multicriteria decision making.
- Torra, V. (2004). OWA operators in data modeling and reidentification.
- Choquet, G. (1953). Theory of capacities.

Fuzzy Choquet Integral:
    C_mu(f) = sum_{i=1}^{n} [f(sigma(i)) - f(sigma(i-1))] * mu(A_sigma(i))

Trong đó:
    - f(sigma(i)): giá trị thứ i sau khi sắp xếp tăng dần
    - A_sigma(i) = {sigma(i), ..., sigma(n)}: tập hợp con từ phần tử thứ i
    - mu(A): fuzzy measure (khả năng đo đạc fuzzy) của tập A

Cách học tham số mu:
    - Sử dụng 2^n - 2 tham số (không kể empty set và full set)
    - Ràng buộc đơn điệu: mu(A) <= mu(B) nếu A ⊆ B
    - Tối ưu bằng gradient descent với soft monotone constraints
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fuzzy Measure (Capacité) Utilities
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyMeasure:
    """
    Lưu trữ và quản lý Fuzzy Measure (capacité) cho n phần tử.
    
    Sử dụng dictionary với key là tuple (frozen set) các chỉ số.
    Đảm bảo ràng buộc đơn điệu (monotonicity):
        A ⊆ B => mu(A) <= mu(B)
    và ràng buộc boundary:
        mu({}) = 0, mu(N) = 1
    """

    def __init__(self, n_sources: int) -> None:
        self.n = n_sources
        self.all_subsets: List[Tuple[int, ...]] = []
        # Generate all non-empty subsets
        for r in range(1, n_sources + 1):
            for combo in itertools.combinations(range(n_sources), r):
                self.all_subsets.append(combo)
        # Initialize with symmetric (Shapley) measure: mu(A) = |A|/n
        self._mu: Dict[Tuple[int, ...], float] = {}
        for subset in self.all_subsets:
            self._mu[subset] = len(subset) / n_sources

    def get(self, subset: Tuple[int, ...]) -> float:
        """Lấy giá trị mu cho một tập con."""
        if not subset:
            return 0.0
        key = tuple(sorted(subset))
        return self._mu.get(key, len(subset) / self.n)

    def set(self, subset: Tuple[int, ...], value: float) -> None:
        """Cập nhật giá trị mu, clamp về [0, 1]."""
        key = tuple(sorted(subset))
        self._mu[key] = float(np.clip(value, 0.0, 1.0))

    def from_vector(self, params: np.ndarray) -> None:
        """Nạp params (sigmoid-transformed) vào dictionary mu."""
        subsets_ordered = sorted(self.all_subsets, key=lambda s: (len(s), s))
        for idx, subset in enumerate(subsets_ordered):
            if idx < len(params):
                self.set(subset, float(params[idx]))

    def to_vector(self) -> np.ndarray:
        """Xuất params dạng vector để tối ưu."""
        subsets_ordered = sorted(self.all_subsets, key=lambda s: (len(s), s))
        return np.array([self.get(s) for s in subsets_ordered], dtype=np.float64)

    def enforce_monotonicity(self) -> None:
        """Ép ràng buộc đơn điệu theo bottom-up (từ tập nhỏ đến tập lớn)."""
        subsets_ordered = sorted(self.all_subsets, key=lambda s: len(s))
        for subset in subsets_ordered:
            for elem in range(self.n):
                if elem not in subset:
                    # Subset ∪ {elem} phải >= mu(subset)
                    larger = tuple(sorted(subset + (elem,)))
                    if larger in self._mu:
                        new_val = max(self._mu[larger], self.get(subset))
                        self._mu[larger] = new_val
        # Ensure full set = 1.0
        full = tuple(range(self.n))
        self._mu[full] = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fuzzy Choquet Integral (FCI) computation
# ─────────────────────────────────────────────────────────────────────────────

def choquet_integral_single(
    scores: np.ndarray,
    mu: FuzzyMeasure,
) -> float:
    """
    Tính Choquet Integral cho một vector điểm số.

    Parameters
    ----------
    scores : np.ndarray, shape (n_sources,)
        Điểm số của từng mô hình cho một sample (ví dụ: xác suất class k).
    mu : FuzzyMeasure
        Fuzzy measure đã học.

    Returns
    -------
    float : Giá trị tích phân Choquet.

    Formula
    -------
    C_mu(f) = sum_{i=1}^{n} [f_sigma(i) - f_sigma(i-1)] * mu(A_sigma(i))
    
    Trong đó sigma là hoán vị sắp xếp f tăng dần, và A_sigma(i) là tập hợp
    {sigma(i), sigma(i+1), ..., sigma(n)}.
    """
    n = len(scores)
    # Sắp xếp tăng dần
    sigma = np.argsort(scores)  # chỉ số theo thứ tự tăng dần
    scores_sorted = scores[sigma]

    result = 0.0
    for i in range(n):
        # A_sigma(i) = {sigma(i), sigma(i+1), ..., sigma(n-1)}
        a_sigma_i = tuple(sorted(sigma[i:]))
        mu_val = mu.get(a_sigma_i)

        if i == 0:
            f_prev = 0.0
        else:
            f_prev = float(scores_sorted[i - 1])

        f_curr = float(scores_sorted[i])
        result += (f_curr - f_prev) * mu_val

    return result


def choquet_integral_batch(
    proba_3d: np.ndarray,
    mu: FuzzyMeasure,
) -> np.ndarray:
    """
    Tính Choquet Integral theo lô (batch) cho bài toán phân loại đa lớp.

    Parameters
    ----------
    proba_3d : np.ndarray, shape (n_samples, n_sources, n_classes)
        Ma trận xác suất dự đoán của từng mô hình, từng sample, từng class.
    mu : FuzzyMeasure
        Fuzzy measure đã học.

    Returns
    -------
    np.ndarray, shape (n_samples, n_classes)
        Xác suất tổng hợp sau khi áp dụng Fuzzy Choquet Integral.
    """
    n_samples, n_sources, n_classes = proba_3d.shape
    result = np.zeros((n_samples, n_classes), dtype=np.float64)

    for c in range(n_classes):
        # proba_3d[:, :, c] -> (n_samples, n_sources)
        class_scores = proba_3d[:, :, c]  # (n_samples, n_sources)
        for sample_idx in range(n_samples):
            result[sample_idx, c] = choquet_integral_single(
                class_scores[sample_idx], mu
            )

    # Normalize to ensure probabilities sum to 1
    row_sums = result.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    result = result / row_sums

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Learning Fuzzy Measures từ dữ liệu validation
# ─────────────────────────────────────────────────────────────────────────────

def _objective_fn(
    params: np.ndarray,
    proba_3d: np.ndarray,
    y_true: np.ndarray,
    mu: FuzzyMeasure,
    lambda_mono: float = 0.1,
) -> float:
    """
    Hàm mục tiêu cho tối ưu hóa fuzzy measure.
    
    Loss = CrossEntropy(Choquet(proba), y_true) + lambda * MonotonicityPenalty
    """
    # Nạp params vào mu (qua sigmoid để đảm bảo [0, 1])
    sigm_params = 1.0 / (1.0 + np.exp(-params))
    mu.from_vector(sigm_params)
    # Ensure full set = 1.0
    mu.set(tuple(range(mu.n)), 1.0)

    # Tính Choquet integral
    fused_proba = choquet_integral_batch(proba_3d, mu)  # (N, C)
    fused_proba = np.clip(fused_proba, 1e-9, 1.0)

    # Cross-entropy loss
    n_samples = len(y_true)
    ce_loss = -np.mean(np.log(fused_proba[np.arange(n_samples), y_true.astype(int)]))

    # Monotonicity penalty (soft constraint)
    mono_penalty = 0.0
    subsets_ordered = sorted(mu.all_subsets, key=lambda s: len(s))
    for subset in subsets_ordered:
        for elem in range(mu.n):
            if elem not in subset:
                larger = tuple(sorted(subset + (elem,)))
                diff = mu.get(subset) - mu.get(larger)
                if diff > 0:
                    mono_penalty += diff ** 2

    return ce_loss + lambda_mono * mono_penalty


def learn_fuzzy_measure(
    proba_3d: np.ndarray,
    y_true: np.ndarray,
    n_sources: int,
    max_iter: int = 500,
    lambda_mono: float = 0.1,
    verbose: bool = True,
) -> FuzzyMeasure:
    """
    Học Fuzzy Measure từ tập validation bằng tối ưu hóa scipy.

    Parameters
    ----------
    proba_3d : np.ndarray, shape (n_val_samples, n_sources, n_classes)
    y_true : np.ndarray, shape (n_val_samples,)
    n_sources : int
        Số lượng mô hình cơ sở.
    max_iter : int
        Số lần lặp tối đa.
    lambda_mono : float
        Trọng số penalty đơn điệu.
    verbose : bool

    Returns
    -------
    FuzzyMeasure : Fuzzy measure tối ưu.
    """
    mu = FuzzyMeasure(n_sources)
    n_params = len(mu.to_vector())

    # Khởi tạo params ≈ |A|/n (trước khi qua sigmoid)
    init_vals = mu.to_vector()
    # logit của giá trị khởi tạo
    init_params = np.log(init_vals / (1.0 - init_vals + 1e-9))

    if verbose:
        print(f"[FuzzyMeasure] Tối ưu hóa với {n_params} tham số, max_iter={max_iter}...")

    result = minimize(
        _objective_fn,
        init_params,
        args=(proba_3d, y_true, mu, lambda_mono),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6},
    )

    # Nạp params tốt nhất
    best_params = 1.0 / (1.0 + np.exp(-result.x))
    mu.from_vector(best_params)
    mu.set(tuple(range(n_sources)), 1.0)
    mu.enforce_monotonicity()

    if verbose:
        print(f"[FuzzyMeasure] Loss cuối: {result.fun:.6f}")
        print(f"[FuzzyMeasure] Converged: {result.success}")
        print("[FuzzyMeasure] Shapley values (importance per source):")
        for i in range(n_sources):
            shapley = _shapley_value(mu, i)
            print(f"  Source {i}: {shapley:.4f}")

    return mu


def _shapley_value(mu: FuzzyMeasure, i: int) -> float:
    """
    Tính Shapley value của nguồn thứ i từ Fuzzy Measure mu.
    
    Shapley(i) = sum_{S not containing i} [|S|!(n-|S|-1)!/n!] * [mu(S∪{i}) - mu(S)]
    """
    n = mu.n
    shapley = 0.0
    others = [j for j in range(n) if j != i]

    for r in range(n):
        for combo in itertools.combinations(others, r):
            s = tuple(sorted(combo))
            s_with_i = tuple(sorted(combo + (i,)))
            marginal = mu.get(s_with_i) - mu.get(s)
            coeff = (
                np.math.factorial(len(s))
                * np.math.factorial(n - len(s) - 1)
                / np.math.factorial(n)
            )
            shapley += coeff * marginal

    return shapley


# ─────────────────────────────────────────────────────────────────────────────
# 4. FuzzyChoquetEnsemble: Main class
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyChoquetEnsemble:
    """
    Ensemble tổng hợp xác suất của nhiều mô hình bằng Fuzzy Choquet Integral.
    
    Workflow:
    1. Nhận OOF predictions (Out-Of-Fold) từ các mô hình baseline.
    2. Học fuzzy measure trên tập validation (tối ưu cross-entropy).
    3. Áp dụng fuzzy measure đã học lên tập test.
    4. Báo cáo Shapley values (tầm quan trọng từng mô hình).

    Parameters
    ----------
    model_names : List[str]
        Tên của từng mô hình cơ sở.
    n_classes : int
        Số lớp phân loại.
    max_iter : int
        Số lần lặp tối ưu hóa.
    lambda_mono : float
        Hệ số penalty đơn điệu.
    """

    def __init__(
        self,
        model_names: List[str],
        n_classes: int = 3,
        max_iter: int = 500,
        lambda_mono: float = 0.1,
    ) -> None:
        self.model_names = model_names
        self.n_sources = len(model_names)
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.lambda_mono = lambda_mono
        self.mu: Optional[FuzzyMeasure] = None
        self._shapley_values: Optional[Dict[str, float]] = None

    def fit(
        self,
        val_probas: List[np.ndarray],
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> "FuzzyChoquetEnsemble":
        """
        Học fuzzy measure từ xác suất validation.
        
        Parameters
        ----------
        val_probas : List[np.ndarray]
            List gồm n_sources phần tử, mỗi phần tử có shape (n_val, n_classes).
        y_val : np.ndarray, shape (n_val,)
            Nhãn thực tế tập validation.
        """
        # Stack: (n_val, n_sources, n_classes)
        proba_3d = np.stack(val_probas, axis=1)
        assert proba_3d.shape == (len(y_val), self.n_sources, self.n_classes), (
            f"Kích thước không khớp: {proba_3d.shape} vs "
            f"({len(y_val)}, {self.n_sources}, {self.n_classes})"
        )

        self.mu = learn_fuzzy_measure(
            proba_3d,
            y_val,
            n_sources=self.n_sources,
            max_iter=self.max_iter,
            lambda_mono=self.lambda_mono,
            verbose=verbose,
        )

        # Tính Shapley values
        self._shapley_values = {
            name: _shapley_value(self.mu, i)
            for i, name in enumerate(self.model_names)
        }

        return self

    def predict_proba(self, test_probas: List[np.ndarray]) -> np.ndarray:
        """
        Tổng hợp xác suất trên tập test.
        
        Parameters
        ----------
        test_probas : List[np.ndarray]
            List gồm n_sources phần tử, mỗi phần tử có shape (n_test, n_classes).

        Returns
        -------
        np.ndarray, shape (n_test, n_classes)
        """
        if self.mu is None:
            raise RuntimeError("Chưa gọi .fit(). Hãy chạy học fuzzy measure trước.")
        proba_3d = np.stack(test_probas, axis=1)
        return choquet_integral_batch(proba_3d, self.mu)

    def predict(self, test_probas: List[np.ndarray]) -> np.ndarray:
        """Dự đoán nhãn lớp từ xác suất tổng hợp."""
        fused_proba = self.predict_proba(test_probas)
        return np.argmax(fused_proba, axis=1)

    def evaluate(
        self,
        test_probas: List[np.ndarray],
        y_test: np.ndarray,
        split_name: str = "Test",
    ) -> Dict[str, float]:
        """
        Đánh giá ensemble trên tập test.

        Returns
        -------
        Dict[str, float] : Các metric bao gồm Accuracy, Macro_F1, AUC, QWK.
        """
        fused_proba = self.predict_proba(test_probas)
        y_pred = np.argmax(fused_proba, axis=1)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

        try:
            y_bin = label_binarize(y_test, classes=list(range(self.n_classes)))
            auc_score = roc_auc_score(y_bin, fused_proba, average='weighted', multi_class='ovr')
        except Exception:
            auc_score = float('nan')

        metrics = {
            'Split': split_name,
            'Accuracy': acc,
            'Macro_F1': f1_macro,
            'Weighted_F1': f1_weighted,
            'AUC': auc_score,
            'QWK': qwk,
        }

        print(f"\n[{split_name}] Fuzzy Choquet Ensemble Results:")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {f1_macro:.4f}")
        print(f"  Weighted F1: {f1_weighted:.4f}")
        print(f"  AUC:         {auc_score:.4f}")
        print(f"  QWK:         {qwk:.4f}")

        return metrics

    def get_shapley_values(self) -> Dict[str, float]:
        """Trả về Shapley value (tầm quan trọng) của từng mô hình."""
        if self._shapley_values is None:
            raise RuntimeError("Chưa gọi .fit(). Hãy huấn luyện trước.")
        return self._shapley_values

    def save(self, path: Union[str, Path]) -> None:
        """Lưu fuzzy measure đã học."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'model_names': self.model_names,
            'n_sources': self.n_sources,
            'n_classes': self.n_classes,
            'mu_vector': self.mu.to_vector() if self.mu else None,
            'shapley_values': self._shapley_values,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[FuzzyChoquetEnsemble] Đã lưu tại: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FuzzyChoquetEnsemble":
        """Nạp fuzzy measure đã học từ file."""
        import pickle
        with open(Path(path), 'rb') as f:
            state = pickle.load(f)
        obj = cls(
            model_names=state['model_names'],
            n_classes=state['n_classes'],
        )
        if state['mu_vector'] is not None:
            obj.mu = FuzzyMeasure(state['n_sources'])
            obj.mu.from_vector(state['mu_vector'])
            obj.mu.enforce_monotonicity()
        obj._shapley_values = state['shapley_values']
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 5. Utility: Load pre-saved OOF probabilities
# ─────────────────────────────────────────────────────────────────────────────

def load_oof_probas(
    artifact_dir: Union[str, Path],
    model_name: str,
    split: str = "val",
) -> Optional[np.ndarray]:
    """
    Tải xác suất OOF đã lưu từ thư mục artifact.
    
    Tìm kiếm các file có tên dạng:
    - {model_name}_oof_{split}_proba.npy
    - {model_name}_{split}_proba.npy

    Parameters
    ----------
    artifact_dir : str | Path
    model_name : str
        Ví dụ: 'tBiLSTM', 'tcn', 'xgboost'
    split : str
        'val' hoặc 'test'

    Returns
    -------
    np.ndarray hoặc None nếu không tìm thấy.
    """
    artifact_dir = Path(artifact_dir)
    candidates = [
        artifact_dir / f"{model_name}_oof_{split}_proba.npy",
        artifact_dir / f"{model_name}_{split}_proba.npy",
        artifact_dir / f"{model_name.lower()}_oof_{split}_proba.npy",
        artifact_dir / f"{model_name.lower()}_{split}_proba.npy",
    ]
    for path in candidates:
        if path.exists():
            arr = np.load(str(path))
            print(f"[load_oof_probas] Đã tải: {path} - shape: {arr.shape}")
            return arr

    warnings.warn(
        f"[load_oof_probas] Không tìm thấy xác suất cho '{model_name}/{split}'. "
        f"Đã thử: {[str(c) for c in candidates]}"
    )
    return None


def load_oof_labels(
    artifact_dir: Union[str, Path],
    split: str = "val",
) -> Optional[np.ndarray]:
    """Tải nhãn thực tế cho split."""
    artifact_dir = Path(artifact_dir)
    candidates = [
        artifact_dir / f"y_{split}.npy",
        artifact_dir / f"y_{split}_true.npy",
        artifact_dir / f"{split}_labels.npy",
    ]
    for path in candidates:
        if path.exists():
            arr = np.load(str(path))
            print(f"[load_oof_labels] Đã tải: {path} - shape: {arr.shape}")
            return arr
    warnings.warn(f"[load_oof_labels] Không tìm thấy nhãn cho split='{split}'.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot Utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_shapley_bar(
    shapley_values: Dict[str, float],
    ensemble_name: str = "Fuzzy Choquet Ensemble",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Vẽ biểu đồ cột Shapley values."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    names = list(shapley_values.keys())
    values = [shapley_values[n] for n in names]
    colors = sns.color_palette("viridis", len(names))

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4), dpi=150)
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel("Shapley Value (Model Importance)")
    ax.set_title(f"{ensemble_name}\nModel Importance via Shapley Values", fontweight='bold')
    ax.set_xlim(0, max(values) * 1.2)
    for bar, val in zip(bars, values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_fuzzy_measure_heatmap(
    mu: FuzzyMeasure,
    model_names: List[str],
    ensemble_name: str = "Fuzzy Choquet",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Hiển thị fuzzy measure dưới dạng heatmap cho các tập kết hợp cặp."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = mu.n
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = mu.get((i,))
        for j in range(i + 1, n):
            val = mu.get((i, j))
            matrix[i, j] = val
            matrix[j, i] = val

    df_heat = pd.DataFrame(matrix, index=model_names, columns=model_names)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)), dpi=150)
    sns.heatmap(df_heat, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title(f"{ensemble_name}\nFuzzy Measure μ(A) - Pairwise Subsets", fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
