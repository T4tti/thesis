import os
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc, cohen_kappa_score,
    confusion_matrix, classification_report,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect_kaggle_runtime() -> bool:
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '').strip():
        return True
    return Path('/kaggle/input').exists() and Path('/kaggle/working').exists()


IN_KAGGLE = detect_kaggle_runtime()


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'data').exists() and (p / 'src').exists():
            return p
    return start


PROJECT_ROOT = Path('/kaggle/working') if IN_KAGGLE else find_project_root(Path.cwd().resolve())
ARTIFACT_DIR = PROJECT_ROOT / 'credit_rating_artifacts'
DMF_ARTIFACT_DIR = ARTIFACT_DIR / 'dmf_gat_lstm'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DMF_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print('Device:', device)
print('Project root:', PROJECT_ROOT)
print('DMF artifact dir:', DMF_ARTIFACT_DIR)


# --- CELL BOUNDARY ---

# ── Shared two-tier loss, calibration, and probability metrics ──────────────
# Inlined from src/models/losses.py for standalone execution (e.g. Kaggle).
# The benchmark protocol uses plain multiclass negative log-likelihood (NLL).
# The ordinal ablation adds a normalized squared CDF distance (EMD) without
# changing the model output shape or using CORAL/CORN.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

BENCHMARK_PROTOCOL = "benchmark_ce"
ORDINAL_PROTOCOL = "ordinal_ce_emd"
SUPPORTED_PROTOCOLS = (BENCHMARK_PROTOCOL, ORDINAL_PROTOCOL)
DEFAULT_ORDINAL_LAMBDA = 0.10
DEFAULT_LABEL_ORDER = ("Distressed", "HY", "IG")
_EPS = 1e-12

# Papermill-injectable parameters.
LOSS_PROTOCOL = str(globals().get("LOSS_PROTOCOL", "benchmark_ce")).strip().lower()
ORDINAL_LAMBDA = float(globals().get("ORDINAL_LAMBDA", 0.10))
TARGET_ORDERED_LABELS = ["Distressed", "HY", "IG"]


def normalize_protocol(protocol: str) -> str:
    """Validate and normalize a two-tier loss protocol name."""
    normalized = str(protocol).strip().lower()
    if normalized not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"Unsupported loss protocol {protocol!r}; expected one of "
            f"{SUPPORTED_PROTOCOLS}."
        )
    return normalized


def _normalize_probabilities(probabilities: Any) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes>=2)")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain NaN or infinite values")
    probs = np.clip(probs, _EPS, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each probability row must have a positive sum")
    return probs / row_sums


def _validate_targets(targets: Any, n_samples: int, n_classes: int) -> np.ndarray:
    y_true = np.asarray(targets, dtype=np.int64).reshape(-1)
    if len(y_true) != n_samples:
        raise ValueError("targets length does not match probabilities")
    if np.any((y_true < 0) | (y_true >= n_classes)):
        raise ValueError("targets contain class ids outside the probability columns")
    return y_true


def numpy_nll(probabilities: Any, targets: Any) -> float:
    """Mean multiclass negative log-likelihood from probabilities."""
    probs = _normalize_probabilities(probabilities)
    y_true = _validate_targets(targets, len(probs), probs.shape[1])
    return float(-np.log(probs[np.arange(len(y_true)), y_true]).mean())


def numpy_cdf_emd2(probabilities: Any, targets: Any) -> float:
    """Normalized squared CDF distance for ordered multiclass probabilities."""
    probs = _normalize_probabilities(probabilities)
    y_true = _validate_targets(targets, len(probs), probs.shape[1])
    target_one_hot = np.eye(probs.shape[1], dtype=np.float64)[y_true]
    predicted_cdf = np.cumsum(probs, axis=1)[:, :-1]
    target_cdf = np.cumsum(target_one_hot, axis=1)[:, :-1]
    return float(np.square(predicted_cdf - target_cdf).mean())


def numpy_objective(
    probabilities: Any,
    targets: Any,
    *,
    protocol: str = BENCHMARK_PROTOCOL,
    ordinal_lambda: float = DEFAULT_ORDINAL_LAMBDA,
) -> float:
    """Evaluate the selected two-tier objective from class probabilities."""
    protocol = normalize_protocol(protocol)
    nll = numpy_nll(probabilities, targets)
    if protocol == BENCHMARK_PROTOCOL:
        return nll
    return nll + float(ordinal_lambda) * numpy_cdf_emd2(probabilities, targets)


def benchmark_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Plain multiclass CE used by the primary benchmark."""
    return F.cross_entropy(logits.float(), targets.long(), label_smoothing=0.0)


def cdf_emd2(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Differentiable normalized squared CDF distance from logits."""
    logits = logits.float()
    targets = targets.long()
    probabilities = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(
        targets,
        num_classes=probabilities.shape[1],
    ).to(dtype=probabilities.dtype)
    predicted_cdf = probabilities.cumsum(dim=1)[:, :-1]
    target_cdf = target_one_hot.cumsum(dim=1)[:, :-1]
    return (predicted_cdf - target_cdf).square().mean()


def ordinal_ce_emd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ordinal_lambda: float = DEFAULT_ORDINAL_LAMBDA,
) -> torch.Tensor:
    """CE plus normalized squared CDF-EMD for the ordinal ablation."""
    return benchmark_ce(logits, targets) + float(ordinal_lambda) * cdf_emd2(
        logits,
        targets,
    )


class TwoTierClassificationLoss(nn.Module):
    """Single loss API shared by all neural notebook baselines."""

    def __init__(
        self,
        protocol: str = BENCHMARK_PROTOCOL,
        ordinal_lambda: float = DEFAULT_ORDINAL_LAMBDA,
    ) -> None:
        super().__init__()
        self.protocol = normalize_protocol(protocol)
        self.ordinal_lambda = float(ordinal_lambda)
        if self.ordinal_lambda < 0.0:
            raise ValueError("ordinal_lambda must be non-negative")

    def monitor_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Always return comparable plain NLL for curves and reports."""
        return benchmark_ce(logits, targets)

    def loss_parts(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        nll = benchmark_ce(logits, targets)
        emd = cdf_emd2(logits, targets)
        objective = (
            nll
            if self.protocol == BENCHMARK_PROTOCOL
            else nll + self.ordinal_lambda * emd
        )
        return {
            "objective": objective,
            "nll": nll,
            "cdf_emd2": emd,
            # Compatibility aliases for existing notebook training loops.
            "ce_loss": nll,
            "aux_loss": objective - nll,
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        return self.loss_parts(logits, targets)["objective"]


def build_loss(
    protocol: str = BENCHMARK_PROTOCOL,
    ordinal_lambda: float = DEFAULT_ORDINAL_LAMBDA,
) -> TwoTierClassificationLoss:
    """Factory used by notebook import and fallback paths."""
    return TwoTierClassificationLoss(
        protocol=protocol,
        ordinal_lambda=ordinal_lambda,
    )


def apply_temperature(probabilities: Any, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to probabilities via log-probabilities."""
    probs = _normalize_probabilities(probabilities)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scaled_logits = np.log(probs) / temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    scaled = np.exp(scaled_logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    probabilities: Any,
    targets: Any,
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit one temperature on validation probabilities by minimizing NLL."""
    probs = _normalize_probabilities(probabilities)
    y_true = _validate_targets(targets, len(probs), probs.shape[1])
    lower, upper = map(float, bounds)
    if not 0.0 < lower < upper:
        raise ValueError("temperature bounds must satisfy 0 < lower < upper")
    result = minimize_scalar(
        lambda value: numpy_nll(apply_temperature(probs, value), y_true),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success or not np.isfinite(result.x):
        return 1.0
    return float(np.clip(result.x, lower, upper))


@dataclass(frozen=True)
class TemperatureCalibrationResult:
    """Cross-fitted validation and final test calibration output."""

    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    temperature: float
    fold_temperatures: tuple[float, ...]
    n_splits: int


def cross_fit_temperature_scaling(
    validation_probabilities: Any,
    validation_targets: Any,
    test_probabilities: Any,
    *,
    max_splits: int = 5,
    seed: int = 42,
) -> TemperatureCalibrationResult:
    """Cross-fit validation calibration and fit one final validation temperature."""
    val_probs = _normalize_probabilities(validation_probabilities)
    test_probs = _normalize_probabilities(test_probabilities)
    if val_probs.shape[1] != test_probs.shape[1]:
        raise ValueError("validation and test probabilities need equal class counts")
    y_val = _validate_targets(validation_targets, len(val_probs), val_probs.shape[1])

    class_counts = np.bincount(y_val, minlength=val_probs.shape[1])
    positive_counts = class_counts[class_counts > 0]
    n_splits = min(int(max_splits), int(positive_counts.min())) if len(positive_counts) else 0
    calibrated_val = np.empty_like(val_probs)
    fold_temperatures: list[float] = []

    if n_splits >= 2:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        for fit_idx, holdout_idx in splitter.split(val_probs, y_val):
            temperature = fit_temperature(val_probs[fit_idx], y_val[fit_idx])
            calibrated_val[holdout_idx] = apply_temperature(
                val_probs[holdout_idx],
                temperature,
            )
            fold_temperatures.append(temperature)
    else:
        n_splits = 1
        temperature = fit_temperature(val_probs, y_val)
        calibrated_val[:] = apply_temperature(val_probs, temperature)
        fold_temperatures.append(temperature)

    final_temperature = fit_temperature(val_probs, y_val)
    calibrated_test = apply_temperature(test_probs, final_temperature)
    return TemperatureCalibrationResult(
        validation_probabilities=calibrated_val,
        test_probabilities=calibrated_test,
        temperature=final_temperature,
        fold_temperatures=tuple(fold_temperatures),
        n_splits=n_splits,
    )


def multiclass_brier_score(probabilities: Any, targets: Any) -> float:
    """Mean multiclass Brier score."""
    probs = _normalize_probabilities(probabilities)
    y_true = _validate_targets(targets, len(probs), probs.shape[1])
    target_one_hot = np.eye(probs.shape[1], dtype=np.float64)[y_true]
    return float(np.square(probs - target_one_hot).sum(axis=1).mean())


def expected_calibration_error(
    probabilities: Any,
    targets: Any,
    *,
    n_bins: int = 15,
) -> float:
    """Top-label expected calibration error."""
    probs = _normalize_probabilities(probabilities)
    y_true = _validate_targets(targets, len(probs), probs.shape[1])
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correctness = predictions == y_true
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for index in range(int(n_bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (confidence >= lower) & (confidence < upper)
            if index < int(n_bins) - 1
            else (confidence >= lower) & (confidence <= upper)
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return float(ece)


def probability_report(
    targets: Any,
    probabilities: Any,
    *,
    protocol: str = BENCHMARK_PROTOCOL,
    ordinal_lambda: float = DEFAULT_ORDINAL_LAMBDA,
    last_y: Any | None = None,
) -> dict[str, float | str]:
    """Return the common benchmark/ablation metric contract."""
    probs = _normalize_probabilities(probabilities)
    y_true = _validate_targets(targets, len(probs), probs.shape[1])
    y_pred = probs.argmax(axis=1)
    nll = numpy_nll(probs, y_true)
    emd = numpy_cdf_emd2(probs, y_true)
    normalized_protocol = normalize_protocol(protocol)
    objective = (
        nll
        if normalized_protocol == BENCHMARK_PROTOCOL
        else nll + float(ordinal_lambda) * emd
    )
    y_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
    try:
        auc = float(
            roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
        )
    except ValueError:
        auc = float("nan")
    report: dict[str, float | str] = {
        "Protocol": normalized_protocol,
        "NLL": nll,
        "Objective": objective,
        "CDF_EMD2": emd,
        "Brier": multiclass_brier_score(probs, y_true),
        "ECE": expected_calibration_error(probs, y_true),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "QWK": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "Ordinal_MAE": float(np.abs(y_true - y_pred).mean()),
        "AUC_ROC_OVR_Macro": auc,
        "AUC": auc,
    }
    if last_y is None:
        report["ChgAcc"] = float("nan")
    else:
        previous = np.asarray(last_y, dtype=np.int64).reshape(-1)
        if len(previous) != len(y_true):
            raise ValueError("last_y length does not match targets")
        change_mask = previous != y_true
        report["ChgAcc"] = (
            float(accuracy_score(y_true[change_mask], y_pred[change_mask]))
            if change_mask.any()
            else float("nan")
        )
    return report


def reliability_weights_from_nll(
    validation_targets: Any,
    model_probabilities: Iterable[Any],
) -> np.ndarray:
    """Normalize exp(-validation NLL) reliability weights across models."""
    losses = np.asarray(
        [numpy_nll(probabilities, validation_targets) for probabilities in model_probabilities],
        dtype=np.float64,
    )
    raw = np.exp(-(losses - losses.min()))
    return raw / raw.sum()


# ── Protocol validation ─────────────────────────────────────────────────────
TWO_TIER_LOSS_SOURCE = "inline (losses.py)"
if LOSS_PROTOCOL not in {BENCHMARK_PROTOCOL, ORDINAL_PROTOCOL}:
    raise ValueError(f"LOSS_PROTOCOL must be {BENCHMARK_PROTOCOL!r} or {ORDINAL_PROTOCOL!r}")
print(f"Two-tier loss source={TWO_TIER_LOSS_SOURCE} | protocol={LOSS_PROTOCOL} | ordinal_lambda={ORDINAL_LAMBDA:.2f}")


# --- CELL BOUNDARY ---

FINANCIAL_FEATURES = [
    'current_ratio', 'debt_equity_ratio', 'gross_profit_margin', 'operating_profit_margin',
    'ebit_margin', 'pretax_profit_margin', 'net_profit_margin', 'asset_turnover',
    'roe', 'roa', 'operating_cashflow_ps', 'free_cashflow_ps'
]
TARGET_COL = 'rating_detail'
TARGET_ORDERED_LABELS = ['Distressed', 'HY', 'IG']


def resolve_split_path(default_path, local_fallbacks=None):
    candidates = [Path(default_path)]
    for p in (local_fallbacks or []):
        p_obj = Path(p)
        candidates.append(PROJECT_ROOT / p_obj if not p_obj.is_absolute() else p_obj)
    if IN_KAGGLE:
        kaggle_root = Path('/kaggle/input')
        expanded = []
        for p in candidates:
            expanded.append(p)
            if not p.exists() and kaggle_root.exists():
                expanded.extend(kaggle_root.rglob(p.name))
        candidates = expanded
    seen = set()
    deduped = []
    for p in candidates:
        p = Path(p)
        key = str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    for p in deduped:
        if p.exists():
            return p
    raise FileNotFoundError(f'Khong tim thay file split: {deduped}')


TRAIN_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/train_augmented_timegan.csv',
    ['data/processed/test/train.csv'],
)
VAL_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/val.csv',
    ['data/processed/test/val.csv'],
)
TEST_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/test.csv',
    ['data/processed/test/test.csv'],
)


def load_split(path, split_name):
    frame = pd.read_csv(path)
    frame = frame.copy()
    frame['__split__'] = split_name
    frame['__split_row_index__'] = np.arange(len(frame), dtype=int)
    if 'row_id' not in frame.columns:
        frame['row_id'] = [f'{split_name}_{i:06d}' for i in range(len(frame))]
    else:
        frame['row_id'] = frame['row_id'].astype(str)
    return frame


train_df = load_split(TRAIN_PATH, 'train')
val_df = load_split(VAL_PATH, 'val')
test_df = load_split(TEST_PATH, 'test')
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

split_contract = {
    'train_path': str(TRAIN_PATH),
    'val_path': str(VAL_PATH),
    'test_path': str(TEST_PATH),
    'train_rows': int(len(train_df)),
    'val_rows': int(len(val_df)),
    'test_rows': int(len(test_df)),
    'row_id_rule': 'existing row_id if present, otherwise <split>_<zero_padded_original_split_index>',
}
print('DMF split contract:', split_contract)

df = df.dropna(subset=[TARGET_COL]).copy()
target_as_num = pd.to_numeric(df[TARGET_COL], errors='coerce')
if target_as_num.notna().all():
    df[TARGET_COL] = target_as_num.astype(int)
    observed = sorted(df[TARGET_COL].unique().tolist())
    raw_to_id = {int(v): i for i, v in enumerate(observed)}
    id_to_raw = {i: int(v) for v, i in raw_to_id.items()}
    df[TARGET_COL] = df[TARGET_COL].map(raw_to_id).astype(int)
else:
    tgt = df[TARGET_COL].astype(str).str.strip()
    observed = sorted(tgt.unique().tolist())
    ordered = [x for x in TARGET_ORDERED_LABELS if x in observed] if set(observed).issubset(set(TARGET_ORDERED_LABELS)) else observed
    raw_to_id = {v: i for i, v in enumerate(ordered)}
    id_to_raw = {i: v for i, v in raw_to_id.items()}
    df[TARGET_COL] = tgt.map(raw_to_id).astype(int)

n_classes = int(df[TARGET_COL].nunique())
if n_classes != 3:
    raise ValueError(f"Two-tier benchmark requires exactly 3 classes, got {n_classes}.")
id_to_raw = {idx: label for idx, label in enumerate(TARGET_ORDERED_LABELS)}
raw_to_id = {label: idx for idx, label in id_to_raw.items()}
print("Two-tier label contract:", id_to_raw)

label_contract = pd.DataFrame({
    'label_id': list(range(n_classes)),
    'label_name': [str(id_to_raw.get(i, i)) for i in range(n_classes)],
})
label_contract.to_csv(DMF_ARTIFACT_DIR / 'label_mapping.csv', index=False, encoding='utf-8-sig')

df['rating_date'] = pd.to_datetime(df['rating_date'], errors='coerce', format='mixed')
if 'sector' not in df.columns:
    df['sector'] = 'UNKNOWN'
df['sector'] = df['sector'].fillna('UNKNOWN').astype(str)
if 'ticker' not in df.columns:
    df['ticker'] = 'UNKNOWN'
df['ticker'] = df['ticker'].fillna('UNKNOWN').astype(str)
if 'company_name' not in df.columns:
    df['company_name'] = df['ticker']
df['company_name'] = df['company_name'].fillna(df['ticker']).astype(str)

sector_encoder = LabelEncoder()
df['sector_id'] = sector_encoder.fit_transform(df['sector'])
n_sectors = int(df['sector_id'].nunique())

train_mask_raw = df['__split__'].eq('train')
stats_ref = df.loc[train_mask_raw].copy()
for c in FINANCIAL_FEATURES:
    med = stats_ref[c].median() if stats_ref[c].notna().any() else 0.0
    df[c] = df[c].fillna(float(0.0 if pd.isna(med) else med))
for c in FINANCIAL_FEATURES:
    lo = stats_ref[c].quantile(0.01)
    hi = stats_ref[c].quantile(0.99)
    if pd.notna(lo) and pd.notna(hi):
        df[c] = df[c].clip(float(lo), float(hi))

df = df.sort_values(['ticker', 'rating_date', '__split__', '__split_row_index__']).reset_index(drop=True)
for c in FINANCIAL_FEATURES:
    df[f'{c}_delta'] = df.groupby('ticker')[c].diff().fillna(0.0)
MODEL_FEATURES = FINANCIAL_FEATURES + [f'{c}_delta' for c in FINANCIAL_FEATURES]

scaler = RobustScaler()
scaler.fit(df.loc[df['__split__'].eq('train'), MODEL_FEATURES].values)
df[MODEL_FEATURES] = scaler.transform(df[MODEL_FEATURES].values)

df['last_y'] = df.groupby('ticker')[TARGET_COL].shift(1)
df['last_y'] = df['last_y'].fillna(df[TARGET_COL]).astype(int)

x_all = torch.tensor(df[MODEL_FEATURES].values.astype(np.float32), dtype=torch.float32, device=device)
y_all = torch.tensor(df[TARGET_COL].values.astype(int), dtype=torch.long, device=device)
last_y_all = torch.tensor(df['last_y'].values.astype(int), dtype=torch.long, device=device)
sector_all = torch.tensor(df['sector_id'].values.astype(int), dtype=torch.long, device=device)

train_mask = torch.tensor(df['__split__'].eq('train').values, dtype=torch.bool, device=device)
val_mask = torch.tensor(df['__split__'].eq('val').values, dtype=torch.bool, device=device)
test_mask = torch.tensor(df['__split__'].eq('test').values, dtype=torch.bool, device=device)

train_class_counts = torch.bincount(y_all[train_mask], minlength=n_classes).float()
print('Class weighting/sampling: disabled for the shared benchmark protocol.')

print('Rows train/val/test:', int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
print('n_classes:', n_classes, '| n_sectors:', n_sectors, '| n_features:', len(MODEL_FEATURES))


# --- CELL BOUNDARY ---

CLASS0_LABEL_ID = 0

CLASS0_THRESHOLD_CONFIG = {
    'enabled': False,
    'metric': 'Class0_F2',
    'accuracy_floor_drop': 0.0,
    'min_accuracy_gain': 0.001,
    'threshold_grid': np.round(np.arange(0.05, 0.501, 0.01), 2).tolist(),
}


def predict_with_class0_threshold(proba, class0_threshold=None):
    pred = np.asarray(proba).argmax(axis=1).astype(int)
    if class0_threshold is None:
        return pred
    promote_mask = np.asarray(proba)[:, CLASS0_LABEL_ID] >= float(class0_threshold)
    pred[promote_mask] = CLASS0_LABEL_ID
    return pred


def compute_metrics(y_true, y_pred, proba, n_cls, last_y=None):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    f1m = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    f1w = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    prec = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    class_prec, class_rec, class_f1, class_support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(range(n_cls)),
        zero_division=0,
    )
    c0_precision = float(class_prec[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_prec) else float('nan')
    c0_recall = float(class_rec[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_rec) else float('nan')
    c0_f1 = float(class_f1[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_f1) else float('nan')
    c0_support = int(class_support[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_support) else 0
    if c0_precision + c0_recall > 0:
        c0_f2 = float(5.0 * c0_precision * c0_recall / (4.0 * c0_precision + c0_recall))
    else:
        c0_f2 = 0.0
    qwk = cohen_kappa_score(y_true_arr, y_pred_arr, weights='quadratic')
    try:
        y_bin = label_binarize(y_true_arr, classes=list(range(n_cls)))
        auc_score = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
    except Exception:
        auc_score = float('nan')
    ordinal_mae = np.mean(np.abs(y_true_arr - y_pred_arr))
    # ChgAcc: accuracy on samples where label changed vs last known rating.
    if last_y is not None:
        last_y_arr = np.asarray(last_y)
        change_mask = last_y_arr != y_true_arr
        if change_mask.sum() > 0:
            chg_acc = float(accuracy_score(y_true_arr[change_mask], y_pred_arr[change_mask]))
        else:
            chg_acc = float('nan')
    else:
        chg_acc = float('nan')
    return {
        'Accuracy': float(acc),
        'Precision_Weighted': float(prec),
        'Recall_Weighted': float(rec),
        'Macro_F1': float(f1m),
        'Weighted_F1': float(f1w),
        'Class0_Precision': c0_precision,
        'Class0_Recall': c0_recall,
        'Class0_F1': c0_f1,
        'Class0_F2': c0_f2,
        'Class0_Support': c0_support,
        'AUC': float(auc_score),
        'QWK': float(qwk),
        'ChgAcc': chg_acc,
        'Ordinal_MAE': float(ordinal_mae),
    }


def evaluate_logits(logits, mask, class0_threshold=None):
    probs = torch.softmax(logits[mask], dim=1).detach().cpu().numpy()
    y_true = y_all[mask].detach().cpu().numpy()
    y_pred = predict_with_class0_threshold(probs, class0_threshold=class0_threshold)
    last_y_np = last_y_all[mask].detach().cpu().numpy()
    return compute_metrics(y_true, y_pred, probs, n_classes, last_y=last_y_np), y_true, y_pred, probs


def selection_score(metrics):
    chg_acc = 0.0 if np.isnan(metrics['ChgAcc']) else metrics['ChgAcc']
    return (
        0.60 * metrics['Accuracy']
        + 0.15 * metrics['QWK']
        + 0.10 * metrics['Macro_F1']
        + 0.10 * metrics['Class0_F2']
        + 0.05 * chg_acc
        - 0.05 * metrics['Ordinal_MAE']
    )


def calibrate_class0_threshold(y_true, proba, last_y=None, config=None):
    config = config or CLASS0_THRESHOLD_CONFIG
    baseline_pred = predict_with_class0_threshold(proba, class0_threshold=None)
    baseline_metrics = compute_metrics(y_true, baseline_pred, proba, n_classes, last_y=last_y)
    rows = [{'class0_threshold': np.nan, 'candidate': 'raw_argmax', **baseline_metrics}]
    for threshold in config['threshold_grid']:
        pred = predict_with_class0_threshold(proba, class0_threshold=threshold)
        metrics = compute_metrics(y_true, pred, proba, n_classes, last_y=last_y)
        rows.append({'class0_threshold': float(threshold), 'candidate': 'class0_threshold', **metrics})
    sweep_df = pd.DataFrame(rows)
    threshold_accuracy_floor = (
        baseline_metrics['Accuracy']
        + float(config.get('min_accuracy_gain', 0.0))
        - float(config.get('accuracy_floor_drop', 0.0))
    )
    raw_candidate = sweep_df['candidate'].eq('raw_argmax')
    threshold_candidate = sweep_df['candidate'].eq('class0_threshold') & (sweep_df['Accuracy'] >= threshold_accuracy_floor)
    candidates = sweep_df[raw_candidate | threshold_candidate].copy()
    if candidates.empty:
        candidates = sweep_df[raw_candidate].copy()
    sort_cols = [config.get('metric', 'Class0_F2'), 'Accuracy', 'Macro_F1', 'QWK']
    best_row = candidates.sort_values(sort_cols, ascending=False).iloc[0]
    best_threshold = None if pd.isna(best_row['class0_threshold']) else float(best_row['class0_threshold'])
    return best_threshold, sweep_df, baseline_metrics, best_row.to_dict()


# --- CELL BOUNDARY ---

# Sparse row-graph configuration.
# Node la moi dong rating; edge la quan he thua giua cac dong doanh nghiep-thoi diem.
# Edge khong dung label: kNN theo financial features trong cung sector + temporal ticker edges + self-loops.

LAST_Y_EMB_DIM = 8
SECTOR_EMB_DIM = 8
SPARSE_GRAPH_HIDDEN = 96
SPARSE_GRAPH_LAYERS = 3
SPARSE_GRAPH_KNN_K = 12
SPARSE_GRAPH_MIN_SIMILARITY = 0.05
SPARSE_GRAPH_TEMPORAL_WEIGHT = 1.25
SPARSE_GRAPH_SELF_LOOP_WEIGHT = 1.0
PERSISTENCE_PRIOR_SCALE = 2.0


def _add_sparse_edge(edge_store, dst, src, weight, edge_type):
    key = (int(dst), int(src))
    payload = edge_store.setdefault(key, {'raw_weight': 0.0, 'edge_types': set()})
    payload['raw_weight'] = max(float(payload['raw_weight']), float(weight))
    payload['edge_types'].add(str(edge_type))


def build_sparse_credit_graph(frame, feature_cols, knn_k=12, min_similarity=0.05):
    frame = frame.reset_index(drop=True).copy()
    x_np = frame[feature_cols].to_numpy(dtype=np.float32)
    n_nodes = len(frame)
    edge_store = {}

    for node_idx in range(n_nodes):
        _add_sparse_edge(
            edge_store,
            node_idx,
            node_idx,
            SPARSE_GRAPH_SELF_LOOP_WEIGHT,
            'self_loop',
        )

    for _, sector_index in frame.groupby('sector_id', sort=False).groups.items():
        sector_nodes = np.asarray(list(sector_index), dtype=int)
        if sector_nodes.size <= 1:
            continue
        n_neighbors = min(int(knn_k) + 1, int(sector_nodes.size))
        nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        sector_features = x_np[sector_nodes]
        nn_index.fit(sector_features)
        distances, neighbors = nn_index.kneighbors(sector_features, return_distance=True)
        for local_dst, (dist_row, neigh_row) in enumerate(zip(distances, neighbors)):
            dst = int(sector_nodes[local_dst])
            for distance, local_src in zip(dist_row[1:], neigh_row[1:]):
                src = int(sector_nodes[int(local_src)])
                similarity = 1.0 - float(distance)
                weight = max(float(min_similarity), similarity)
                _add_sparse_edge(edge_store, dst, src, weight, 'sector_feature_knn')
                _add_sparse_edge(edge_store, src, dst, weight, 'sector_feature_knn')

    temporal_sort_cols = ['rating_date', '__split__', '__split_row_index__']
    for _, ticker_rows in frame.groupby('ticker', sort=False):
        ordered = ticker_rows.sort_values(temporal_sort_cols, kind='mergesort')
        row_indices = ordered.index.to_numpy(dtype=int)
        if row_indices.size <= 1:
            continue
        for prev_idx, next_idx in zip(row_indices[:-1], row_indices[1:]):
            _add_sparse_edge(edge_store, next_idx, prev_idx, SPARSE_GRAPH_TEMPORAL_WEIGHT, 'ticker_temporal_prev')
            _add_sparse_edge(edge_store, prev_idx, next_idx, SPARSE_GRAPH_TEMPORAL_WEIGHT, 'ticker_temporal_next')

    rows = []
    for (dst, src), payload in edge_store.items():
        rows.append({
            'dst': int(dst),
            'src': int(src),
            'dst_row_id': str(frame.loc[dst, 'row_id']),
            'src_row_id': str(frame.loc[src, 'row_id']),
            'dst_split': str(frame.loc[dst, '__split__']),
            'src_split': str(frame.loc[src, '__split__']),
            'dst_ticker': str(frame.loc[dst, 'ticker']),
            'src_ticker': str(frame.loc[src, 'ticker']),
            'edge_type': '+'.join(sorted(payload['edge_types'])),
            'raw_weight': float(payload['raw_weight']),
        })
    edge_df = pd.DataFrame(rows)
    edge_df = edge_df.sort_values(['dst', 'src']).reset_index(drop=True)

    dst = edge_df['dst'].to_numpy(dtype=np.int64)
    src = edge_df['src'].to_numpy(dtype=np.int64)
    raw_weight = edge_df['raw_weight'].to_numpy(dtype=np.float32)
    row_sum = np.bincount(dst, weights=raw_weight, minlength=n_nodes).astype(np.float32)
    norm_weight = raw_weight / np.maximum(row_sum[dst], 1e-12)
    edge_df['norm_weight'] = norm_weight.astype(float)

    edge_index = torch.tensor(np.vstack([dst, src]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(norm_weight, dtype=torch.float32, device=device)
    return edge_index, edge_weight, edge_df


edge_index, edge_weight, edge_df = build_sparse_credit_graph(
    df,
    MODEL_FEATURES,
    knn_k=SPARSE_GRAPH_KNN_K,
    min_similarity=SPARSE_GRAPH_MIN_SIMILARITY,
)

sparse_graph_contract = {
    'baseline': 'Sparse row graph / pure PyTorch GraphSAGE-style message passing',
    'node_unit': 'one rating row/company observation',
    'node_count': int(len(df)),
    'edge_count': int(edge_index.size(1)),
    'edge_sources': {
        'sector_feature_knn': int(edge_df['edge_type'].str.contains('sector_feature_knn').sum()),
        'ticker_temporal': int(edge_df['edge_type'].str.contains('ticker_temporal').sum()),
        'self_loop': int(edge_df['edge_type'].str.contains('self_loop').sum()),
    },
    'knn_scope': 'within sector, features only, no target labels',
    'knn_k': SPARSE_GRAPH_KNN_K,
    'min_similarity_weight': SPARSE_GRAPH_MIN_SIMILARITY,
    'temporal_weight': SPARSE_GRAPH_TEMPORAL_WEIGHT,
    'normalization': 'row-normalized incoming weights: adj[dst, src]',
    'message_passing': 'torch.sparse.mm(adj, node_embeddings)',
    'hidden_dim': SPARSE_GRAPH_HIDDEN,
    'num_layers': SPARSE_GRAPH_LAYERS,
    'persistence_prior_scale': PERSISTENCE_PRIOR_SCALE,
}
print('Sparse graph contract:', sparse_graph_contract)
print('Sparse graph edge sample:')
display(edge_df.head(10))


# --- CELL BOUNDARY ---

class SparseGraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.20):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.neigh_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_lin.weight)
        nn.init.xavier_uniform_(self.neigh_lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, h, sparse_adj):
        # sparse_adj[dst, src] da row-normalized; sparse.mm gom thong tin lang gieng vao node dst.
        neigh = torch.sparse.mm(sparse_adj, h)
        return self.self_lin(h) + self.neigh_lin(neigh) + self.bias


class SparseCreditGraphBaseline(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        n_sectors,
        n_nodes,
        edge_index,
        edge_weight,
        hidden=96,
        num_layers=3,
        dropout=0.25,
        last_y_emb_dim=8,
        sector_emb_dim=8,
        context_dropout=0.15,
        persistence_prior_scale=0.0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.context_dropout = float(context_dropout)
        self.context_unknown_id = int(n_classes)
        self.persistence_prior_scale = float(persistence_prior_scale)
        self.last_y_emb_dim = int(last_y_emb_dim)
        self.last_y_emb = nn.Embedding(n_classes + 1, last_y_emb_dim)
        self.sector_emb = nn.Embedding(n_sectors, sector_emb_dim)
        in_dim = int(n_features) + int(last_y_emb_dim) + int(sector_emb_dim)
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            SparseGraphSAGELayer(hidden, hidden, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(int(num_layers))])
        self.layer_dropout = nn.Dropout(dropout)
        readout_dim = hidden * (int(num_layers) + 1)
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.register_buffer('edge_index', edge_index.detach().long().clone())
        self.register_buffer('edge_weight', edge_weight.detach().float().clone())

    def build_sparse_adj(self, edge_index=None, edge_weight=None):
        edge_index = self.edge_index if edge_index is None else edge_index.long()
        edge_weight = self.edge_weight if edge_weight is None else edge_weight.float()
        return torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            size=(self.n_nodes, self.n_nodes),
            device=edge_weight.device,
        ).coalesce()

    def encode_context(self, x, last_y, sector_id, context_mask_prob=0.0):
        if self.training and context_mask_prob > 0.0:
            mask = torch.rand(last_y.shape, device=last_y.device) < float(context_mask_prob)
            last_y = last_y.masked_fill(mask, self.context_unknown_id)
        last_y_emb = self.last_y_emb(last_y)
        if self.training and self.context_dropout > 0:
            last_y_emb = F.dropout(last_y_emb, p=self.context_dropout, training=True)
        h = torch.cat([x, last_y_emb, self.sector_emb(sector_id)], dim=1)
        h = self.input_norm(h)
        return self.input_proj(h)

    def forward(self, x, last_y, sector_id, edge_index=None, edge_weight=None, return_embeddings=False, context_mask_prob=0.0):
        if x.size(0) != self.n_nodes:
            raise ValueError('SparseCreditGraphBaseline yeu cau full-batch x_all dung so node da build graph.')
        sparse_adj = self.build_sparse_adj(edge_index=edge_index, edge_weight=edge_weight)
        h = self.encode_context(x, last_y, sector_id, context_mask_prob=context_mask_prob)
        states = [h]
        for layer, norm in zip(self.layers, self.norms):
            h_next = F.gelu(layer(h, sparse_adj))
            h = norm(h + h_next)
            h = self.layer_dropout(h)
            states.append(h)
        readout = torch.cat(states, dim=1)
        logits = self.head(readout)
        if self.persistence_prior_scale > 0:
            valid_last_y = last_y.clamp(min=0, max=logits.size(1) - 1)
            prior = torch.zeros_like(logits)
            prior.scatter_(1, valid_last_y.view(-1, 1), self.persistence_prior_scale)
            logits = logits + prior
        if return_embeddings:
            return logits, readout
        return logits


CreditGAT = SparseCreditGraphBaseline




    n_features=len(MODEL_FEATURES),
    n_classes=n_classes,
    n_sectors=n_sectors,
    n_nodes=x_all.size(0),
    edge_index=edge_index,
    edge_weight=edge_weight,
    hidden=SPARSE_GRAPH_HIDDEN,
    num_layers=SPARSE_GRAPH_LAYERS,
    dropout=0.28,
    last_y_emb_dim=LAST_Y_EMB_DIM,
    sector_emb_dim=SECTOR_EMB_DIM,
    context_dropout=0.16,
    persistence_prior_scale=PERSISTENCE_PRIOR_SCALE,
).to(device)

    protocol=LOSS_PROTOCOL,
    ordinal_lambda=ORDINAL_LAMBDA,
).to(device)
    'lr': 1.0e-3,
    'weight_decay': 5e-5,
    'plateau_factor': 0.60,
    'plateau_patience': 8,
    'min_lr': 1e-5,
}
    model.parameters(),
    lr=OPTIMIZER_CONFIG['lr'],
    weight_decay=OPTIMIZER_CONFIG['weight_decay'],
)
    optimizer,
    mode='min',
    factor=OPTIMIZER_CONFIG['plateau_factor'],
    patience=OPTIMIZER_CONFIG['plateau_patience'],
    min_lr=OPTIMIZER_CONFIG['min_lr'],
)


def summarize_sparse_graph(edge_df):
    n_nodes = int(len(df))
    non_self = edge_df[edge_df['dst'] != edge_df['src']]
    by_type = edge_df['edge_type'].str.get_dummies(sep='+').sum().sort_values(ascending=False)
    return {
        'nodes': n_nodes,
        'edges': int(len(edge_df)),
        'non_self_edges': int(len(non_self)),
        'avg_in_degree_non_self': round(float(len(non_self) / max(1, n_nodes)), 4),
        'edge_density': round(float(len(non_self) / max(1, n_nodes * max(1, n_nodes - 1))), 8),
        'edge_types': by_type.astype(int).to_dict(),
    }


print('Initial sparse graph stats:', summarize_sparse_graph(edge_df))

# --- CELL BOUNDARY ---

# Visualization: sparse row graph around one validation company-row.
# Yeu cau: chay cell graph va cell model truoc.
if 'model' not in globals() or 'edge_df' not in globals():
    raise RuntimeError('Khong tim thay model hoac edge_df. Hay chay cell graph va cell model truoc.')

import networkx as nx
from matplotlib.colors import Normalize, TwoSlopeNorm

sample_candidates = np.flatnonzero(df['__split__'].eq('val').values)
sample_idx = int(sample_candidates[0]) if len(sample_candidates) else 0

incoming = edge_df[(edge_df['dst'] == sample_idx) & (edge_df['src'] != sample_idx)].copy()
outgoing = edge_df[(edge_df['src'] == sample_idx) & (edge_df['dst'] != sample_idx)].copy()
neighbor_edges = pd.concat([incoming, outgoing], ignore_index=True).drop_duplicates(['dst', 'src'])
neighbor_edges = neighbor_edges.sort_values('norm_weight', ascending=False).head(40).reset_index(drop=True)
display(neighbor_edges[[
    'dst', 'src', 'dst_row_id', 'src_row_id', 'dst_ticker', 'src_ticker',
    'edge_type', 'raw_weight', 'norm_weight',
]].head(20))

G = nx.DiGraph()
node_ids = set([sample_idx])
for row in neighbor_edges.itertuples(index=False):
    node_ids.add(int(row.dst))
    node_ids.add(int(row.src))

for node in sorted(node_ids):
    split = str(df.loc[node, '__split__'])
    ticker = str(df.loc[node, 'ticker'])
    label = f'{ticker}\n{split}:{df.loc[node, "row_id"]}'
    G.add_node(node, label=label, split=split, target=int(df.loc[node, TARGET_COL]))
for row in neighbor_edges.itertuples(index=False):
    G.add_edge(int(row.src), int(row.dst), weight=float(row.norm_weight), edge_type=str(row.edge_type))

fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
pos = nx.spring_layout(G, seed=SEED, k=0.65, iterations=120)
node_values = [G.nodes[n]['target'] for n in G.nodes]
if min(node_values) < max(node_values):
    color_norm = Normalize(vmin=min(node_values), vmax=max(node_values))
else:
    color_norm = Normalize(vmin=0, vmax=max(1, n_classes - 1))
node_colors = plt.get_cmap('viridis')(color_norm(node_values))
node_sizes = [700 if n == sample_idx else 260 + 30 * G.degree(n) for n in G.nodes]
edge_widths = [0.6 + 4.0 * G.edges[e]['weight'] for e in G.edges]
nx.draw_networkx_edges(G, pos, alpha=0.28, width=edge_widths, edge_color='#555555', arrows=True, arrowsize=8, ax=ax)
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    linewidths=[1.8 if n == sample_idx else 0.6 for n in G.nodes],
    edgecolors=['#d62728' if n == sample_idx else '#222222' for n in G.nodes],
    ax=ax,
)
labels = {n: G.nodes[n]['label'] for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=6.5, ax=ax)
colorbar = plt.cm.ScalarMappable(cmap='viridis', norm=color_norm)
colorbar.set_array([])
fig.colorbar(colorbar, ax=ax, shrink=0.72, label='Rating label id')
ax.set_title(f'Sparse Row Graph Ego Network | row_id={df.loc[sample_idx, "row_id"]}')
ax.axis('off')
fig.tight_layout()
graph_path = ARTIFACT_DIR / 'gat_graph_visualization.png'
fig.savefig(graph_path, dpi=300, bbox_inches='tight')
plt.show()

print('Ego nodes:', G.number_of_nodes(), '| ego edges:', G.number_of_edges())
print('Saved:', graph_path)
