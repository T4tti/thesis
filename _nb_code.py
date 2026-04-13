# === Cell 3 ===
import platform
import sys
from pathlib import Path

IN_KAGGLE = Path('/kaggle').exists()
WORKING_DIR = Path('/kaggle/working') if IN_KAGGLE else Path('.')
ARTIFACT_DIR = WORKING_DIR / 'credit_rating_artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print('Python:', platform.python_version())
print('Running on Kaggle:', IN_KAGGLE)
print('Working directory:', WORKING_DIR.resolve())
print('Artifact directory:', ARTIFACT_DIR.resolve())

# === Cell 4 ===
import random
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, label_binarize, RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# === Cell 6 ===
def resolve_split_path(default_path, local_fallbacks=None):
    """Resolve Kaggle split path first, then local fallbacks."""
    candidates = [Path(default_path)]
    if local_fallbacks:
        candidates.extend([Path(p) for p in local_fallbacks])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f'Cannot find data file. Tried: {[str(c) for c in candidates]}'
)

TRAIN_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/train_smote_augmented.csv',
    local_fallbacks=[
        'archive/ctgan/splits/train_augmented_ctgan.csv',
        'data/processed/ctgan/splits/train_augmented_ctgan.csv'
    ]
)
VAL_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/val_split.csv',
    local_fallbacks=[
        'archive/ctgan/splits/val.csv',
        'data/processed/ctgan/splits/val.csv'
    ]
)
TEST_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/test_split.csv',
    local_fallbacks=[
        'archive/ctgan/splits/test.csv',
        'data/processed/ctgan/splits/test.csv'
    ]
)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df['__split__'] = 'train'
val_df['__split__'] = 'val'
test_df['__split__'] = 'test'

df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print('Train path:', TRAIN_PATH)
print('Val path:  ', VAL_PATH)
print('Test path: ', TEST_PATH)
print('Train shape:', train_df.shape)
print('Val shape:  ', val_df.shape)
print('Test shape: ', test_df.shape)
print('Combined shape:', df.shape)
display(df.head())
df.info()

# === Cell 7 ===
# 1. Vẽ 2 biểu đồ biểu diễn Bar charts (1 hàng, 2 cột)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Đồ thị 1: Top 20 Target Classes
target_counts = df['rating_detail'].value_counts().sort_values(ascending=False)
axes[0].bar(target_counts.index.astype(str), target_counts.values, color='#1b9e77')
axes[0].set_title('Top 20 Target Classes (rating_detail)')
axes[0].set_xlabel('rating_detail')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=75)

# Đồ thị 2: Top 15 Missing Value (%)
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
top_missing = missing_pct.head(15)
axes[1].barh(top_missing.index[::-1], top_missing.values[::-1], color='#7570b3')
axes[1].set_title('Top 15 Missing Value (%)')
axes[1].set_xlabel('% Missing')

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'eda_bars.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# 2. Tách riêng biểu đồ Correlation Heatmap
# Lựa chọn 20 features số thực (loại bỏ binary_rating) để quan sát chi tiết nhất vòng Tương quan
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols_corr = [c for c in num_cols_all if c not in ['binary_rating']][:20] 
corr = df[num_cols_corr].corr()

plt.figure(figsize=(14, 10)) # Định cấu hình ảnh to, dễ review
sns.heatmap(
    corr, 
    cmap='coolwarm', 
    center=0, 
    annot=True,         # Bật chi tiết số liệu correlation trên từng ô
    fmt=".2f",          # Làm tròn 2 chữ số thập phân
    linewidths=0.5,     # Tạo đường lưới mỏng nét cho các ô
    annot_kws={"size": 8} # Thu nhỏ font size số để tránh bị đè nếu chữ quá dài
)
plt.title('Detailed Correlation Heatmap (Top Numeric Features)', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(ARTIFACT_DIR / 'eda_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print('EDA plots (bars and heatmap) saved successfully.')


# === Cell 9 ===
FINANCIAL_FEATURES = [
    'current_ratio', 'debt_equity_ratio',
    'gross_profit_margin', 'operating_profit_margin',
    'ebit_margin', 'pretax_profit_margin',
    'net_profit_margin', 'asset_turnover',
    'roe', 'roa',
    'operating_cashflow_ps', 'free_cashflow_ps'
]

# rating_detail is already mapped to numeric labels in [0, 22].
TARGET_MIN, TARGET_MAX = 0, 21
EXPECTED_CLASSES = list(range(TARGET_MIN, TARGET_MAX + 1))

df['rating_detail'] = pd.to_numeric(df['rating_detail'], errors='coerce')
before_rows = len(df)
df = df.dropna(subset=['rating_detail']).copy()
df['rating_detail'] = df['rating_detail'].astype(int)
in_range_mask = df['rating_detail'].between(TARGET_MIN, TARGET_MAX)
dropped_out_of_range = int((~in_range_mask).sum())
if dropped_out_of_range > 0:
    df = df[in_range_mask].copy()

df['rating_numeric'] = df['rating_detail']
n_classes = len(EXPECTED_CLASSES)
rating_mapping = {i: i for i in EXPECTED_CLASSES}

# Keep a simple encoder-like object for downstream metadata compatibility.
class _StaticLabelEncoder:
    pass
le = _StaticLabelEncoder()
le.classes_ = np.array(EXPECTED_CLASSES, dtype=int)

print(f'Rows kept after target cleanup: {len(df)} / {before_rows}')
print(f'Dropped out-of-range labels: {dropped_out_of_range}')
print(f'Number of classes (fixed): {n_classes}')
print(f'Label range: {TARGET_MIN}..{TARGET_MAX}')
print('Observed labels in data:', sorted(df['rating_numeric'].unique().tolist())[:22], '...')

df['rating_date'] = pd.to_datetime(df['rating_date'], format='mixed')


for col in FINANCIAL_FEATURES:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in FINANCIAL_FEATURES:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

print(f'\nData after preprocessing: {df.shape}')

# === Cell 11 ===
df_sorted = df.sort_values(['ticker', 'rating_date']).reset_index(drop=True)

panel_df = df_sorted[['ticker', 'rating_date', 'rating_numeric', '__split__'] + FINANCIAL_FEATURES].copy()
panel_df = panel_df.rename(columns={
    'ticker': 'unique_id',
    'rating_date': 'ds',
    'rating_numeric': 'y'
})

ticker_counts = panel_df.groupby('unique_id').size().reset_index(name='count')
print('Ticker count statistics:')
print(ticker_counts['count'].describe())
print()

MIN_HISTORY = 3
valid_tickers = ticker_counts[ticker_counts['count'] >= MIN_HISTORY]['unique_id'].tolist()
panel_df = panel_df[panel_df['unique_id'].isin(valid_tickers)].reset_index(drop=True)

print(f'Tickers with >= {MIN_HISTORY} data points: {len(valid_tickers)}')
print(f'Panel DataFrame shape: {panel_df.shape}')
print(f'Unique tickers: {panel_df["unique_id"].nunique()}')
print(f'Date range: {panel_df["ds"].min()} to {panel_df["ds"].max()}')
print('Split distribution:')
print(panel_df['__split__'].value_counts())
display(panel_df.head(10))

# === Cell 13 ===
median_seq_len = int(ticker_counts[ticker_counts['count'] >= MIN_HISTORY]['count'].median())
INPUT_SIZE = max(2, min(median_seq_len - 1, 24))
HORIZON = 1

print(f'Median sequence length: {median_seq_len}')
print(f'INPUT_SIZE (input window): {INPUT_SIZE}')
print(f'HORIZON (forecast): {HORIZON}')

min_required = INPUT_SIZE + HORIZON
valid_tickers2 = ticker_counts[ticker_counts['count'] >= min_required]['unique_id'].tolist()
panel_df_filtered = panel_df[panel_df['unique_id'].isin(valid_tickers2)].reset_index(drop=True)

print(f'\nTickers with >= {min_required} data points: {len(valid_tickers2)}')
print(f'Filtered panel shape: {panel_df_filtered.shape}')

# -- Scale financial features with train split only, then transform all --
scaler = RobustScaler()
train_feature_mask = panel_df_filtered['__split__'] == 'train'
if train_feature_mask.any():
    scaler.fit(panel_df_filtered.loc[train_feature_mask, FINANCIAL_FEATURES].values)
else:
    scaler.fit(panel_df_filtered[FINANCIAL_FEATURES].values)
panel_df_filtered[FINANCIAL_FEATURES] = scaler.transform(
    panel_df_filtered[FINANCIAL_FEATURES].values
)

# -- Build sliding window dataset --
class CreditRatingDataset(Dataset):
    """Sliding window dataset.
    Returns X shape (INPUT_SIZE, n_channels) and last_y context."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, last_y, y_target = self.sequences[idx]
        return (
            torch.FloatTensor(X),
            torch.LongTensor([last_y])[0],
            torch.LongTensor([y_target])[0],
        )

feature_cols = FINANCIAL_FEATURES  # 12 financial channels (y removed)
n_channels = len(feature_cols)

train_seqs = []
val_seqs = []
test_seqs = []

for uid, grp in panel_df_filtered.groupby('unique_id'):
    grp = grp.sort_values('ds').reset_index(drop=True)
    values = grp[feature_cols].values  # (T, C)
    n = len(values)
    if n < INPUT_SIZE + HORIZON:
        continue

    for i in range(n - INPUT_SIZE - HORIZON + 1):
        X = values[i : i + INPUT_SIZE]
        last_y = int(grp['y'].iloc[i + INPUT_SIZE - 1])
        target_idx = i + INPUT_SIZE
        y_target = int(grp['y'].iloc[target_idx])  # next-step class
        split_label = str(grp['__split__'].iloc[target_idx]).lower()

        sample = (X, last_y, y_target)
        if split_label == 'test':
            test_seqs.append(sample)
        elif split_label == 'val':
            val_seqs.append(sample)
        else:
            train_seqs.append(sample)

train_tickers = set(
    panel_df_filtered.loc[panel_df_filtered['__split__'] == 'train', 'unique_id'].tolist()
)
val_tickers = set(
    panel_df_filtered.loc[panel_df_filtered['__split__'] == 'val', 'unique_id'].tolist()
)

if len(train_seqs) == 0 or len(val_seqs) == 0 or len(test_seqs) == 0:
    raise ValueError(
        f'Empty split sequences found: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}. '
        'Please check split files and time-order windows.'
    )

train_ds = CreditRatingDataset(train_seqs)
val_ds = CreditRatingDataset(val_seqs)
test_ds = CreditRatingDataset(test_seqs)

# Handle class imbalance on training windows.
# CTGAN train split is usually more balanced than val/test; enable weighted sampling
# only when imbalance is meaningful to avoid over-noising minority classes.
train_labels = np.array([s[2] for s in train_seqs], dtype=int)
class_freq_raw = np.bincount(train_labels, minlength=n_classes).astype(float)
class_freq_safe = np.maximum(class_freq_raw, 1.0)
non_zero_freq = class_freq_raw[class_freq_raw > 0]
imbalance_ratio = (
    float(class_freq_safe.max() / non_zero_freq.min())
    if len(non_zero_freq) > 0 else 1.0
)
USE_WEIGHTED_SAMPLER = imbalance_ratio >= 2.0

if USE_WEIGHTED_SAMPLER:
    sample_weights = (1.0 / class_freq_safe[train_labels]).astype(np.float64)
    sample_weights = sample_weights / sample_weights.mean()
    train_sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )
else:
    sample_weights = np.ones(len(train_labels), dtype=np.float64)
    train_sampler = None

BATCH_SIZE = 32
NUM_WORKERS = 4 if IN_KAGGLE else 0
PIN_MEMORY = torch.cuda.is_available()

if train_sampler is not None:
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
else:
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

print(f'\nChannels: {n_channels} ({feature_cols})')
print(f'Train tickers: {len(train_tickers)} | Val tickers: {len(val_tickers)}')
print(f'Train samples: {len(train_ds)}')
print(f'Val samples:   {len(val_ds)}')
print(f'Test samples:  {len(test_ds)}')
print(f'Train class freq min/max: {class_freq_raw.min():.0f}/{class_freq_raw.max():.0f}')
print(f'Imbalance ratio (max/min non-zero): {imbalance_ratio:.2f}')
print(f'Weighted sampler enabled: {USE_WEIGHTED_SAMPLER}')
print(f'Sampler weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]')

# Verify shape
sample_X, sample_last_y, sample_y = train_ds[0]
print(f'\nSample X shape: {sample_X.shape}  (T, C)')
print(f'Sample last_y: {sample_last_y}')
print(f'Sample y: {sample_y}')

# === Cell 15 ===
class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.
    Addresses class imbalance by down-weighting easy examples."""
    def __init__(self, gamma=2.0, weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def build_effective_num_weights(class_counts, beta=0.995):
    """Class-Balanced weighting from effective number of samples."""
    class_counts = np.asarray(class_counts, dtype=np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.mean()
    return weights


# Compute class weights using y_target (next-step class)
all_labels = [s[2] for s in train_seqs]
class_counts = np.bincount(all_labels, minlength=n_classes).astype(float)
class_counts = np.maximum(class_counts, 1.0)

# Effective-number weighting is more stable than pure inverse-freq on long tails.
class_weights = build_effective_num_weights(class_counts, beta=0.995)

# Clip tails for stability and avoid extreme gradients.
lower_cap = np.percentile(class_weights, 5)
upper_cap = np.percentile(class_weights, 95)
class_weights = np.clip(class_weights, lower_cap, upper_cap)
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

criterion_settings = {
    'loss_name': 'focal',         # {'focal', 'ce'}
    'focal_gamma': 2.0,
    'label_smoothing': 0.03,
    'use_class_weights': True,
}

loss_weight_tensor = class_weights_tensor if criterion_settings['use_class_weights'] else None

if criterion_settings['loss_name'] == 'ce':
    criterion = nn.CrossEntropyLoss(
        weight=loss_weight_tensor,
        label_smoothing=criterion_settings['label_smoothing'],
    )
else:
    criterion = FocalLoss(
        gamma=criterion_settings['focal_gamma'],
        weight=loss_weight_tensor,
        label_smoothing=criterion_settings['label_smoothing'],
    )

FOCAL_GAMMA = criterion_settings['focal_gamma']
print(f"Loss: {criterion_settings['loss_name']} | gamma={criterion_settings['focal_gamma']} | smoothing={criterion_settings['label_smoothing']}")
print(f'Class weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]')

# === Cell 17 ===
class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, n_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, n_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, n_channels, 1))

    def forward(self, x, mode='norm'):
        # x: (B, C, T)
        if mode == 'norm':
            self._mean = x.mean(dim=-1, keepdim=True)
            self._std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self._std + self._mean
        return x


class FuzzyLayer(nn.Module):
    """Gaussian fuzzy membership expansion with 2 MFs per feature."""
    def __init__(self, input_features, n_mfs=2, init_sigma=1.0):
        super().__init__()
        self.input_features = input_features
        self.n_mfs = n_mfs
        self.centers = nn.Parameter(torch.linspace(-1.0, 1.0, n_mfs).repeat(input_features, 1))
        self.log_sigma = nn.Parameter(torch.full((input_features, n_mfs), math.log(init_sigma)))

    def forward(self, x):
        # x: (B, T, F)
        x_exp = x.unsqueeze(-1)  # (B, T, F, 1)
        centers = self.centers.unsqueeze(0).unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0).unsqueeze(0).clamp(min=1e-3)
        memberships = torch.exp(-0.5 * ((x_exp - centers) / sigma) ** 2)
        return memberships.reshape(x.shape[0], x.shape[1], self.input_features * self.n_mfs)


class TransformerEncoderStep(nn.Module):
    """Single self-attention block from T-LSTM formulation."""
    def __init__(self, d_model=24, dropout=0.4):
        super().__init__()
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # x: (B, T, d_model)
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
        S = torch.softmax(scores, dim=-1)
        x_tilde = torch.bmm(S, V)
        x_add = x + x_tilde
        x_hat = torch.tanh(x_add + self.mlp(x_add))
        return x_hat


class TLSTMFuzzyClassifier(nn.Module):
    """Kaggle-optimized Transformer-LSTM + Fuzzy classifier."""
    def __init__(self, n_channels, n_classes, hidden_size=48, dropout=0.4, n_mfs=2):
        super().__init__()
        self.revin = RevIN(n_channels, affine=True)
        self.fuzzy = FuzzyLayer(n_channels, n_mfs=n_mfs)
        self.d_model = n_channels * n_mfs
        self.transformer_layer = TransformerEncoderStep(d_model=self.d_model, dropout=dropout)
        self.lstm = nn.LSTM(self.d_model, hidden_size, batch_first=True)
        self.last_y_embed = nn.Embedding(n_classes, hidden_size)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden_size, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, last_y):
        # x: (B, T, C)
        x_norm = self.revin(x.permute(0, 2, 1), mode='norm').permute(0, 2, 1)
        x_fuzzy = self.fuzzy(x_norm)
        x_hat = self.transformer_layer(x_fuzzy)
        _, (hn, _) = self.lstm(x_hat)
        z_t = hn[-1]
        last_y_emb = self.last_y_embed(last_y)
        out = torch.cat([z_t, last_y_emb], dim=-1)
        logits = self.head(out)
        return logits


FUZZY_MFS = 4
LSTM_HIDDEN = 64
TLSTM_DROPOUT = 0.4

model = TLSTMFuzzyClassifier(
    n_channels=n_channels,
    n_classes=n_classes,
    hidden_size=LSTM_HIDDEN,
    dropout=TLSTM_DROPOUT,
    n_mfs=FUZZY_MFS,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Fuzzy MFs per feature: {FUZZY_MFS}')
print(f'Transformer d_model after fuzzy expansion: {n_channels * FUZZY_MFS}')
print(f'LSTM hidden size: {LSTM_HIDDEN}')
print(f'Channels: {n_channels} (financial features)')
print(f'Model parameters: {n_params:,}')
print('\nTLSTM-Fuzzy model created successfully!')
print(model)

# === Cell 18 ===
# ============================================================
# Lightweight Hyperparameter Search (val_f1_weighted)
# ============================================================
from torch.utils.data import Subset


def build_training_criterion(config, class_counts_full):
    cb_beta = float(config.get('cb_beta', 0.995))
    class_weights_local = build_effective_num_weights(class_counts_full, beta=cb_beta)
    low_cap = np.percentile(class_weights_local, 5)
    high_cap = np.percentile(class_weights_local, 95)
    class_weights_local = np.clip(class_weights_local, low_cap, high_cap)
    class_weights_local = class_weights_local / class_weights_local.mean()

    if config.get('use_class_weights', True):
        weight_tensor = torch.FloatTensor(class_weights_local).to(device)
    else:
        weight_tensor = None

    if config.get('loss_name', 'focal') == 'ce':
        return nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=float(config.get('label_smoothing', 0.0)),
        )

    return FocalLoss(
        gamma=float(config.get('focal_gamma', 2.0)),
        weight=weight_tensor,
        label_smoothing=float(config.get('label_smoothing', 0.0)),
    )


def run_quick_trial(config, train_subset_loader, val_subset_loader, n_epochs=6, patience=2):
    trial_model = TLSTMFuzzyClassifier(
        n_channels=n_channels,
        n_classes=n_classes,
        hidden_size=int(config['hidden_size']),
        dropout=float(config['dropout']),
        n_mfs=int(config['fuzzy_mfs']),
    ).to(device)

    trial_criterion = build_training_criterion(config, class_counts)
    trial_optimizer = torch.optim.AdamW(
        trial_model.parameters(),
        lr=float(config['lr']),
        weight_decay=float(config['weight_decay']),
    )

    steps_per_epoch = max(1, len(train_subset_loader))
    trial_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trial_optimizer,
        max_lr=float(config['max_lr']),
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=max(float(config['max_lr']) / float(config['lr']), 1.0),
        final_div_factor=100.0,
    )

    amp_enabled_local = torch.cuda.is_available()
    amp_device_local = 'cuda' if amp_enabled_local else 'cpu'
    scaler_local = torch.amp.GradScaler(amp_device_local, enabled=amp_enabled_local)

    best_f1w = -np.inf
    no_improve = 0

    for _ in range(n_epochs):
        trial_model.train()
        for X_batch, last_y_batch, y_batch in train_subset_loader:
            X_batch = X_batch.to(device)
            last_y_batch = last_y_batch.to(device)
            y_batch = y_batch.to(device)

            trial_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_local, enabled=amp_enabled_local):
                logits = trial_model(X_batch, last_y_batch)
                loss = trial_criterion(logits, y_batch)

            scaler_local.scale(loss).backward()
            scaler_local.unscale_(trial_optimizer)
            torch.nn.utils.clip_grad_norm_(trial_model.parameters(), 1.0)
            scaler_local.step(trial_optimizer)
            scaler_local.update()
            trial_scheduler.step()

        trial_model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for X_batch, last_y_batch, y_batch in val_subset_loader:
                X_batch = X_batch.to(device)
                last_y_batch = last_y_batch.to(device)
                logits = trial_model(X_batch, last_y_batch)
                pred = logits.argmax(dim=1).cpu().numpy()
                val_pred.extend(pred.tolist())
                val_true.extend(y_batch.numpy().tolist())

        f1w = f1_score(val_true, val_pred, average='weighted', zero_division=0)
        if f1w > best_f1w + 1e-4:
            best_f1w = f1w
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return float(best_f1w)


RUN_HYPERPARAM_SEARCH = True
TUNE_MAX_TRIALS = 8
TUNE_EPOCHS = 6
TUNE_PATIENCE = 2
TUNE_BATCH_SIZE = 64
TUNE_TRAIN_FRACTION = 0.6
TUNE_VAL_FRACTION = 1.0

base_config = {
    'hidden_size': 64,
    'dropout': 0.4,
    'fuzzy_mfs': 4,
    'lr': 3e-4,
    'max_lr': 8e-4,
    'weight_decay': 1e-3,
    'loss_name': 'focal',
    'focal_gamma': 2.0,
    'label_smoothing': 0.03,
    'use_class_weights': True,
    'cb_beta': 0.995,
}

search_space = {
    'hidden_size': [48, 64, 96],
    'dropout': [0.30, 0.40],
    'fuzzy_mfs': [3, 4],
    'lr': [2e-4, 3e-4, 5e-4],
    'max_lr': [6e-4, 8e-4, 1.2e-3],
    'weight_decay': [5e-4, 1e-3],
    'loss_name': ['focal', 'ce'],
    'focal_gamma': [1.5, 2.0],
    'label_smoothing': [0.0, 0.03],
    'use_class_weights': [True, False],
    'cb_beta': [0.99, 0.995],
}

if RUN_HYPERPARAM_SEARCH:
    train_size = len(train_ds)
    val_size = len(val_ds)
    n_train_sub = max(256, int(train_size * TUNE_TRAIN_FRACTION))
    n_val_sub = max(256, int(val_size * TUNE_VAL_FRACTION))

    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(train_size, size=min(train_size, n_train_sub), replace=False)
    val_idx = rng.choice(val_size, size=min(val_size, n_val_sub), replace=False)

    train_sub_ds = Subset(train_ds, train_idx.tolist())
    val_sub_ds = Subset(val_ds, val_idx.tolist())

    train_sub_loader = DataLoader(
        train_sub_ds,
        batch_size=TUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_sub_loader = DataLoader(
        val_sub_ds,
        batch_size=TUNE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_candidates = []
    for hs in search_space['hidden_size']:
        for dr in search_space['dropout']:
            for mfs in search_space['fuzzy_mfs']:
                for lr in search_space['lr']:
                    for max_lr in search_space['max_lr']:
                        if max_lr <= lr:
                            continue
                        for wd in search_space['weight_decay']:
                            for loss_name in search_space['loss_name']:
                                for gamma in search_space['focal_gamma']:
                                    for smooth in search_space['label_smoothing']:
                                        for use_w in search_space['use_class_weights']:
                                            for beta in search_space['cb_beta']:
                                                all_candidates.append({
                                                    'hidden_size': hs,
                                                    'dropout': dr,
                                                    'fuzzy_mfs': mfs,
                                                    'lr': lr,
                                                    'max_lr': max_lr,
                                                    'weight_decay': wd,
                                                    'loss_name': loss_name,
                                                    'focal_gamma': gamma,
                                                    'label_smoothing': smooth,
                                                    'use_class_weights': use_w,
                                                    'cb_beta': beta,
                                                })

    perm = rng.permutation(len(all_candidates))
    selected = [all_candidates[i] for i in perm[:min(TUNE_MAX_TRIALS, len(all_candidates))]]

    best_cfg = dict(base_config)
    best_score = -np.inf
    tune_results = []

    print(f'Running hyperparameter search on {len(selected)} trials...')
    print(f'Train subset: {len(train_sub_ds)} | Val subset: {len(val_sub_ds)}')

    for t, cfg in enumerate(selected, start=1):
        score = run_quick_trial(
            cfg,
            train_subset_loader=train_sub_loader,
            val_subset_loader=val_sub_loader,
            n_epochs=TUNE_EPOCHS,
            patience=TUNE_PATIENCE,
        )
        tune_results.append({'trial': t, 'score_f1_weighted': score, **cfg})
        print(
            f"Trial {t:02d}/{len(selected)} | f1w={score:.4f} | "
            f"hs={cfg['hidden_size']}, mfs={cfg['fuzzy_mfs']}, dr={cfg['dropout']}, "
            f"lr={cfg['lr']}, max_lr={cfg['max_lr']}, loss={cfg['loss_name']}"
        )
        if score > best_score:
            best_score = score
            best_cfg = dict(cfg)

    tuning_df = pd.DataFrame(tune_results).sort_values('score_f1_weighted', ascending=False)
    tuning_path = ARTIFACT_DIR / 'hyperparameter_search_results.csv'
    tuning_df.to_csv(tuning_path, index=False)

    BEST_TUNED_CONFIG = best_cfg
    print('\nBest tuned config:')
    print(BEST_TUNED_CONFIG)
    print(f'Best trial val_f1_weighted: {best_score:.4f}')
    print(f'Tuning report saved to: {tuning_path}')
else:
    BEST_TUNED_CONFIG = dict(base_config)
    print('Hyperparameter search skipped. Using base config.')
    print(BEST_TUNED_CONFIG)

# === Cell 20 ===
# ============================================================
# Training Loop with Full Metric Tracking (TLSTM-Fuzzy + AMP)
# ============================================================

FAST_DEV_RUN = False
MAX_EPOCHS = 100 if not FAST_DEV_RUN else 5
PATIENCE = 14
EARLY_STOP_MIN_DELTA = 1.5e-3

train_config = dict(globals().get('BEST_TUNED_CONFIG', {
    'hidden_size': 48,
    'dropout': 0.4,
    'fuzzy_mfs': 4,
    'lr': 2e-4,
    'max_lr': 6e-4,
    'weight_decay': 5e-4,
    'loss_name': 'ce',
    'focal_gamma': 1.5,
    'label_smoothing': 0.03,
    'use_class_weights': True,
    'cb_beta': 0.995,
}))

FUZZY_MFS = int(train_config['fuzzy_mfs'])
LSTM_HIDDEN = int(train_config['hidden_size'])
TLSTM_DROPOUT = float(train_config['dropout'])
LR = float(train_config['lr'])
MAX_LR = float(train_config['max_lr'])
WEIGHT_DECAY = float(train_config['weight_decay'])

model = TLSTMFuzzyClassifier(
    n_channels=n_channels,
    n_classes=n_classes,
    hidden_size=LSTM_HIDDEN,
    dropout=TLSTM_DROPOUT,
    n_mfs=FUZZY_MFS,
).to(device)
criterion = build_training_criterion(train_config, class_counts)
FOCAL_GAMMA = float(train_config.get('focal_gamma', 0.0))

BEST_MODEL_PATH = ARTIFACT_DIR / 'transformer_best_model.pt'
BEST_META_PATH = ARTIFACT_DIR / 'transformer_best_model_meta.pt'

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=MAX_EPOCHS,
    pct_start=0.15,
    anneal_strategy='cos',
    div_factor=max(MAX_LR / LR, 1.0),
    final_div_factor=100.0,
)

AMP_ENABLED = torch.cuda.is_available()
AMP_DEVICE = 'cuda' if AMP_ENABLED else 'cpu'
scaler_amp = torch.amp.GradScaler(AMP_DEVICE, enabled=AMP_ENABLED)
print(f'AMP enabled: {AMP_ENABLED}')

history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_f1': [], 'val_f1': [],
    'train_f1_weighted': [], 'val_f1_weighted': [],
    'train_auc': [], 'val_auc': [],
    'train_mae': [], 'val_mae': [],
    'lr': []
}


def compute_cls_metrics(y_true, y_pred_logits, n_cls):
    probs = torch.softmax(y_pred_logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    y_t = y_true.cpu().numpy()

    acc = accuracy_score(y_t, y_pred)
    f1_macro = f1_score(y_t, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_t, y_pred, average='weighted', zero_division=0)
    mae = float(np.mean(np.abs(y_pred.astype(float) - y_t.astype(float))))

    try:
        y_true_bin = label_binarize(y_t, classes=list(range(n_cls)))
        auc = roc_auc_score(y_true_bin, probs, average='weighted', multi_class='ovr')
    except Exception:
        auc = 0.5

    return acc, f1_macro, f1_weighted, auc, mae


best_val_f1_weighted = -np.inf
best_epoch = -1
patience_counter = 0
best_state = None

print(f'Using tuned config: {train_config}')
print(f'Training for max {MAX_EPOCHS} epochs (patience={PATIENCE})...')
print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
print(f'LR init: {LR} | Max LR: {MAX_LR} | Weight decay: {WEIGHT_DECAY} | EarlyStop metric: val_f1_weighted')
print(f'Best model path: {BEST_MODEL_PATH}\n')

for epoch in range(MAX_EPOCHS):
    # -- Train --
    model.train()
    epoch_loss = []
    all_yt, all_logits = [], []

    for X_batch, last_y_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=PIN_MEMORY)
        last_y_batch = last_y_batch.to(device, non_blocking=PIN_MEMORY)
        y_batch = y_batch.to(device, non_blocking=PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
            logits = model(X_batch, last_y_batch)
            loss = criterion(logits, y_batch)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        epoch_loss.append(loss.item())
        all_yt.append(y_batch.detach())
        all_logits.append(logits.detach())

    train_loss = np.mean(epoch_loss)
    all_yt = torch.cat(all_yt)
    all_logits = torch.cat(all_logits)
    tr_acc, tr_f1, tr_f1w, tr_auc, tr_mae = compute_cls_metrics(all_yt, all_logits, n_classes)

    # -- Validate --
    model.eval()
    val_losses, vl_yt, vl_logits = [], [], []
    with torch.no_grad():
        for X_batch, last_y_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=PIN_MEMORY)
            last_y_batch = last_y_batch.to(device, non_blocking=PIN_MEMORY)
            y_batch = y_batch.to(device, non_blocking=PIN_MEMORY)
            with torch.amp.autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
                logits = model(X_batch, last_y_batch)
                loss = criterion(logits, y_batch)
            val_losses.append(loss.item())
            vl_yt.append(y_batch)
            vl_logits.append(logits)

    val_loss = np.mean(val_losses)
    vl_yt = torch.cat(vl_yt)
    vl_logits = torch.cat(vl_logits)
    vl_acc, vl_f1, vl_f1w, vl_auc, vl_mae = compute_cls_metrics(vl_yt, vl_logits, n_classes)

    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)
    history['train_f1'].append(tr_f1)
    history['val_f1'].append(vl_f1)
    history['train_f1_weighted'].append(tr_f1w)
    history['val_f1_weighted'].append(vl_f1w)
    history['train_auc'].append(tr_auc)
    history['val_auc'].append(vl_auc)
    history['train_mae'].append(tr_mae)
    history['val_mae'].append(vl_mae)
    history['lr'].append(current_lr)

    # Early stopping + checkpoint based on weighted F1
    if vl_f1w > best_val_f1_weighted + EARLY_STOP_MIN_DELTA:
        best_val_f1_weighted = vl_f1w
        best_epoch = epoch + 1
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        torch.save(best_state, BEST_MODEL_PATH)
        torch.save({
            'best_epoch': best_epoch,
            'best_val_f1_weighted': float(best_val_f1_weighted),
            'n_classes': n_classes,
            'n_channels': n_channels,
            'input_size': INPUT_SIZE,
            'model_type': 'TLSTMFuzzyClassifier',
            'fuzzy_mfs': FUZZY_MFS,
            'd_model': n_channels * FUZZY_MFS,
            'lstm_hidden': LSTM_HIDDEN,
            'dropout': TLSTM_DROPOUT,
            'loss_name': train_config.get('loss_name', 'focal'),
            'focal_gamma': FOCAL_GAMMA,
            'label_smoothing': float(train_config.get('label_smoothing', 0.0)),
            'use_class_weights': bool(train_config.get('use_class_weights', True)),
            'cb_beta': float(train_config.get('cb_beta', 0.995)),
            'learning_rate': LR,
            'max_learning_rate': MAX_LR,
            'weight_decay': WEIGHT_DECAY,
            'amp_enabled': AMP_ENABLED,
            'label_encoder_classes': list(le.classes_),
            'financial_features': FINANCIAL_FEATURES,
            'scheduler': 'OneCycleLR',
            'tuned_config': train_config,
        }, BEST_META_PATH)
    else:
        patience_counter += 1

    print(
        f'Epoch {epoch+1:3d}/{MAX_EPOCHS} | '
        f'TrLoss: {train_loss:.4f} | VlLoss: {val_loss:.4f} | '
        f'TrAcc: {tr_acc:.3f} | VlAcc: {vl_acc:.3f} | '
        f'TrF1w: {tr_f1w:.3f} | VlF1w: {vl_f1w:.3f} | '
        f'VlAUC: {vl_auc:.3f} | LR: {current_lr:.6f}'
    )

    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

print(f'\nTraining completed! Best val F1-weighted: {best_val_f1_weighted:.4f} at epoch {best_epoch}')
print(f'Epochs trained: {len(history["train_loss"])}')
print(f'Best model saved to: {BEST_MODEL_PATH}')
print(f'Best metadata saved to: {BEST_META_PATH}')

# === Cell 22 ===
# ============================================================
# Prediction on Test Set + Quick Evaluation
# ============================================================

model.eval()
test_preds = []
test_trues = []
test_logits_all = []

with torch.no_grad():
    for X_batch, last_y_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        last_y_batch = last_y_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch, last_y_batch)
        preds = logits.argmax(dim=1).cpu().numpy()

        test_preds.extend(preds.tolist())
        test_trues.extend(y_batch.cpu().numpy().tolist())
        test_logits_all.append(logits.cpu())

test_logits_all = torch.cat(test_logits_all)
y_true = np.array(test_trues)
y_pred = np.array(test_preds)

print(f'Test samples: {len(y_true)}')
print(f'Unique true classes:  {len(set(y_true))}')
print(f'Unique pred classes:  {len(set(y_pred))}')

# === Cell 23 ===
# ============================================================
# Evaluation Metrics (final test)
# ============================================================

acc = accuracy_score(y_true, y_pred)
print(f'Accuracy: {acc:.4f}')

f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
print(f'F1 (macro):    {f1_macro:.4f}')
print(f'F1 (weighted): {f1_weighted:.4f}')

all_classes = sorted(set(y_true) | set(y_pred))
try:
    if len(all_classes) > 2:
        # Use softmax probabilities for better AUC
        probs = F.softmax(test_logits_all, dim=1).numpy()
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        auc_ovr = roc_auc_score(y_true_bin, probs, average='weighted', multi_class='ovr')
    else:
        auc_ovr = roc_auc_score(y_true, y_pred)
    print(f'AUC-ROC (OvR): {auc_ovr:.4f}')
except ValueError as e:
    auc_ovr = float('nan')
    print(f'AUC-ROC: Could not compute ({e})')

mae_raw = np.mean(np.abs(y_pred.astype(float) - y_true.astype(float)))
print(f'MAE (raw numeric): {mae_raw:.4f}')

print('\nClassification Report:')
report_labels = sorted(set(y_true) | set(y_pred))
report_target_names = [str(rating_mapping.get(i, i)) for i in report_labels]
print(classification_report(
    y_true, y_pred,
    labels=report_labels,
    target_names=report_target_names,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)

metrics_dict = {
    'accuracy': acc, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
    'auc_roc_ovr': auc_ovr, 'mae_raw': mae_raw,
    'n_samples': len(y_true), 'n_classes': n_classes,
}
pd.DataFrame([metrics_dict]).to_csv(ARTIFACT_DIR / 'transformer_metrics.csv', index=False)
print('\nMetrics saved to transformer_metrics.csv')

# === Cell 24 ===
# ============================================================
# Visualization: Metric Curves (train & val)
# ============================================================

for k, v in history.items():
    print(f'  {k}: {len(v)} data points')

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# 1) Loss
axes[0,0].plot(history['train_loss'], label='Train', color='#1f77b4')
axes[0,0].plot(history['val_loss'], label='Val', color='#ff7f0e')
axes[0,0].set_title('Loss (Focal)')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
axes[0,0].grid(alpha=0.3); axes[0,0].legend()

# 2) Accuracy
axes[0,1].plot(history['train_acc'], label='Train', color='#2ca02c')
axes[0,1].plot(history['val_acc'], label='Val', color='#d62728')
axes[0,1].set_title('Accuracy')
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Accuracy')
axes[0,1].grid(alpha=0.3); axes[0,1].legend()

# 3) F1
axes[0,2].plot(history['train_f1'], label='Train', color='#9467bd')
axes[0,2].plot(history['val_f1'], label='Val', color='#8c564b')
axes[0,2].set_title('F1-macro')
axes[0,2].set_xlabel('Epoch'); axes[0,2].set_ylabel('F1')
axes[0,2].grid(alpha=0.3); axes[0,2].legend()

# 4) AUC
axes[1,0].plot(history['train_auc'], label='Train', color='#17becf')
axes[1,0].plot(history['val_auc'], label='Val', color='#bcbd22')
axes[1,0].set_title('AUC (OVR)')
axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('AUC')
axes[1,0].grid(alpha=0.3); axes[1,0].legend()

# 5) MAE
axes[1,1].plot(history['train_mae'], label='Train', color='#e377c2')
axes[1,1].plot(history['val_mae'], label='Val', color='#7f7f7f')
axes[1,1].set_title('MAE')
axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('MAE')
axes[1,1].grid(alpha=0.3); axes[1,1].legend()

# 6) Learning Rate
axes[1,2].plot(history['lr'], color='#ff6600')
axes[1,2].set_title('Learning Rate')
axes[1,2].set_xlabel('Step'); axes[1,2].set_ylabel('LR')
axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'metric_curves.png', dpi=200, bbox_inches='tight')
plt.show()
print('Saved metric curves figure.')

# === Cell 25 ===
# ============================================================
# Visualization: Confusion Matrix + Distributions
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

all_labels = sorted(set(y_true) | set(y_pred))
label_names = [rating_mapping.get(i, str(i)) for i in all_labels]
cm_display = confusion_matrix(y_true, y_pred, labels=all_labels)

if len(all_labels) <= 25:
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
else:
    sns.heatmap(cm_display, cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].tick_params(axis='x', rotation=90)
axes[0].tick_params(axis='y', rotation=0)

true_counts = pd.Series(y_true).value_counts().sort_index()
pred_counts = pd.Series(y_pred).value_counts().sort_index()
all_idx = sorted(set(true_counts.index) | set(pred_counts.index))
x_pos = np.arange(len(all_idx))
width = 0.35
axes[1].bar(x_pos - width/2, [true_counts.get(i, 0) for i in all_idx],
            width, label='True', alpha=0.8, color='#2196F3')
axes[1].bar(x_pos + width/2, [pred_counts.get(i, 0) for i in all_idx],
            width, label='Predicted', alpha=0.8, color='#FF9800')
axes[1].set_title('True vs Predicted Distribution')
axes[1].set_xlabel('Rating Class')
axes[1].set_ylabel('Count')
axes[1].legend()
if len(all_idx) <= 25:
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([rating_mapping.get(i, str(i)) for i in all_idx],
                             rotation=90, fontsize=7)

errors = y_pred - y_true
axes[2].hist(errors, bins=max(10, len(set(errors))), color='#4CAF50',
             edgecolor='black', alpha=0.8)
axes[2].set_title('Prediction Error Distribution')
axes[2].set_xlabel('Error (Predicted - True)')
axes[2].set_ylabel('Frequency')
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[2].legend()

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'transformer_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('Result plots saved.')

# === Cell 26 ===
# Map test tickers back
test_ticker_list = []
for uid, grp in panel_df_filtered.groupby('unique_id'):
    grp = grp.sort_values('ds').reset_index(drop=True)
    if len(grp) >= INPUT_SIZE + HORIZON:
        test_ticker_list.append(uid)

results = pd.DataFrame({
    'unique_id': test_ticker_list[:len(y_true)],
    'y_true_numeric': y_true,
    'y_pred_numeric': y_pred,
})
results['y_true_label'] = results['y_true_numeric'].map(rating_mapping)
results['y_pred_label'] = results['y_pred_numeric'].map(rating_mapping)

print('\n=== Sample Predictions ===')
print(f'{"Ticker":<10} {"True Rating":<15} {"Predicted":>15} {"Match":>8}')
print('-' * 50)

sample_n = min(30, len(results))
for _, row in results.head(sample_n).iterrows():
    match = 'Y' if row['y_true_numeric'] == row['y_pred_numeric'] else 'N'
    print(f'{row["unique_id"]:<10} '
          f'{str(row["y_true_label"]):<15} '
          f'{str(row["y_pred_label"]):>15} '
          f'{match:>8}')

print(f'\n=== Summary ===')
print(f'Total predictions: {len(results)}')
print(f'Correct: {(results["y_true_numeric"] == results["y_pred_numeric"]).sum()}')
print(f'Accuracy: {acc:.4f}')
print(f'F1 (weighted): {f1_weighted:.4f}')
if not np.isnan(auc_ovr):
    print(f'AUC-ROC (OvR): {auc_ovr:.4f}')
else:
    print('AUC-ROC: N/A')
print(f'\nAll artifacts saved to: {ARTIFACT_DIR.resolve()}')

