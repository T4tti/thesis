import os
import platform
from pathlib import Path

def detect_kaggle_runtime() -> bool:
    """Detect Kaggle runtime robustly to avoid false positives on local Windows."""
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '').strip():
        return True
    return Path('/kaggle/input').exists() and Path('/kaggle/working').exists()

IN_KAGGLE = detect_kaggle_runtime()

def find_project_root(start: Path) -> Path:
    """Find repository root containing data/ and src/ when running locally."""
    for p in [start, *start.parents]:
        if (p / 'data').exists() and (p / 'src').exists():
            return p
    return start

CURRENT_DIR = Path.cwd().resolve()
PROJECT_ROOT = Path('/kaggle/working') if IN_KAGGLE else find_project_root(CURRENT_DIR)
WORKING_DIR = PROJECT_ROOT
ARTIFACT_DIR = WORKING_DIR / 'credit_rating_artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print('Python:', platform.python_version())
print('Running on Kaggle:', IN_KAGGLE)
print('Current directory:', CURRENT_DIR)
print('Project root:', PROJECT_ROOT)
print('Artifact directory:', ARTIFACT_DIR.resolve())

import random
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Training reproducibility knobs for more stable runs across executions.
DETERMINISTIC_TRAINING = True
if DETERMINISTIC_TRAINING:
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print('Deterministic training:', DETERMINISTIC_TRAINING)

def resolve_split_path(default_path, local_fallbacks=None):
    """Resolve Kaggle path first, then local paths relative to PROJECT_ROOT."""
    candidates = [Path(default_path)]
    if local_fallbacks:
        for p in local_fallbacks:
            p_obj = Path(p)
            if p_obj.is_absolute():
                candidates.append(p_obj)
            else:
                candidates.append(PROJECT_ROOT / p_obj)
                candidates.append(p_obj)

    seen = set()
    ordered_candidates = []
    for c in candidates:
        c_norm = str(c.resolve()) if c.exists() else str(c)
        if c_norm not in seen:
            seen.add(c_norm)
            ordered_candidates.append(c)

    for p in ordered_candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f'Cannot find data file. Tried: {[str(c) for c in ordered_candidates]}'
    )

TRAIN_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/train_augmented_timegan.csv',
    local_fallbacks=[
        'data/processed/train_augmented_timegan.csv',
        'archive/ctgan/splits/train_augmented_timegan.csv',
        'archive/ctgan/splits/train_augmented_ctgan.csv',
    ]
)
VAL_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/val.csv',
    local_fallbacks=[
        'data/processed/val.csv',
        'archive/ctgan/splits/val.csv',
        'data/processed/ctgan/splits/val.csv',
    ]
)
TEST_PATH = resolve_split_path(
    '/kaggle/input/datasets/tailength/corporate-credit-rating/test.csv',
    local_fallbacks=[
        'data/processed/test.csv',
        'archive/ctgan/splits/test.csv',
        'data/processed/ctgan/splits/test.csv',
    ]
)

ENFORCE_TICKER_DISJOINT = False

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

def ticker_set(frame):
    return set(frame['ticker'].astype(str).unique())

train_tickers_raw = ticker_set(train_df)
val_tickers_raw = ticker_set(val_df)
test_tickers_raw = ticker_set(test_df)

print('Ticker overlaps before optional disjoint filter:')
print(f"train∩val:  {len(train_tickers_raw & val_tickers_raw)}")
print(f"train∩test: {len(train_tickers_raw & test_tickers_raw)}")
print(f"val∩test:   {len(val_tickers_raw & test_tickers_raw)}")

if ENFORCE_TICKER_DISJOINT:
    val_before = len(val_df)
    test_before = len(test_df)

    val_df = val_df[~val_df['ticker'].astype(str).isin(train_tickers_raw)].copy()
    val_tickers_after = ticker_set(val_df)
    blocked_for_test = train_tickers_raw | val_tickers_after
    test_df = test_df[~test_df['ticker'].astype(str).isin(blocked_for_test)].copy()

    print('\nDisjoint ticker filtering enabled:')
    print(f'Removed from val:  {val_before - len(val_df)} rows')
    print(f'Removed from test: {test_before - len(test_df)} rows')

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            'ENFORCE_TICKER_DISJOINT removed all rows from val/test. '
            'Please regenerate split files with ticker-disjoint partitioning.'
        )

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


FINANCIAL_FEATURES = [
    'current_ratio', 'debt_equity_ratio',
    'gross_profit_margin', 'operating_profit_margin',
    'ebit_margin', 'pretax_profit_margin',
    'net_profit_margin', 'asset_turnover',
    'roe', 'roa',
    'operating_cashflow_ps', 'free_cashflow_ps'

 ]

SECTOR_COL = 'sector'
SECTOR_UNKNOWN = 'UNKNOWN'
TARGET_COL = 'rating_detail'

# If labels are strings, keep this risk order when available.
TARGET_ORDERED_LABELS = ['Distressed', 'HY', 'IG']

before_rows = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()

target_as_num = pd.to_numeric(df[TARGET_COL], errors='coerce')
if target_as_num.notna().all():
    # Numeric labels path (e.g., 0/1/2). Remap to contiguous IDs from observed classes.
    df[TARGET_COL] = target_as_num.astype(int)
    observed_classes_raw = sorted(df[TARGET_COL].unique().tolist())
    raw_to_id = {int(raw): idx for idx, raw in enumerate(observed_classes_raw)}
    id_to_raw = {idx: int(raw) for raw, idx in raw_to_id.items()}
    df[TARGET_COL] = df[TARGET_COL].map(raw_to_id).astype(int)
else:
    # String labels path (e.g., Distressed/HY/IG).
    target_as_str = df[TARGET_COL].astype(str).str.strip()
    observed_str = sorted(target_as_str.unique().tolist())
    if set(observed_str).issubset(set(TARGET_ORDERED_LABELS)):
        ordered_present = [c for c in TARGET_ORDERED_LABELS if c in observed_str]
    else:
        ordered_present = observed_str
    raw_to_id = {raw: idx for idx, raw in enumerate(ordered_present)}
    id_to_raw = {idx: raw for raw, idx in raw_to_id.items()}
    df[TARGET_COL] = target_as_str.map(raw_to_id).astype(int)

EXPECTED_CLASSES = sorted(df[TARGET_COL].unique().tolist())
if EXPECTED_CLASSES != list(range(len(EXPECTED_CLASSES))):
    raise ValueError(f'Label IDs must be contiguous from 0. Got: {EXPECTED_CLASSES}')

n_classes = len(EXPECTED_CLASSES)
TARGET_MIN, TARGET_MAX = 0, n_classes - 1

# Keep encoder-like metadata object for downstream checkpoint compatibility.
class _StaticLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes, dtype=object)

le = _StaticLabelEncoder(classes=[id_to_raw[i] for i in range(n_classes)])

df['rating_date'] = pd.to_datetime(df['rating_date'], format='mixed')

if SECTOR_COL not in df.columns:
    df[SECTOR_COL] = SECTOR_UNKNOWN
df[SECTOR_COL] = df[SECTOR_COL].fillna(SECTOR_UNKNOWN).astype(str).str.strip()
df.loc[df[SECTOR_COL] == '', SECTOR_COL] = SECTOR_UNKNOWN

sector_encoder = LabelEncoder()
df['sector_id'] = sector_encoder.fit_transform(df[SECTOR_COL])
SECTOR_CLASSES = sector_encoder.classes_.tolist()
n_sectors = len(SECTOR_CLASSES)

# Leakage guard: estimate imputation/clipping statistics on train split only.
if '__split__' in df.columns:
    split_series = df['__split__'].astype(str).str.lower()
    train_mask_raw = split_series == 'train'
    has_train_rows = bool(train_mask_raw.any())
else:
    train_mask_raw = pd.Series(False, index=df.index)
    has_train_rows = False

stats_ref = df.loc[train_mask_raw].copy() if has_train_rows else df.copy()

for col in FINANCIAL_FEATURES:
    med = stats_ref[col].median() if stats_ref[col].notna().any() else df[col].median()
    if pd.isna(med):
        med = 0.0
    if df[col].isna().any():
        df[col] = df[col].fillna(float(med))

for col in FINANCIAL_FEATURES:
    lower = stats_ref[col].quantile(0.01)
    upper = stats_ref[col].quantile(0.99)
    if pd.isna(lower) or pd.isna(upper):
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
    if pd.isna(lower) or pd.isna(upper):
        continue
    df[col] = df[col].clip(float(lower), float(upper))

print(f'Rows kept after target cleanup: {len(df)} / {before_rows}')
print(f'Number of classes (observed): {n_classes}')
print(f'Label range (model IDs): {TARGET_MIN}..{TARGET_MAX}')
print('Observed label IDs:', sorted(df[TARGET_COL].unique().tolist()))
print('Raw-to-ID mapping:', raw_to_id)
print('Decoder classes (ID -> raw):', le.classes_.tolist())
print(f'Sectors encoded: {n_sectors}')
print(f"Leakage guard reference split: {'train' if has_train_rows else 'all_data_fallback'}")
print('Sample sectors:', SECTOR_CLASSES[:10])
print(f'\nData after preprocessing: {df.shape}')

df_sorted = df.sort_values(['ticker', 'rating_date']).reset_index(drop=True)

panel_df = df_sorted[
    ['ticker', 'rating_date', 'rating_detail', 'sector_id', '__split__'] + FINANCIAL_FEATURES
].copy()
panel_df = panel_df.rename(columns={
    'ticker': 'unique_id',
    'rating_date': 'ds',
    'rating_detail': 'y'
})

ticker_counts = panel_df.groupby('unique_id').size().reset_index(name='count')
print('Ticker count statistics:')
print(ticker_counts['count'].describe())
print()

MIN_HISTORY = 1
valid_tickers = ticker_counts[ticker_counts['count'] >= MIN_HISTORY]['unique_id'].tolist()
panel_df = panel_df[panel_df['unique_id'].isin(valid_tickers)].reset_index(drop=True)

print(f'Tickers with >= {MIN_HISTORY} data points: {len(valid_tickers)}')
print(f'Panel DataFrame shape: {panel_df.shape}')
print(f'Unique tickers: {panel_df["unique_id"].nunique()}')
print(f'Date range: {panel_df["ds"].min()} to {panel_df["ds"].max()}')
print(f'Unique sectors (encoded): {panel_df["sector_id"].nunique()}')
print('Split distribution:')
print(panel_df['__split__'].value_counts())
display(panel_df.head(10))

# Adaptive context length search to maximize usable windows while preserving enough temporal context.
HORIZON = 1
ALLOW_SHORT_TICKER_PADDING = True
WINDOW_PADDING_MODE = 'edge'   # 'edge' or 'zero'

INPUT_SIZE_SEARCH_ENABLED = False
INPUT_SIZE_DEFAULT = 1
INPUT_SIZE_MIN = 0
INPUT_SIZE_MAX = 24
SINGLETON_TICKER_POLICY = 'self_target_padded'
ENABLE_BOOTSTRAP_T0_WINDOW = True
ENABLE_SYNTH_QC = False
ENABLE_WEIGHTED_SAMPLER = True
SAMPLER_WEIGHT_POWER = 0.75


def estimate_window_counts(panel_frame, input_size, horizon=1, allow_short_padding=True):
    """Estimate split-wise window counts for a given input_size using current split labels."""
    counts = {'train': 0, 'val': 0, 'test': 0}
    padded_counts = {'train': 0, 'val': 0, 'test': 0}

    for _, grp in panel_frame.groupby('unique_id'):
        g = grp.sort_values('ds').reset_index(drop=True)
        n = len(g)

        if n < 1:
            continue
        if n < horizon + 1 and not allow_short_padding:
            continue

        if n >= input_size + horizon:
            max_i = n - input_size - horizon + 1
            for i in range(max_i):
                target_idx = i + input_size
                split_label = str(g['__split__'].iloc[target_idx]).lower()
                if split_label in counts:
                    counts[split_label] += 1
        elif allow_short_padding:
            # One padded sample from the latest available target when history is short.
            target_idx = n - 1
            split_label = str(g['__split__'].iloc[target_idx]).lower()
            if split_label in counts:
                counts[split_label] += 1
                padded_counts[split_label] += 1

    total = int(sum(counts.values()))
    total_padded = int(sum(padded_counts.values()))
    return counts, padded_counts, total, total_padded


# Candidate input sizes based on available ticker history.
max_ticker_len = int(ticker_counts['count'].max()) if len(ticker_counts) > 0 else (INPUT_SIZE_DEFAULT + HORIZON)
search_upper = max(INPUT_SIZE_MIN, min(INPUT_SIZE_MAX, max_ticker_len - HORIZON))
input_candidates = list(range(INPUT_SIZE_MIN, search_upper + 1)) if search_upper >= INPUT_SIZE_MIN else [INPUT_SIZE_DEFAULT]
if INPUT_SIZE_DEFAULT not in input_candidates:
    input_candidates.append(INPUT_SIZE_DEFAULT)
input_candidates = sorted(set([int(v) for v in input_candidates if int(v) >= 1]))

search_rows = []
for k in input_candidates:
    est_counts, est_padded, est_total, est_total_padded = estimate_window_counts(
        panel_df,
        input_size=int(k),
        horizon=HORIZON,
        allow_short_padding=ALLOW_SHORT_TICKER_PADDING,
    )
    search_rows.append({
        'input_size': int(k),
        'train_windows': int(est_counts['train']),
        'val_windows': int(est_counts['val']),
        'test_windows': int(est_counts['test']),
        'total_windows': int(est_total),
        'padded_windows_total': int(est_total_padded),
    })

search_df = pd.DataFrame(search_rows).sort_values('input_size').reset_index(drop=True)
if len(search_df) == 0:
    raise ValueError('No valid INPUT_SIZE candidate found for sliding-window construction.')

max_train = float(search_df['train_windows'].max()) if search_df['train_windows'].max() > 0 else 1.0
max_total = float(search_df['total_windows'].max()) if search_df['total_windows'].max() > 0 else 1.0
max_k = float(search_df['input_size'].max()) if search_df['input_size'].max() > 0 else 1.0

# Score balances: train coverage (70%), overall coverage (20%), context length (10%).
search_df['score'] = (
    0.70 * (search_df['train_windows'] / max_train) +
    0.20 * (search_df['total_windows'] / max_total) +
    0.10 * (search_df['input_size'] / max_k)
)

if INPUT_SIZE_SEARCH_ENABLED:
    feasible_df = search_df[
        (search_df['train_windows'] > 0) &
        (search_df['val_windows'] > 0) &
        (search_df['test_windows'] > 0)
    ].copy()

    if len(feasible_df) > 0:
        best_row = feasible_df.sort_values(['score', 'train_windows', 'input_size'], ascending=[False, False, False]).iloc[0]
        INPUT_SIZE = int(best_row['input_size'])
        input_size_reason = 'adaptive_search'
    else:
        INPUT_SIZE = int(np.clip(INPUT_SIZE_DEFAULT, 1, INPUT_SIZE_MAX))
        input_size_reason = 'fallback_default_no_feasible_candidate'
else:
    INPUT_SIZE = int(np.clip(INPUT_SIZE_DEFAULT, 1, INPUT_SIZE_MAX))
    input_size_reason = 'search_disabled'

INPUT_SIZE_SEARCH_REPORT = search_df.copy()

print(f'INPUT_SIZE selected: {INPUT_SIZE} ({input_size_reason})')
print(f'HORIZON (forecast): {HORIZON}')
print(f'Short ticker padding enabled: {ALLOW_SHORT_TICKER_PADDING} ({WINDOW_PADDING_MODE})')
print(f'Singleton ticker policy: {SINGLETON_TICKER_POLICY}')
print('\nTop input-size candidates by score:')
print(
    search_df
    .sort_values(['score', 'train_windows', 'input_size'], ascending=[False, False, False])
    .head(8)
    .to_string(index=False)
)

min_required = INPUT_SIZE + HORIZON
eligible_full_window_tickers = ticker_counts[ticker_counts['count'] >= min_required]['unique_id'].tolist()
panel_df_filtered = panel_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

print(f'\nTickers with >= {min_required} data points (full windows): {len(eligible_full_window_tickers)}')
print(f'Panel shape (no hard drop by min_required): {panel_df_filtered.shape}')

# -- Feature engineering: add first-order deltas by ticker to capture trend changes --
DELTA_FEATURES = []
for col in FINANCIAL_FEATURES:
    dcol = f'{col}_delta'
    panel_df_filtered[dcol] = (
        panel_df_filtered.groupby('unique_id')[col].diff().fillna(0.0)
    )
    DELTA_FEATURES.append(dcol)

MODEL_FEATURES = FINANCIAL_FEATURES + DELTA_FEATURES

# -- Global scaling (fit on train split only), replacing RevIN normalization --
scaler = RobustScaler()
train_feature_mask = panel_df_filtered['__split__'] == 'train'
if train_feature_mask.any():
    scaler.fit(panel_df_filtered.loc[train_feature_mask, MODEL_FEATURES].values)
else:
    scaler.fit(panel_df_filtered[MODEL_FEATURES].values)
panel_df_filtered[MODEL_FEATURES] = scaler.transform(
    panel_df_filtered[MODEL_FEATURES].values
)

# -- Build sliding window dataset --
class CreditRatingDataset(Dataset):
    """Sliding window dataset.
    Returns X shape (INPUT_SIZE, n_channels), last_y context, and sector_id."""
    def __init__(self, sequences, augment=False, noise_std=0.0, feature_dropout_prob=0.0):
        self.sequences = sequences
        self.augment = augment
        self.noise_std = float(noise_std)
        self.feature_dropout_prob = float(feature_dropout_prob)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, last_y, sector_id, y_target = self.sequences[idx]
        X = np.asarray(X, dtype=np.float32).copy()

        if self.augment:
            if self.noise_std > 0:
                X += np.random.normal(0.0, self.noise_std, size=X.shape).astype(np.float32)
            if self.feature_dropout_prob > 0:
                drop_mask = np.random.rand(X.shape[1]) < self.feature_dropout_prob
                if drop_mask.any():
                    X[:, drop_mask] = 0.0

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(last_y, dtype=torch.long),
            torch.tensor(sector_id, dtype=torch.long),
            torch.tensor(y_target, dtype=torch.long),
        )


def build_padded_window(values, target_idx, input_size, mode='edge'):
    """Build fixed-length window ending right before target_idx with optional left padding."""
    if values.shape[0] == 0:
        return None

    # For singleton tickers, fallback to the first row as minimal context.
    if target_idx <= 0:
        X_raw = values[:1]
    else:
        start_idx = max(0, target_idx - input_size)
        X_raw = values[start_idx:target_idx]

    if X_raw.shape[0] == 0:
        X_raw = values[:1]

    if X_raw.shape[0] >= input_size:
        return X_raw[-input_size:]

    pad_len = input_size - X_raw.shape[0]
    if mode == 'zero':
        pad_block = np.zeros((pad_len, X_raw.shape[1]), dtype=X_raw.dtype)
    else:
        pad_block = np.repeat(X_raw[[0]], pad_len, axis=0)

    return np.concatenate([pad_block, X_raw], axis=0)


feature_cols = MODEL_FEATURES
n_channels = len(feature_cols)

train_seqs = []
val_seqs = []
test_seqs = []
test_seq_tickers = []
train_seq_is_synth = []
train_seq_is_change = []
has_synth_col = 'is_synthetic' in panel_df_filtered.columns

short_seq_windows_added = 0
singleton_windows_added = 0
bootstrap_t0_windows_added = 0
short_seq_tickers_used = set()

for uid, grp in panel_df_filtered.groupby('unique_id'):
    grp = grp.sort_values('ds').reset_index(drop=True)
    values = grp[feature_cols].values  # (T, C)
    n = len(values)

    if n == 0:
        continue

    if ENABLE_BOOTSTRAP_T0_WINDOW and n >= INPUT_SIZE + HORIZON:
        # Add one extra sample at the first timestamp to maximize train coverage.
        target_idx = 0
        X0 = build_padded_window(values, target_idx=target_idx, input_size=INPUT_SIZE, mode=WINDOW_PADDING_MODE)
        if X0 is not None and X0.shape[0] == INPUT_SIZE:
            last_y0 = int(grp['y'].iloc[target_idx])
            y_target0 = int(grp['y'].iloc[target_idx])
            sector_id0 = int(grp['sector_id'].iloc[target_idx])
            split_label0 = str(grp['__split__'].iloc[target_idx]).lower()

            sample0 = (X0, last_y0, sector_id0, y_target0)
            if split_label0 == 'test':
                test_seqs.append(sample0)
                test_seq_tickers.append(uid)
            elif split_label0 == 'val':
                val_seqs.append(sample0)
            else:
                train_seqs.append(sample0)
                train_seq_is_change.append(int(y_target0 != last_y0))
                if has_synth_col:
                    train_seq_is_synth.append(int(grp['is_synthetic'].iloc[target_idx]))
                else:
                    train_seq_is_synth.append(0)

            bootstrap_t0_windows_added += 1
    if n < HORIZON + 1 and not ALLOW_SHORT_TICKER_PADDING:
        continue

    if n < INPUT_SIZE + HORIZON and ALLOW_SHORT_TICKER_PADDING:
        # Keep short ticker (including n=1) by creating one padded window.
        target_idx = n - 1
        X = build_padded_window(values, target_idx=target_idx, input_size=INPUT_SIZE, mode=WINDOW_PADDING_MODE)
        if X is not None and X.shape[0] == INPUT_SIZE:
            last_y_idx = max(0, target_idx - 1)
            last_y = int(grp['y'].iloc[last_y_idx])
            y_target = int(grp['y'].iloc[target_idx])
            sector_id = int(grp['sector_id'].iloc[target_idx])
            split_label = str(grp['__split__'].iloc[target_idx]).lower()

            sample = (X, last_y, sector_id, y_target)
            if split_label == 'test':
                test_seqs.append(sample)
                test_seq_tickers.append(uid)
            elif split_label == 'val':
                val_seqs.append(sample)
            else:
                train_seqs.append(sample)
                train_seq_is_change.append(int(y_target != last_y))
                if has_synth_col:
                    train_seq_is_synth.append(int(grp['is_synthetic'].iloc[target_idx]))
                else:
                    train_seq_is_synth.append(0)

            short_seq_windows_added += 1
            if n == 1:
                singleton_windows_added += 1
            short_seq_tickers_used.add(uid)
        continue

    for i in range(n - INPUT_SIZE - HORIZON + 1):
        X = values[i : i + INPUT_SIZE]
        last_y = int(grp['y'].iloc[i + INPUT_SIZE - 1])
        target_idx = i + INPUT_SIZE
        y_target = int(grp['y'].iloc[target_idx])  # next-step class
        sector_id = int(grp['sector_id'].iloc[target_idx])
        split_label = str(grp['__split__'].iloc[target_idx]).lower()

        sample = (X, last_y, sector_id, y_target)
        if split_label == 'test':
            test_seqs.append(sample)
            test_seq_tickers.append(uid)
        elif split_label == 'val':
            val_seqs.append(sample)
        else:
            train_seqs.append(sample)
            train_seq_is_change.append(int(y_target != last_y))
            if has_synth_col:
                train_seq_is_synth.append(int(grp['is_synthetic'].iloc[target_idx]))
            else:
                train_seq_is_synth.append(0)

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

test_seq_tickers = np.array(test_seq_tickers, dtype=object)
train_seq_is_synth = np.array(train_seq_is_synth, dtype=int)
train_seq_is_change = np.array(train_seq_is_change, dtype=int)

# -- Synthetic sequence quality control and ratio cap --
SYNTH_MAX_ABS_FEATURE = 4.0
SYNTH_MAX_RATIO = 0.20
train_seqs_before_synth_qc = len(train_seqs)
if ENABLE_SYNTH_QC and has_synth_col and len(train_seqs) > 0:
    seq_max_abs = np.array([float(np.max(np.abs(s[0]))) for s in train_seqs], dtype=np.float32)
    keep_mask = np.ones(len(train_seqs), dtype=bool)
    synth_mask = train_seq_is_synth == 1

    # Remove synthetic windows with extreme amplitudes (often unrealistic).
    keep_mask[synth_mask & (seq_max_abs > SYNTH_MAX_ABS_FEATURE)] = False

    # Cap remaining synthetic ratio to reduce domain shift from generated data.
    kept_synth_idx = np.where(keep_mask & synth_mask)[0]
    kept_real_idx = np.where(keep_mask & (~synth_mask))[0]
    max_synth_keep = int(len(kept_real_idx) * SYNTH_MAX_RATIO / max(1e-9, 1.0 - SYNTH_MAX_RATIO))
    if len(kept_synth_idx) > max_synth_keep > 0:
        rng_seq = np.random.default_rng(SEED)
        drop_idx = rng_seq.choice(
            kept_synth_idx,
            size=len(kept_synth_idx) - max_synth_keep,
            replace=False,
        )
        keep_mask[drop_idx] = False

    if not keep_mask.all():
        train_seqs = [s for s, k in zip(train_seqs, keep_mask) if k]
        train_seq_is_synth = train_seq_is_synth[keep_mask]
        train_seq_is_change = train_seq_is_change[keep_mask]
train_seqs_after_synth_qc = len(train_seqs)

# Temporarily disable sequence-level regularization to test model capacity/underfit.
TRAIN_WINDOW_NOISE_STD = 0.01  # Reduced from 0.02 for training stability
TRAIN_FEATURE_DROPOUT = 0.03  # Reduced from 0.05 to close train/val loss gap

train_ds = CreditRatingDataset(
    train_seqs,
    augment=True,
    noise_std=TRAIN_WINDOW_NOISE_STD,
    feature_dropout_prob=TRAIN_FEATURE_DROPOUT,
)
val_ds = CreditRatingDataset(val_seqs, augment=False)
test_ds = CreditRatingDataset(test_seqs, augment=False)

# Simplified: no sampler re-weight; keep natural training distribution.
train_labels = np.array([s[3] for s in train_seqs], dtype=int)
class_freq_raw = np.bincount(train_labels, minlength=n_classes).astype(float)
non_zero_freq = class_freq_raw[class_freq_raw > 0]
imbalance_ratio = (
    float(class_freq_raw.max() / non_zero_freq.min())
    if len(non_zero_freq) > 0 else 1.0
)
synthetic_ratio_train = float(train_seq_is_synth.mean()) if len(train_seq_is_synth) > 0 else 0.0
transition_ratio_train = float(train_seq_is_change.mean()) if len(train_seq_is_change) > 0 else 0.0
weighted_sampler = None
if ENABLE_WEIGHTED_SAMPLER and len(train_labels) > 0:
    class_weights = np.zeros_like(class_freq_raw, dtype=np.float64)
    non_zero_mask = class_freq_raw > 0
    class_weights[non_zero_mask] = class_freq_raw[non_zero_mask].sum() / class_freq_raw[non_zero_mask]
    sample_weights = np.power(class_weights[train_labels], SAMPLER_WEIGHT_POWER)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

BATCH_SIZE = 64  # Increased from 32 for more stable gradients
NUM_WORKERS = 4 if IN_KAGGLE else 0
PIN_MEMORY = torch.cuda.is_available()

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=(weighted_sampler is None),
    sampler=weighted_sampler,
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

print(f'\nChannels: {n_channels}')
print(f'Base features: {len(FINANCIAL_FEATURES)} | Delta features: {len(DELTA_FEATURES)}')
print(f'Train tickers: {len(train_tickers)} | Val tickers: {len(val_tickers)}')
print(f'Train samples: {len(train_ds)}')
print(f'Val samples:   {len(val_ds)}')
print(f'Test samples:  {len(test_ds)}')
print(f'Test ticker refs: {len(test_seq_tickers)}')
print(f'Short padded windows added: {short_seq_windows_added} (tickers: {len(short_seq_tickers_used)})')
print(f'Singleton (n=1) windows kept via padding: {singleton_windows_added}')
print(f'Bootstrap t0 windows added: {bootstrap_t0_windows_added}')
print(f'Synth QC enabled: {ENABLE_SYNTH_QC} | removed windows: {train_seqs_before_synth_qc - train_seqs_after_synth_qc}')
print(f'Train synthetic ratio (windows): {synthetic_ratio_train:.3f}')
print(f'Train transition ratio (y_t+1 != y_t): {transition_ratio_train:.3f}')
print(f'Train class freq min/max: {class_freq_raw.min():.0f}/{class_freq_raw.max():.0f}')
print(f'Imbalance ratio (max/min non-zero): {imbalance_ratio:.2f}')
print(f'Weighted sampler enabled: {weighted_sampler is not None}')
print(f'Sampler weight power: {SAMPLER_WEIGHT_POWER}')
print(f'Train window noise std: {TRAIN_WINDOW_NOISE_STD}')
print(f'Train feature dropout prob: {TRAIN_FEATURE_DROPOUT}')

# Verify shape
sample_X, sample_last_y, sample_sector_id, sample_y = train_ds[0]
print(f'\nSample X shape: {sample_X.shape}  (T, C)')
print(f'Sample last_y: {sample_last_y}')
print(f'Sample sector_id: {sample_sector_id}')
print(f'Sample y: {sample_y}')

# ============================================================
# Deterministic DataLoader Rebuild (stability/reproducibility)
# ============================================================

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

weighted_sampler = None
if ENABLE_WEIGHTED_SAMPLER and len(train_labels) > 0:
    class_weights = np.zeros_like(class_freq_raw, dtype=np.float64)
    non_zero_mask = class_freq_raw > 0
    class_weights[non_zero_mask] = class_freq_raw[non_zero_mask].sum() / class_freq_raw[non_zero_mask]
    sample_weights = np.power(class_weights[train_labels], SAMPLER_WEIGHT_POWER)
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=loader_generator,
    )

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=(weighted_sampler is None),
    sampler=weighted_sampler,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    persistent_workers=bool(NUM_WORKERS > 0),
    generator=loader_generator if weighted_sampler is None else None,
 )

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    persistent_workers=bool(NUM_WORKERS > 0),
    generator=loader_generator,
 )

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    persistent_workers=bool(NUM_WORKERS > 0),
    generator=loader_generator,
 )

print('Rebuilt DataLoaders with deterministic generator and worker seeding.')
print(f'Weighted sampler enabled: {weighted_sampler is not None}')
print(f'Persistent workers: {bool(NUM_WORKERS > 0)}')

def build_effective_num_class_weights(class_counts, beta=0.995):
    """Effective-number reweighting (Cui et al.) with mean=1 normalization."""
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.maximum(counts, 0.0)
    weights = np.zeros_like(counts, dtype=np.float64)

    valid = counts > 0
    if valid.any():
        effective_num = 1.0 - np.power(float(beta), counts[valid])
        weights[valid] = (1.0 - float(beta)) / np.maximum(effective_num, 1e-12)
        weights[valid] = weights[valid] / np.mean(weights[valid])
    else:
        weights[:] = 1.0
    return weights


class FocalOrdinalLoss(nn.Module):
    """Focal loss + ordinal distance regularization with optional class-balance weights."""
    def __init__(self, n_classes, gamma=1.5, ordinal_alpha=0.04, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.gamma = float(gamma)
        self.ordinal_alpha = float(ordinal_alpha)
        self.label_smoothing = float(label_smoothing)
        if class_weights is None:
            self.class_weights = None
        else:
            cw = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', cw)

    def forward(self, logits, targets):
        # Keep CE computation in FP32 and move class weights to the same device as logits.
        logits_for_loss = logits.float()
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=torch.float32)

        ce_loss = F.cross_entropy(
            logits_for_loss,
            targets,
            reduction='none',
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_ce = ((1.0 - pt) ** self.gamma) * ce_loss
        focal_ce = focal_ce.mean()

        probs = F.softmax(logits_for_loss, dim=-1)
        classes = torch.arange(self.n_classes, device=logits.device, dtype=probs.dtype)
        expected_class = torch.sum(probs * classes.unsqueeze(0), dim=-1)
        dist_loss = F.mse_loss(expected_class, targets.float())

        return focal_ce + self.ordinal_alpha * dist_loss


CLASS_BALANCE_BETA = 0.995
class_weights_effective = build_effective_num_class_weights(
    class_freq_raw if 'class_freq_raw' in globals() else np.ones(n_classes),
    beta=CLASS_BALANCE_BETA,
 )

criterion_settings = {
    'loss_name': 'focal_ordinal_class_balanced',
    'focal_gamma': 1.5,
    'label_smoothing': 0.00,
    'ordinal_alpha': 0.04,
    'class_balance_beta': CLASS_BALANCE_BETA,
}

criterion = FocalOrdinalLoss(
    n_classes=n_classes,
    gamma=criterion_settings['focal_gamma'],
    ordinal_alpha=criterion_settings['ordinal_alpha'],
    label_smoothing=criterion_settings['label_smoothing'],
    class_weights=class_weights_effective,
).to(device)

FOCAL_GAMMA = criterion_settings['focal_gamma']
print(
    f"Loss: {criterion_settings['loss_name']} | "
    f"gamma={criterion_settings['focal_gamma']} | "
    f"smoothing={criterion_settings['label_smoothing']} | "
    f"ordinal_alpha={criterion_settings['ordinal_alpha']} | "
    f"beta={criterion_settings['class_balance_beta']}"
 )
print('Effective class weights:', np.round(class_weights_effective, 4).tolist())

class FuzzyLayer(nn.Module):
    """Gaussian fuzzy membership expansion with configurable MFs per feature."""
    def __init__(self, input_features, n_mfs=5, init_sigma=1.0):
        super().__init__()
        self.input_features = input_features
        self.n_mfs = n_mfs
        self.centers = nn.Parameter(torch.linspace(-1.0, 1.0, n_mfs).repeat(input_features, 1))
        self.log_sigma = nn.Parameter(torch.full((input_features, n_mfs), math.log(init_sigma)))

    def forward(self, x):
        # x: (B, T, F)
        x_exp = x.unsqueeze(-1)
        centers = self.centers.unsqueeze(0).unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0).unsqueeze(0).clamp(min=1e-3)
        memberships = torch.exp(-0.5 * ((x_exp - centers) / sigma) ** 2)
        return memberships.reshape(x.shape[0], x.shape[1], self.input_features * self.n_mfs)


class TemporalSelfAttentionBlock(nn.Module):
    """Pre-norm Transformer block with learnable relative positional bias."""
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, max_relative_position=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.max_relative_position = int(max_relative_position)
        self.relative_bias = nn.Parameter(torch.zeros(2 * self.max_relative_position + 1))

    def _relative_bias_matrix(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device)
        rel = pos[None, :] - pos[:, None]
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        rel = rel + self.max_relative_position
        return self.relative_bias[rel].to(dtype=dtype)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_bias = self._relative_bias_matrix(x.size(1), x.device, x.dtype)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AttentivePool(nn.Module):
    """Learnable weighted pooling over temporal axis."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        ctx = (w * x).sum(dim=1)
        return ctx, w


class TLSTMFuzzyClassifier(nn.Module):
    """Transformer-LSTM + Fuzzy classifier with sector embedding + transition head."""
    def __init__(
        self,
        n_channels,
        n_classes,
        n_sectors,
        hidden_size=128,
        dropout=0.10,
        n_mfs=5,
        d_model=128,
        n_heads=4,
        n_layers=3,
        sector_emb_dim=16,
        max_relative_position=32,
    ):
        super().__init__()
        self.fuzzy = FuzzyLayer(n_channels, n_mfs=n_mfs)

        fuzzy_dim = n_channels * n_mfs
        self.input_proj = nn.Sequential(
            nn.Linear(fuzzy_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.blocks = nn.ModuleList([
            TemporalSelfAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                max_relative_position=max_relative_position,
            )
            for _ in range(n_layers)
        ])

        self.pre_lstm_norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_pool = AttentivePool(hidden_size * 2)
        self.last_y_embed = nn.Embedding(n_classes, hidden_size)
        self.sector_embed = nn.Embedding(n_sectors, sector_emb_dim)

        transition_in_dim = hidden_size * 2 + sector_emb_dim
        self.transition_head = nn.Sequential(
            nn.Linear(transition_in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        head_in_dim = hidden_size * 3 + sector_emb_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_size * 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, last_y, sector_id, return_aux=False):
        x_fuzzy = self.fuzzy(x)
        x_embed = self.input_proj(x_fuzzy)

        for blk in self.blocks:
            x_embed = blk(x_embed)

        x_embed = self.pre_lstm_norm(x_embed)
        lstm_out, _ = self.lstm(x_embed)
        seq_ctx, _ = self.attn_pool(lstm_out)

        last_y_emb = self.last_y_embed(last_y)
        sector_emb = self.sector_embed(sector_id)

        transition_logits = self.transition_head(torch.cat([seq_ctx, sector_emb], dim=-1)).squeeze(-1)

        out = torch.cat([seq_ctx, last_y_emb, sector_emb], dim=-1)
        logits = self.head(out)

        if return_aux:
            return logits, transition_logits
        return logits


FUZZY_MFS = 5
MODEL_D_MODEL = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3
LSTM_HIDDEN = 128
SECTOR_EMB_DIM = 16
TLSTM_DROPOUT = 0.10
MAX_RELATIVE_POSITION = 32

model = TLSTMFuzzyClassifier(
    n_channels=n_channels,
    n_classes=n_classes,
    n_sectors=n_sectors,
    hidden_size=LSTM_HIDDEN,
    dropout=TLSTM_DROPOUT,
    n_mfs=FUZZY_MFS,
    d_model=MODEL_D_MODEL,
    n_heads=TRANSFORMER_HEADS,
    n_layers=TRANSFORMER_LAYERS,
    sector_emb_dim=SECTOR_EMB_DIM,
    max_relative_position=MAX_RELATIVE_POSITION,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Fuzzy MFs per feature: {FUZZY_MFS}')
print(f'Projected d_model: {MODEL_D_MODEL}')
print(f'Transformer layers/heads: {TRANSFORMER_LAYERS}/{TRANSFORMER_HEADS}')
print(f'Relative position window: +/-{MAX_RELATIVE_POSITION}')
print(f'BiLSTM hidden size: {LSTM_HIDDEN}')
print(f'Sector embedding dim: {SECTOR_EMB_DIM}')
print(f'Channels: {n_channels} (base + delta features)')
print(f'Sectors: {n_sectors}')
print(f'Model parameters: {n_params:,}')
print('\nTLSTM-Fuzzy model created successfully!')
print(model)

# ============================================================
# Lightweight Hyperparameter Search (val_f1_weighted)
# ============================================================
