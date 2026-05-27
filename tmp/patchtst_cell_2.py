FINANCIAL_FEATURES = [
    'current_ratio', 'debt_equity_ratio', 'gross_profit_margin', 'operating_profit_margin',
    'ebit_margin', 'pretax_profit_margin', 'net_profit_margin', 'asset_turnover',
    'roe', 'roa', 'operating_cashflow_ps', 'free_cashflow_ps'
]
TARGET_COL = 'rating_detail'
TARGET_ORDERED_LABELS = ['Distressed', 'HY', 'IG']
INPUT_SIZE = 4
HORIZON = 1

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

TRAIN_PATH = resolve_split_path('/kaggle/input/datasets/tailength/corporate-credit-rating/test/train.csv', ['data/processed/train_augmented_timegan.csv'])
VAL_PATH = resolve_split_path('/kaggle/input/datasets/tailength/corporate-credit-rating/test/val.csv', ['data/processed/val.csv'])
TEST_PATH = resolve_split_path('/kaggle/input/datasets/tailength/corporate-credit-rating/test/test.csv', ['data/processed/test.csv'])

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df['__split__'] = 'train'
val_df['__split__'] = 'val'
test_df['__split__'] = 'test'
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

df = df.dropna(subset=[TARGET_COL]).copy()
target_as_num = pd.to_numeric(df[TARGET_COL], errors='coerce')
if target_as_num.notna().all():
    df[TARGET_COL] = target_as_num.astype(int)
    observed = sorted(df[TARGET_COL].unique().tolist())
    raw_to_id = {int(v): i for i, v in enumerate(observed)}
    df[TARGET_COL] = df[TARGET_COL].map(raw_to_id).astype(int)
else:
    tgt = df[TARGET_COL].astype(str).str.strip()
    observed = sorted(tgt.unique().tolist())
    ordered = [x for x in TARGET_ORDERED_LABELS if x in observed] if set(observed).issubset(set(TARGET_ORDERED_LABELS)) else observed
    raw_to_id = {v: i for i, v in enumerate(ordered)}
    df[TARGET_COL] = tgt.map(raw_to_id).astype(int)

n_classes = int(df[TARGET_COL].nunique())
df['rating_date'] = pd.to_datetime(df['rating_date'], errors='coerce', format='mixed')
if 'sector' not in df.columns:
    df['sector'] = 'UNKNOWN'
df['sector'] = df['sector'].fillna('UNKNOWN').astype(str)
sector_encoder = LabelEncoder()
df['sector_id'] = sector_encoder.fit_transform(df['sector'])
n_sectors = int(df['sector_id'].nunique())

train_mask = df['__split__'].eq('train')
stats_ref = df.loc[train_mask].copy() if train_mask.any() else df.copy()
for c in FINANCIAL_FEATURES:
    med = stats_ref[c].median() if stats_ref[c].notna().any() else 0.0
    df[c] = df[c].fillna(float(0.0 if pd.isna(med) else med))
for c in FINANCIAL_FEATURES:
    lo = stats_ref[c].quantile(0.01)
    hi = stats_ref[c].quantile(0.99)
    if pd.notna(lo) and pd.notna(hi):
        df[c] = df[c].clip(float(lo), float(hi))

df = df.sort_values(['ticker', 'rating_date']).reset_index(drop=True)
for c in FINANCIAL_FEATURES:
    df[f'{c}_delta'] = df.groupby('ticker')[c].diff().fillna(0.0)
MODEL_FEATURES = FINANCIAL_FEATURES + [f'{c}_delta' for c in FINANCIAL_FEATURES]

scaler = RobustScaler()
scaler.fit(df.loc[df['__split__'].eq('train'), MODEL_FEATURES].values)
df[MODEL_FEATURES] = scaler.transform(df[MODEL_FEATURES].values)

def build_padded_window(values, target_idx, input_size):
    if target_idx <= 0:
        x_raw = values[:1]
    else:
        x_raw = values[max(0, target_idx - input_size):target_idx]
    if x_raw.shape[0] == 0:
        x_raw = values[:1]
    if x_raw.shape[0] >= input_size:
        return x_raw[-input_size:]
    pad = np.repeat(x_raw[[0]], input_size - x_raw.shape[0], axis=0)
    return np.concatenate([pad, x_raw], axis=0)

def build_sequences(frame, input_size=12, horizon=1):
    out = {'train': [], 'val': [], 'test': []}
    for _, g in frame.groupby('ticker'):
        g = g.sort_values('rating_date').reset_index(drop=True)
        vals = g[MODEL_FEATURES].values.astype(np.float32)
        ys = g[TARGET_COL].values.astype(int)
        sec = g['sector_id'].values.astype(int)
        sp = g['__split__'].astype(str).str.lower().values
        n = len(g)
        for t in range(n):
            x = build_padded_window(vals, t, input_size)
            last_y_idx = max(0, t - 1)
            out[sp[t]].append((x, int(ys[last_y_idx]), int(sec[t]), int(ys[t])))
    return out

seqs = build_sequences(df, input_size=INPUT_SIZE, horizon=HORIZON)
for split_name in ['train', 'val', 'test']:
    if len(seqs[split_name]) == 0:
        raise ValueError(f'Split {split_name} khong co mau sau khi tao window; kiem tra lai du lieu va INPUT_SIZE.')

class WindowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        x, ly, sec, y = self.samples[i]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(ly, dtype=torch.long),
            torch.tensor(sec, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

train_ds = WindowDataset(seqs['train'])
val_ds = WindowDataset(seqs['val'])
test_ds = WindowDataset(seqs['test'])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

print('Train/Val/Test windows:', len(train_ds), len(val_ds), len(test_ds))
