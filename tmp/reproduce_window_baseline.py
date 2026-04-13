from pathlib import Path
import pandas as pd
import numpy as np

train = pd.read_csv(r'e:/thesis/data/processed/train_augmented_timegan.csv')
val = pd.read_csv(r'e:/thesis/data/processed/val.csv')
test = pd.read_csv(r'e:/thesis/data/processed/test.csv')
for name,df in [('train',train),('val',val),('test',test)]:
    df['__split__'] = name

df = pd.concat([train,val,test], ignore_index=True)
df['rating_date'] = pd.to_datetime(df['rating_date'], errors='coerce')
df['rating_detail'] = pd.to_numeric(df['rating_detail'], errors='coerce')
df = df.dropna(subset=['rating_detail','rating_date']).copy()
df['rating_detail'] = df['rating_detail'].astype(int)

df_sorted = df.sort_values(['ticker','rating_date']).reset_index(drop=True)
panel = df_sorted[['ticker','rating_date','rating_detail','__split__']].copy()
panel = panel.rename(columns={'ticker':'unique_id','rating_date':'ds','rating_detail':'y'})

# same as notebook
MIN_HISTORY = 3
ticker_counts = panel.groupby('unique_id').size().reset_index(name='count')
valid_tickers = ticker_counts[ticker_counts['count'] >= MIN_HISTORY]['unique_id'].tolist()
panel = panel[panel['unique_id'].isin(valid_tickers)].reset_index(drop=True)

median_seq_len = int(ticker_counts[ticker_counts['count'] >= MIN_HISTORY]['count'].median())
INPUT_SIZE = max(2, min(median_seq_len - 1, 24))
HORIZON = 1
min_required = INPUT_SIZE + HORIZON
valid_tickers2 = ticker_counts[ticker_counts['count'] >= min_required]['unique_id'].tolist()
panel = panel[panel['unique_id'].isin(valid_tickers2)].sort_values(['unique_id','ds']).reset_index(drop=True)

train_seqs=[]; val_seqs=[]; test_seqs=[]; train_seq_is_synth=[]
has_synth = 'is_synthetic' in df.columns
# merge synth flag back
synth_map = df_sorted[['ticker','rating_date','is_synthetic']].copy() if has_synth else None
if has_synth:
    synth_map['is_synthetic'] = synth_map['is_synthetic'].fillna(0).astype(int)
    panel = panel.merge(synth_map, left_on=['unique_id','ds'], right_on=['ticker','rating_date'], how='left')
    panel['is_synthetic'] = panel['is_synthetic'].fillna(0).astype(int)

for uid, grp in panel.groupby('unique_id'):
    grp = grp.sort_values('ds').reset_index(drop=True)
    n = len(grp)
    for i in range(n - INPUT_SIZE - HORIZON + 1):
        last_y = int(grp['y'].iloc[i + INPUT_SIZE - 1])
        target_idx = i + INPUT_SIZE
        y_target = int(grp['y'].iloc[target_idx])
        split = str(grp['__split__'].iloc[target_idx]).lower()
        if split == 'train':
            train_seqs.append((last_y, y_target))
            if has_synth:
                train_seq_is_synth.append(int(grp['is_synthetic'].iloc[target_idx]))
        elif split == 'val':
            val_seqs.append((last_y, y_target))
        else:
            test_seqs.append((last_y, y_target))

val_persist = np.mean([int(a==b) for a,b in val_seqs]) if val_seqs else np.nan
train_synth_ratio = np.mean(train_seq_is_synth) if train_seq_is_synth else np.nan

print('median_seq_len', median_seq_len)
print('INPUT_SIZE', INPUT_SIZE, 'min_required', min_required)
print('train_windows', len(train_seqs), 'val_windows', len(val_seqs), 'test_windows', len(test_seqs))
print('val_persistence_baseline', round(float(val_persist), 4))
print('train_window_synthetic_ratio', float(train_synth_ratio) if not np.isnan(train_synth_ratio) else np.nan)
