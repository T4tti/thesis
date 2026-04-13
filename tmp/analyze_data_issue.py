from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

root = Path(r'e:/thesis')
paths = {
    'train': root / 'data/processed/train_augmented_timegan.csv',
    'val': root / 'data/processed/val.csv',
    'test': root / 'data/processed/test.csv',
}

for k,p in paths.items():
    if not p.exists():
        raise FileNotFoundError(f'{k} missing: {p}')

dfs = {k: pd.read_csv(p) for k,p in paths.items()}

print('=== Basic Shape ===')
for k,df in dfs.items():
    print(f'{k:5s}: rows={len(df):6d}, cols={len(df.columns):2d}, tickers={df["ticker"].nunique():4d}')

print('\n=== Column Check ===')
for k,df in dfs.items():
    print(f'{k:5s}: has_is_synthetic={"is_synthetic" in df.columns}')

# Ensure date and numeric labels
for df in dfs.values():
    df['rating_date'] = pd.to_datetime(df['rating_date'], errors='coerce')
    df['rating_detail'] = pd.to_numeric(df['rating_detail'], errors='coerce')

print('\n=== Label Support / Shift ===')
train_classes = set(dfs['train']['rating_detail'].dropna().astype(int).unique())
for k,df in dfs.items():
    y = df['rating_detail'].dropna().astype(int)
    vc = y.value_counts().sort_index()
    print(f'{k:5s}: classes={len(vc):2d}, min_count={vc.min():4d}, max_count={vc.max():4d}, top3={vc.sort_values(ascending=False).head(3).to_dict()}')
    miss = sorted(train_classes - set(vc.index))
    if k != 'train':
        print(f'      missing_vs_train={len(miss)} classes')

# Duplicate/conflict diagnostics
def duplicate_stats(df, name):
    g = df.groupby(['ticker','rating_date']).size()
    dup_pairs = g[g>1]
    n_dup_rows = int(dup_pairs.sum())
    n_dup_pairs = int(len(dup_pairs))

    # conflicting labels at same ticker-date
    c = df.groupby(['ticker','rating_date'])['rating_detail'].nunique()
    conflict_pairs = c[c>1]
    n_conflict_pairs = int(len(conflict_pairs))

    print(f'{name:5s}: duplicate_ticker_date_pairs={n_dup_pairs}, rows_in_dup_pairs={n_dup_rows}, conflicting_label_pairs={n_conflict_pairs}')

print('\n=== Duplicate / Same-Day Conflict ===')
for k,df in dfs.items():
    duplicate_stats(df, k)


def build_transitions(df):
    trans = []
    for t, grp in df.sort_values(['ticker','rating_date']).groupby('ticker'):
        grp = grp.dropna(subset=['rating_date','rating_detail'])
        if len(grp) < 2:
            continue
        y = grp['rating_detail'].astype(int).values
        d = grp['rating_date'].values
        synth = grp['is_synthetic'].fillna(0).astype(int).values if 'is_synthetic' in grp.columns else np.zeros(len(grp), dtype=int)
        for i in range(len(grp)-1):
            trans.append({
                'ticker': t,
                'y_prev': int(y[i]),
                'y_next': int(y[i+1]),
                'is_change': int(y[i] != y[i+1]),
                'day_gap': int((d[i+1] - d[i]).astype('timedelta64[D]').astype(int)),
                'next_is_synth': int(synth[i+1]),
            })
    return pd.DataFrame(trans)

trans = {k: build_transitions(df) for k,df in dfs.items()}

print('\n=== Transition Core Stats ===')
for k,t in trans.items():
    if t.empty:
        print(f'{k:5s}: no transitions')
        continue
    persist = 1.0 - t['is_change'].mean()
    same_day = (t['day_gap'] == 0).mean()
    neg_gap = (t['day_gap'] < 0).mean()
    print(f'{k:5s}: transitions={len(t):6d}, persistence={persist:.4f}, change_rate={1-persist:.4f}, same_day_transitions={same_day:.4f}, negative_gap={neg_gap:.4f}')

print('\n=== Synthetic Impact (train only) ===')
if not trans['train'].empty:
    tt = trans['train']
    if 'is_synthetic' in dfs['train'].columns:
        synth_row_ratio = dfs['train']['is_synthetic'].fillna(0).astype(int).mean()
        synth_next_ratio = tt['next_is_synth'].mean()
        change_given_synth = tt.loc[tt['next_is_synth']==1, 'is_change'].mean() if (tt['next_is_synth']==1).any() else np.nan
        change_given_real = tt.loc[tt['next_is_synth']==0, 'is_change'].mean() if (tt['next_is_synth']==0).any() else np.nan
        print(f'train row synthetic ratio={synth_row_ratio:.4f}')
        print(f'train transition next_is_synth ratio={synth_next_ratio:.4f}')
        print(f'change_rate|next_synth={change_given_synth:.4f} vs change_rate|next_real={change_given_real:.4f}')

# Baselines on val transitions
print('\n=== Baseline Accuracy on val transitions ===')
val_t = trans['val'].copy()
train_t = trans['train'].copy()
if len(val_t) == 0 or len(train_t) == 0:
    print('insufficient transitions')
else:
    persist_acc = (val_t['y_prev'] == val_t['y_next']).mean()

    # Markov argmax predictor from train counts
    markov = train_t.groupby(['y_prev','y_next']).size().rename('cnt').reset_index()
    idx = markov.groupby('y_prev')['cnt'].idxmax()
    best_next = dict(zip(markov.loc[idx, 'y_prev'], markov.loc[idx, 'y_next']))
    default_next = int(train_t['y_next'].value_counts().idxmax())
    pred = val_t['y_prev'].map(best_next).fillna(default_next).astype(int)
    markov_acc = (pred.values == val_t['y_next'].values).mean()

    # Sector+last_y majority baseline
    val = dfs['val'].sort_values(['ticker','rating_date']).copy()
    tr = dfs['train'].sort_values(['ticker','rating_date']).copy()
    tr['y'] = tr['rating_detail'].astype(int)
    tr['sector'] = tr['sector'].astype(str)
    pair_map = tr.groupby(['sector','y'])['y'].agg(lambda s: s.mode().iloc[0]).to_dict()

    # build aligned val rows t+1 with prev y and sector at t+1
    rows = []
    for tk,g in val.groupby('ticker'):
        g = g.dropna(subset=['rating_date','rating_detail']).sort_values('rating_date')
        if len(g)<2:
            continue
        y = g['rating_detail'].astype(int).values
        sec = g['sector'].astype(str).values
        for i in range(len(g)-1):
            rows.append((sec[i+1], int(y[i]), int(y[i+1])))
    if rows:
        arr = np.array(rows, dtype=object)
        pred2 = []
        for sec,yp,_ in rows:
            pred2.append(pair_map.get((str(sec), int(yp)), default_next))
        true2 = [r[2] for r in rows]
        sec_last_acc = (np.array(pred2)==np.array(true2)).mean()
    else:
        sec_last_acc = np.nan

    print(f'persistence_acc(val)={persist_acc:.4f}')
    print(f'markov_argmax_acc(val)={markov_acc:.4f}')
    print(f'sector_lastY_majority_acc(val)={sec_last_acc:.4f}')

print('\n=== Done ===')
