from pathlib import Path
import pandas as pd
import numpy as np

root = Path('e:/thesis')
train = pd.read_csv(root / 'data/processed/train_augmented_timegan.csv')
val = pd.read_csv(root / 'data/processed/val.csv')
test = pd.read_csv(root / 'data/processed/test.csv')
for d, n in [(train, 'train'), (val, 'val'), (test, 'test')]:
    d['__split__'] = n

df = pd.concat([train, val, test], ignore_index=True)
df['rating_detail'] = pd.to_numeric(df['rating_detail'], errors='coerce')
df = df.dropna(subset=['rating_detail']).copy()
df['rating_detail'] = df['rating_detail'].astype(int)
df['rating_date'] = pd.to_datetime(df['rating_date'], format='mixed', errors='coerce')
df = df.dropna(subset=['rating_date'])

print('After cleanup shape:', df.shape)
print('Rows by split after datetime parse:')
print(df['__split__'].value_counts().to_string())

for s in ['train', 'val', 'test']:
    vc = df.loc[df['__split__'] == s, 'rating_detail'].value_counts().sort_index()
    print('')
    print('[' + s + '] rows=' + str(int(vc.sum())) + ', classes=' + str(len(vc)))
    print('min=' + str(int(vc.min())) + ', max=' + str(int(vc.max())) + ', imbalance=' + ('{:.2f}'.format(vc.max()/vc[vc>0].min())))
    print('rare<=10:', vc[vc <= 10].to_dict())

# Window counts using notebook settings INPUT_SIZE=1, HORIZON=1, bootstrap t0 on, allow short padding on
panel = df.sort_values(['ticker', 'rating_date']).reset_index(drop=True)
counts = {'train': [], 'val': [], 'test': []}
for _, grp in panel.groupby('ticker'):
    grp = grp.sort_values('rating_date').reset_index(drop=True)
    y = grp['rating_detail'].astype(int).to_numpy()
    split = grp['__split__'].astype(str).str.lower().to_numpy()
    n = len(grp)

    if n >= 2:
        t = 0
        if split[t] in counts:
            counts[split[t]].append(int(y[t]))

    if n < 2:
        t = n - 1
        if t >= 0 and split[t] in counts:
            counts[split[t]].append(int(y[t]))
        continue

    for t in range(1, n):
        if split[t] in counts:
            counts[split[t]].append(int(y[t]))

for s in ['train', 'val', 'test']:
    arr = np.array(counts[s], dtype=int)
    vc = pd.Series(arr).value_counts().sort_index()
    print('')
    print('[' + s + '_seqs] windows=' + str(arr.size) + ', classes=' + str(len(vc)))
    print('min=' + str(int(vc.min())) + ', max=' + str(int(vc.max())) + ', imbalance=' + ('{:.2f}'.format(vc.max()/vc[vc>0].min())))
    print('rare<=10:', vc[vc <= 10].to_dict())

all_cls = set(range(22))
train_cls = set(np.unique(np.array(counts['train'], dtype=int))) if len(counts['train']) else set()
print('\nMissing classes in train windows:', sorted(all_cls - train_cls))
