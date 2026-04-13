from pathlib import Path
import pandas as pd
import numpy as np

root = Path('e:/thesis')
train_path = root / 'data/processed/train_augmented_timegan.csv'
val_path = root / 'data/processed/val.csv'
test_path = root / 'data/processed/test.csv'

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)
for d, n in [(train, 'train'), (val, 'val'), (test, 'test')]:
    d['__split__'] = n

df = pd.concat([train, val, test], ignore_index=True)
df['rating_detail'] = pd.to_numeric(df['rating_detail'], errors='coerce')
df = df.dropna(subset=['rating_detail']).copy()
df['rating_detail'] = df['rating_detail'].astype(int)
df['rating_date'] = pd.to_datetime(df['rating_date'], errors='coerce')
df = df.dropna(subset=['rating_date'])

print('Row-level class distribution by split:')
for s in ['train', 'val', 'test']:
    vc = df.loc[df['__split__'] == s, 'rating_detail'].value_counts().sort_index()
    print('')
    print('[' + s + '] total=' + str(int(vc.sum())) + ', classes=' + str(len(vc)))
    print(vc.to_string())

panel = df.sort_values(['ticker', 'rating_date']).reset_index(drop=True)
train_targets, val_targets, test_targets = [], [], []

for _, grp in panel.groupby('ticker'):
    grp = grp.sort_values('rating_date').reset_index(drop=True)
    y = grp['rating_detail'].astype(int).to_numpy()
    split = grp['__split__'].astype(str).str.lower().to_numpy()
    n = len(grp)

    if n >= 2:
        t = 0
        if split[t] == 'train':
            train_targets.append(int(y[t]))
        elif split[t] == 'val':
            val_targets.append(int(y[t]))
        elif split[t] == 'test':
            test_targets.append(int(y[t]))

    if n < 2:
        t = n - 1
        if t >= 0:
            if split[t] == 'train':
                train_targets.append(int(y[t]))
            elif split[t] == 'val':
                val_targets.append(int(y[t]))
            elif split[t] == 'test':
                test_targets.append(int(y[t]))
        continue

    for t in range(1, n):
        if split[t] == 'train':
            train_targets.append(int(y[t]))
        elif split[t] == 'val':
            val_targets.append(int(y[t]))
        elif split[t] == 'test':
            test_targets.append(int(y[t]))

for arr, name in [(train_targets, 'train_seqs'), (val_targets, 'val_seqs'), (test_targets, 'test_seqs')]:
    arr = np.array(arr, dtype=int)
    if arr.size == 0:
        print('')
        print('[' + name + '] EMPTY')
        continue
    vc = pd.Series(arr).value_counts().sort_index()
    rare = vc[vc <= 10]
    print('')
    print('[' + name + '] total=' + str(arr.size) + ', classes=' + str(len(vc)))
    print(vc.to_string())
    print('rare_classes(<=10): ' + str(rare.to_dict()))

expected = set(range(22))
train_classes = set(np.unique(np.array(train_targets, dtype=int))) if len(train_targets) > 0 else set()
print('')
print('Missing classes in train windows from expected 0..21:', sorted(expected - train_classes))
