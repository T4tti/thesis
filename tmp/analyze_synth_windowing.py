from pathlib import Path
import pandas as pd
import numpy as np

p = Path(r'e:/thesis/data/processed/train_augmented_timegan.csv')
train = pd.read_csv(p)
train['rating_date'] = pd.to_datetime(train['rating_date'], errors='coerce')
train['rating_detail'] = pd.to_numeric(train['rating_detail'], errors='coerce').astype('Int64')

if 'is_synthetic' not in train.columns:
    print('no is_synthetic col')
    raise SystemExit

train['is_synthetic'] = train['is_synthetic'].fillna(0).astype(int)

# 1) synthetic ticker length
ticker_len = train.groupby('ticker').size()
syn_tickers = train.loc[train['is_synthetic']==1, 'ticker'].unique()
syn_len = ticker_len.reindex(syn_tickers).fillna(0).astype(int)

print('=== Synthetic Ticker Length ===')
print(f'synthetic rows={int((train.is_synthetic==1).sum())}, synthetic tickers={len(syn_tickers)}')
print(f'synthetic ticker len median={syn_len.median()}, mean={syn_len.mean():.2f}, min={syn_len.min()}, max={syn_len.max()}')
print('len distribution (top):', syn_len.value_counts().head(10).to_dict())

# 2) rows eligible for window creation with INPUT_SIZE=8,HORIZON=1 as current run
INPUT_SIZE=8
HORIZON=1
min_required=INPUT_SIZE+HORIZON
eligible_tickers = set(ticker_len[ticker_len>=min_required].index)
eligible_rows = train[train['ticker'].isin(eligible_tickers)]
elig_syn_rows = eligible_rows[eligible_rows['is_synthetic']==1]
print('\n=== Eligibility for Windowing (>=9 rows/ticker) ===')
print(f'eligible tickers={len(eligible_tickers)} / total tickers={train.ticker.nunique()}')
print(f'eligible rows={len(eligible_rows)} / total rows={len(train)}')
print(f'eligible synthetic rows={len(elig_syn_rows)}')

# 3) class distribution shift vs val/test
def dist(df):
    vc = df['rating_detail'].dropna().astype(int).value_counts().sort_index()
    p = vc / vc.sum()
    return p

val = pd.read_csv(r'e:/thesis/data/processed/val.csv')
test = pd.read_csv(r'e:/thesis/data/processed/test.csv')
for d in [val,test]:
    d['rating_detail'] = pd.to_numeric(d['rating_detail'], errors='coerce').astype('Int64')

ptr = dist(train)
pval = dist(val)
ptest = dist(test)
classes = sorted(set(ptr.index)|set(pval.index)|set(ptest.index))
ptr = ptr.reindex(classes, fill_value=0)
pval = pval.reindex(classes, fill_value=0)
ptest = ptest.reindex(classes, fill_value=0)

# Jensen-Shannon distance

def jsd(p,q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5*(p+q)
    def kld(a,b):
        mask = a>0
        return np.sum(a[mask]*np.log(a[mask]/np.clip(b[mask],1e-12,None)))
    return np.sqrt(0.5*kld(p,m)+0.5*kld(q,m))

print('\n=== Class Distribution Shift ===')
print(f'JSD(train,val)={jsd(ptr.values,pval.values):.4f}')
print(f'JSD(train,test)={jsd(ptr.values,ptest.values):.4f}')

print('\nTop classes by split:')
for name,prob in [('train',ptr),('val',pval),('test',ptest)]:
    top = prob.sort_values(ascending=False).head(5)
    print(f'{name}:', {int(k):float(v) for k,v in top.items()})

# 4) check target conflicts on same ticker/date due multi-agency
conf = train.groupby(['ticker','rating_date'])['rating_detail'].nunique()
conf_pairs = conf[conf>1]
print('\n=== Label Conflict Ticker-Date (train) ===')
print(f'conflicting_pairs={len(conf_pairs)}')
if len(conf_pairs)>0:
    # show average spread at conflicts
    merged = train.merge(conf_pairs.rename('nuniq').reset_index(), on=['ticker','rating_date'], how='inner')
    spread = merged.groupby(['ticker','rating_date'])['rating_detail'].agg(lambda s: int(s.max()-s.min()))
    print(f'mean_label_spread_on_conflicts={spread.mean():.2f}, max_spread={spread.max()}')

print('\n=== Done ===')
