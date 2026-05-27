from pathlib import Path
import pandas as pd

p = Path(r'e:/thesis/data/processed/merged_credit_rating_common.csv')
df = pd.read_csv(p)
df['rating_date'] = pd.to_datetime(df['rating_date'], errors='coerce')

print('=== merged_credit_rating_common ===')
print(f'rows={len(df)}, tickers={df["ticker"].nunique()}, classes={df["rating_detail"].nunique()}')

# duplicate and conflicts
pair_size = df.groupby(['ticker','rating_date']).size()
dup_pairs = pair_size[pair_size>1]
conf_pairs = df.groupby(['ticker','rating_date'])['rating_detail'].nunique()
conf_pairs = conf_pairs[conf_pairs>1]
print(f'duplicate_ticker_date_pairs={len(dup_pairs)}, conflicting_label_pairs={len(conf_pairs)}')

# transitions on raw label strings
trans=[]
for tk,g in df.sort_values(['ticker','rating_date']).groupby('ticker'):
    g = g.dropna(subset=['rating_date','rating_detail'])
    y = g['rating_detail'].astype(str).values
    d = g['rating_date'].values
    if len(y)<2:
        continue
    for i in range(len(y)-1):
        trans.append((y[i], y[i+1], int(y[i]==y[i+1]), int((d[i+1]-d[i]).astype('timedelta64[D]').astype(int))))

if trans:
    t = pd.DataFrame(trans, columns=['y_prev','y_next','is_same','day_gap'])
    print(f'transitions={len(t)}, persistence={t.is_same.mean():.4f}, same_day={ (t.day_gap==0).mean():.4f}')
    # top transitions
    top = t.groupby(['y_prev','y_next']).size().sort_values(ascending=False).head(10)
    print('top transitions:', top.to_dict())

# agency inconsistency at same ticker-date
if 'rating_agency' in df.columns:
    multi_ag = df.groupby(['ticker','rating_date'])['rating_agency'].nunique()
    multi_ag = multi_ag[multi_ag>1]
    print(f'multi_agency_same_ticker_date_pairs={len(multi_ag)}')

print('=== Done ===')
