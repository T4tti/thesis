import pandas as pd

df = pd.read_csv('e:\\thesis\\data\\processed\\merged_credit_rating_common_3groups.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'\nData types:\n{df.dtypes}')
print(f'\nSample row:\n{df.iloc[0]}')
print(f'\nUnique companies: {df["company_name"].nunique()}')
print(f'Date range: {df["rating_date"].min()} to {df["rating_date"].max()}')
