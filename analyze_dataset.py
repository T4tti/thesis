import pandas as pd
df = pd.read_csv('data/processed/merged_credit_rating_common_3groups.csv')
print("Columns:", df.columns.tolist())
print("\nShape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nSample Data:")
print(df.head())
