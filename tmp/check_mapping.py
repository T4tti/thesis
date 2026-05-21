import pandas as pd

# Load both datasets
df_years = pd.read_csv('data/processed/processed_dataset_years.csv')
df_rating = pd.read_csv('data/raw/corporate_rating.csv')

print("=== processed_dataset_years.csv ===")
print(f'Shape: {df_years.shape}')
print(f'Sample companies: {df_years["Name_Cpn"].head(10).tolist()}')
print(f'Symbol codes: {df_years["Code_Cpn"].head(10).tolist()}')
print(f'\nYear range: {df_years["Year"].min()} to {df_years["Year"].max()}')

print("\n=== corporate_rating.csv ===")
print(f'Shape: {df_rating.shape}')
print(f'Sample companies: {df_rating["Name"].head(10).tolist()}')
print(f'Symbols: {df_rating["Symbol"].head(10).tolist()}')
print(f'\nDate range: {df_rating["Date"].min()} to {df_rating["Date"].max()}')
print(f'\nRating agencies: {df_rating["Rating Agency Name"].unique()[:5]}')

# Check for matching companies
print("\n=== Checking for overlaps ===")
symbols_years = set(df_years['Code_Cpn'].unique())
symbols_rating = set(df_rating['Symbol'].unique())
overlap = symbols_years & symbols_rating
print(f'Overlapping symbols: {len(overlap)} out of {len(symbols_years)} in years')
print(f'Sample overlaps: {list(overlap)[:20]}')
