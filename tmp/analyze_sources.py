import pandas as pd

# Check processed_dataset_years
df_years = pd.read_csv('e:\\thesis\\data\\processed\\processed_dataset_years.csv')
print("=== processed_dataset_years.csv ===")
print(f'Shape: {df_years.shape}')
print(f'Columns: {list(df_years.columns)}')
print(f'\nUnique companies: {df_years["Code_Cpn"].nunique()}')
print(f'Companies: {sorted(df_years["Code_Cpn"].unique())[:20]}')
print(f'\nYear range: {df_years["Year"].min()} to {df_years["Year"].max()}')
print('\nData sources/files to integrate:')
print('- merged_credit_rating_common_3groups has: gross_profit_margin, operating_profit_margin, ebit_margin, pretax_profit_margin, operating_cashflow_ps, free_cashflow_ps')
print('- But it contains international companies, not Vietnamese')
print('\nProcessed_dataset_years appears to be Vietnamese company financial data organized by year')
