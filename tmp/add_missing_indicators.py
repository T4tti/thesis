import pandas as pd

# Load source data
df_years = pd.read_csv('data/processed/processed_dataset_years.csv')
df_rating = pd.read_csv('data/raw/corporate_rating.csv')

print("Processing corporate_rating.csv to extract year and align with processed_dataset_years...")

# Convert Date to year in corporate_rating
df_rating['Date_parsed'] = pd.to_datetime(df_rating['Date'], format='%m/%d/%Y', errors='coerce')
df_rating['Year'] = df_rating['Date_parsed'].dt.year

# Select only the indicators we need to add
missing_indicators = {
    'Gross_Profit_Margin': 'grossProfitMargin',
    'Operating_Profit_Margin': 'operatingProfitMargin',
    'Pretax_Profit_Margin': 'pretaxProfitMargin',
    'OCF_Per_Share': 'operatingCashFlowPerShare',
    'FCF_Per_Share': 'freeCashFlowPerShare',
}

# Create a merge key based on Symbol and Year
df_rating_agg = df_rating.groupby(['Symbol', 'Year']).agg({
    'grossProfitMargin': 'mean',
    'operatingProfitMargin': 'mean',
    'pretaxProfitMargin': 'mean',
    'operatingCashFlowPerShare': 'mean',
    'freeCashFlowPerShare': 'mean',
}).reset_index()

# Rename columns to match target
df_rating_agg.columns = ['Code_Cpn', 'Year', 'Gross_Profit_Margin', 'Operating_Profit_Margin', 
                         'Pretax_Profit_Margin', 'OCF_Per_Share', 'FCF_Per_Share']

print("\nData from corporate_rating after grouping:")
print(f'Shape: {df_rating_agg.shape}')
print(f'Companies: {df_rating_agg["Code_Cpn"].nunique()}')
print(f'Year range: {df_rating_agg["Year"].min()} to {df_rating_agg["Year"].max()}')

# Merge with processed_dataset_years
print("\nBefore merge:")
print(f'processed_dataset_years shape: {df_years.shape}')

df_merged = df_years.merge(df_rating_agg, on=['Code_Cpn', 'Year'], how='left')

print('\nAfter merge:')
print(f'Merged shape: {df_merged.shape}')
print(f'Rows with new data: {(~df_merged["Gross_Profit_Margin"].isna()).sum()}')
print(f'Rows with missing new data: {df_merged["Gross_Profit_Margin"].isna().sum()}')

# Show sample of merged data
print('\nSample of merged data (with new indicators filled):')
print(df_merged[df_merged['Gross_Profit_Margin'].notna()][
    ['Year', 'Code_Cpn', 'Name_Cpn', 'Net_Profit_Margin', 'Gross_Profit_Margin', 
     'Operating_Profit_Margin', 'Pretax_Profit_Margin', 'OCF_Per_Share', 'FCF_Per_Share']
].head(10))

# Save to a new file
output_path = 'data/processed/processed_dataset_years_enhanced.csv'
df_merged.to_csv(output_path, index=False)
print(f'\nSaved enhanced dataset to: {output_path}')
