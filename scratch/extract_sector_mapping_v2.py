import pandas as pd
from pathlib import Path

# Paths
common_3groups_path = Path(r"e:\thesis\data\processed\merged_credit_rating_common_3groups.csv")
augmented_path = Path(r"e:\thesis\data\processed\train_augmented_timegan.csv")

# Load
df_orig = pd.read_csv(common_3groups_path)
df_aug = pd.read_csv(augmented_path)

# Normalize dates
df_orig['rating_date'] = pd.to_datetime(df_orig['rating_date'])
df_aug['rating_date'] = pd.to_datetime(df_aug['rating_date'])

# Filter real data only from augmented
df_aug_real = df_aug[df_aug['is_synthetic'] == 0].copy()

# Merge
mapping_df = df_aug_real[['ticker', 'rating_date', 'sector']].merge(
    df_orig[['ticker', 'rating_date', 'sector']], 
    on=['ticker', 'rating_date'], 
    suffixes=('_encoded', '_name')
)

# Extract unique pairs
sector_mapping = mapping_df[['sector_encoded', 'sector_name']].drop_duplicates().sort_values('sector_encoded')

print("Sector Mapping (ID -> Name):")
for _, row in sector_mapping.iterrows():
    print(f"{row['sector_encoded']}: {row['sector_name']}")

# Check counts
print("\nCounts per ID:")
print(mapping_df['sector_encoded'].value_counts().sort_index())
