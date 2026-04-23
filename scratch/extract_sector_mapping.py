import pandas as pd
from pathlib import Path

# Paths
common_3groups_path = Path(r"e:\thesis\data\processed\merged_credit_rating_common_3groups.csv")
augmented_path = Path(r"e:\thesis\data\processed\train_augmented_timegan.csv")

# Load
df_orig = pd.read_csv(common_3groups_path)
df_aug = pd.read_csv(augmented_path)

# Filter real data only from augmented
df_aug_real = df_aug[df_aug['is_synthetic'] == 0].copy()

# Merge to find mapping
mapping_df = df_aug_real[['ticker', 'rating_date', 'sector']].merge(
    df_orig[['ticker', 'rating_date', 'sector']], 
    on=['ticker', 'rating_date'], 
    suffixes=('_encoded', '_name')
)

# Extract unique pairs
sector_mapping = mapping_df[['sector_encoded', 'sector_name']].drop_duplicates().sort_values('sector_encoded')

print("Sector Mapping (ID -> Name):")
print(sector_mapping.to_string(index=False))

# Also check for __MISSING__ or other special tokens
all_aug_sectors = df_aug['sector'].unique()
encoded_ids = sector_mapping['sector_encoded'].tolist()
for s_id in all_aug_sectors:
    if s_id not in encoded_ids:
        print(f"ID {s_id} has no direct mapping in real data (likely __MISSING__ or synthetic-only)")
