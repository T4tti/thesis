import pandas as pd
from pathlib import Path

augmented_path = Path(r"e:\thesis\data\processed\train_augmented_timegan.csv")
df_aug = pd.read_csv(augmented_path)

print("Columns:", df_aug.columns.tolist())
print("\nFirst 5 rows (Real):")
print(df_aug[df_aug['is_synthetic'] == 0].head())

print("\nUnique sectors in augmented:", sorted(df_aug['sector'].unique()))
print("Unique rating_agencies in augmented:", sorted(df_aug['rating_agency'].unique()))
