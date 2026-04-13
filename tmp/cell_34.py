# Save augmented training set
output_file_timegan = SPLITS_DIR / 'train_augmented_timegan.csv'
output_file_compat = SPLITS_DIR / 'train_augmented_ctgan.csv'  # compatibility alias

train_augmented.to_csv(output_file_timegan, index=False)
train_augmented.to_csv(output_file_compat, index=False)

print(f"\nâœ“ Augmented training set saved to: {output_file_timegan}")
print(f"âœ“ Compatibility copy saved to: {output_file_compat}")
print(f"  Total records: {len(train_augmented)}")
print(f"  Real: {(train_augmented['is_synthetic']==0).sum()}")
print(f"  Synthetic: {(train_augmented['is_synthetic']==1).sum()}")