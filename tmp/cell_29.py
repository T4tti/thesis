# Combine real and synthetic data
if len(synthetic_df_filtered) > 0:
    # Add flag to distinguish real vs synthetic
    train_real = train.copy()
    train_real['is_synthetic'] = 0

    synthetic_final = synthetic_df_filtered.copy()
    synthetic_final['is_synthetic'] = 1

    # Align synthetic schema to real train schema
    for col in train_real.columns:
        if col not in synthetic_final.columns:
            if pd.api.types.is_numeric_dtype(train_real[col]):
                synthetic_final[col] = np.nan
            else:
                synthetic_final[col] = '__SYNTHETIC__'

    # Keep same column order as real data
    synthetic_final = synthetic_final[train_real.columns]

    # Combine
    train_augmented = pd.concat([train_real, synthetic_final], ignore_index=True)

    # Shuffle
    train_augmented = train_augmented.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print("\n=== Augmented Training Set ===")
    print(f"Real samples: {len(train_real)}")
    print(f"Synthetic samples: {len(synthetic_final)}")
    print(f"Total: {len(train_augmented)}")
    print(f"Synthetic ratio: {100*len(synthetic_final)/len(train_augmented):.1f}%")

    # Class distribution after augmentation
    print("\n=== Class Distribution After Augmentation ===")
    print(train_augmented[CONFIG['target_column']].value_counts().sort_index())

else:
    train_augmented = train.copy()
    train_augmented['is_synthetic'] = 0
    print("\nâ  No augmentation applied (no synthetic samples)")