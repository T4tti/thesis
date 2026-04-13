# ============================================================
# Hybrid Rebalancing: Downsample majority + oversample minority
# ============================================================
# After TimeGAN augmentation, apply optional capping to majority classes

ENABLE_HYBRID_REBALANCING = False  # Set False by default for sequence data. Row-level rebalancing destroys temporal sequences.  # Set False to skip
MAJORITY_CAP_PERCENTILE = 75     # Cap majority classes at this percentile of class counts
MINORITY_FLOOR = 200             # Ensure every class has at least this many samples

if ENABLE_HYBRID_REBALANCING and len(train_augmented) > 0:
    print("\n" + "=" * 70)
    print("HYBRID REBALANCING (Downsample majority + Floor minority)")
    print("=" * 70)

    target_col = CONFIG['target_column']
    aug_counts = train_augmented[target_col].value_counts().sort_index()

    # Determine cap: percentile of non-zero class counts
    nonzero_aug = aug_counts[aug_counts > 0]
    cap_value = int(np.percentile(nonzero_aug.values, MAJORITY_CAP_PERCENTILE))
    cap_value = max(cap_value, MINORITY_FLOOR * 2)

    print(f"  Class count P{MAJORITY_CAP_PERCENTILE}: {cap_value}")
    print(f"  Minority floor: {MINORITY_FLOOR}")

    rebalanced_chunks = []
    for cls_label in sorted(train_augmented[target_col].unique()):
        cls_df = train_augmented[train_augmented[target_col] == cls_label]
        n = len(cls_df)

        if n > cap_value:
            # Downsample majority class
            cls_df = cls_df.sample(n=cap_value, random_state=RANDOM_SEED)
            print(f"  Class {cls_label}: {n} -> {cap_value} (capped)")
        elif n < MINORITY_FLOOR and n > 0:
            # Oversample minority class via duplication
            extra_needed = MINORITY_FLOOR - n
            extra = cls_df.sample(n=extra_needed, random_state=RANDOM_SEED, replace=True)
            cls_df = pd.concat([cls_df, extra], ignore_index=True)
            print(f"  Class {cls_label}: {n} -> {len(cls_df)} (oversampled)")
        else:
            print(f"  Class {cls_label}: {n} (kept)")

        rebalanced_chunks.append(cls_df)

    train_augmented = pd.concat(rebalanced_chunks, ignore_index=True)
    train_augmented = train_augmented.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    final_counts = train_augmented[target_col].value_counts().sort_index()
    nonzero_final = final_counts[final_counts > 0]

    print(f"\n  After hybrid rebalancing: {len(train_augmented)} total samples")
    print(f"  Class count range: [{nonzero_final.min()}, {nonzero_final.max()}]")
    if len(nonzero_final) > 0 and nonzero_final.max() > 0:
        ratio = nonzero_final.min() / nonzero_final.max()
        print(f"  Imbalance ratio: {ratio:.3f}")
        print(f"  {'GOOD' if ratio > 0.3 else 'MODERATE' if ratio > 0.1 else 'STILL IMBALANCED'}")

    print(f"\n  Full distribution:")
    print(final_counts)
else:
    print("\nHybrid rebalancing: SKIPPED")