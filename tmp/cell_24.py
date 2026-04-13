def filter_synthetic_samples(
    synthetic_df,
    train_df,
    numeric_cols,
    quantile_range=(0.01, 0.99)
):
    """Filter synthetic samples with quantile clipping and duplicate removal."""
    if len(synthetic_df) == 0:
        return synthetic_df

    initial_count = len(synthetic_df)
    filtered_df = synthetic_df.copy()

    print(f"\nFiltering with quantile range: {quantile_range}")

    # 1) Clip numeric values to train quantiles
    for col in numeric_cols:
        if col in filtered_df.columns and col in train_df.columns:
            lower = train_df[col].quantile(quantile_range[0])
            upper = train_df[col].quantile(quantile_range[1])
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').clip(lower, upper)

    # 2) Drop duplicated synthetic rows (DISABLED to preserve time-series sequence integrity)
    duplicates = 0
    # filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    final_count = len(filtered_df)
    removed = initial_count - final_count
    removed_pct = 100.0 * removed / max(initial_count, 1)

    print("\nFiltering summary:")
    print(f"  Initial: {initial_count}")
    print(f"  After filtering: {final_count}")
    print(f"  Removed: {removed} ({removed_pct:.1f}%)")
    print(f"  Duplicate rows removed: {duplicates}")

    return filtered_df

# Apply filtering
if len(synthetic_df) > 0:
    q_low = float(CONFIG.get('quality_quantile_low', 0.01))
    q_high = float(CONFIG.get('quality_quantile_high', 0.99))

    synthetic_df_filtered = filter_synthetic_samples(
        synthetic_df=synthetic_df,
        train_df=train,
        numeric_cols=TIMEGAN_NUMERIC_COLUMNS,
        quantile_range=(q_low, q_high)
    )
    print("\nQuality filtering completed")
else:
    synthetic_df_filtered = synthetic_df