# Additional quality summary for TimeGAN generation
if len(synthetic_df_filtered) > 0:
    print(f"\n{'='*70}")
    print("TIMEGAN GENERATION SUMMARY")
    print(f"{'='*70}")

    synthetic_counts = synthetic_df_filtered[CONFIG['target_column']].value_counts().sort_index()
    real_counts = train[CONFIG['target_column']].value_counts().sort_index()

    summary_df = pd.DataFrame({
        'real_train_count': real_counts,
        'synthetic_count': synthetic_counts
    }).fillna(0).astype(int)
    summary_df['augmented_count'] = summary_df['real_train_count'] + summary_df['synthetic_count']
    summary_df['synthetic_share_in_augmented'] = (
        summary_df['synthetic_count'] / summary_df['augmented_count'].replace(0, np.nan)
    ).fillna(0.0).round(4)

    print(summary_df)

    summary_path = REPORTS_DIR / 'timegan_generation_summary.csv'
    summary_df.to_csv(summary_path, index=True)
    print(f"\nâœ“ Generation summary saved: {summary_path}")
else:
    print("\nâ  No synthetic samples to summarize")