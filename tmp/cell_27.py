def analyze_synthetic_quality(real_df, synthetic_df, numeric_cols, categorical_cols, target_col):
    """Analyze quality of synthetic data vs real data"""
    if len(synthetic_df) == 0:
        print("No synthetic data to analyze")
        return {}
    
    results = {}
    
    print("\n=== Synthetic Data Quality Analysis ===")
    
    # 1. KS test for numeric columns
    print("\nKolmogorov-Smirnov Test (Numeric Features):")
    print(f"{'Column':<30} {'KS Statistic':>15} {'p-value':>15} {'Similar?':>10}")
    print("-" * 75)
    
    for col in numeric_cols[:10]:  # Show first 10
        if col in synthetic_df.columns:
            ks_stat, ks_pval = stats.ks_2samp(real_df[col], synthetic_df[col])
            similar = "Yes" if ks_pval > 0.05 else "No"
            results[f'{col}_ks_stat'] = float(ks_stat)
            results[f'{col}_ks_pval'] = float(ks_pval)
            print(f"{col:<30} {ks_stat:>15.4f} {ks_pval:>15.4f} {similar:>10}")
    
    # 2. JS divergence for categorical columns
    if categorical_cols:
        print("\nJensen-Shannon Divergence (Categorical Features):")
        print(f"{'Column':<30} {'JS Divergence':>15} {'Similar?':>10}")
        print("-" * 60)
        
        for col in categorical_cols:
            if col in synthetic_df.columns and col != target_col:
                real_dist = real_df[col].value_counts(normalize=True).sort_index()
                syn_dist = synthetic_df[col].value_counts(normalize=True).sort_index()
                
                # Align distributions
                all_cats = sorted(set(real_dist.index) | set(syn_dist.index))
                real_aligned = np.array([real_dist.get(c, 0) for c in all_cats])
                syn_aligned = np.array([syn_dist.get(c, 0) for c in all_cats])
                
                js_div = jensenshannon(real_aligned, syn_aligned)
                similar = "Yes" if js_div < 0.1 else "No"
                results[f'{col}_js_div'] = float(js_div)
                print(f"{col:<30} {js_div:>15.4f} {similar:>10}")
    
    # 3. Target distribution comparison
    print("\n=== Target Distribution Comparison ===")
    real_target = real_df[target_col].value_counts(normalize=True).sort_index()
    syn_target = synthetic_df[target_col].value_counts(normalize=True).sort_index()
    
    comparison_df = pd.DataFrame({
        'Real': real_target,
        'Synthetic': syn_target
    }).fillna(0)
    print(comparison_df)
    
    return results

# Analyze quality
if len(synthetic_df_filtered) > 0:
    quality_results = analyze_synthetic_quality(
        train,
        synthetic_df_filtered,
        FEATURE_NUMERIC,
        FEATURE_CATEGORICAL,
        CONFIG['target_column']
    )
    
    # Save quality results
    with open(MODELS_DIR / 'quality_analysis.json', 'w') as f:
        json.dump(quality_results, f, indent=2)