# Train Quality Filter Classifier (Random Forest)
print("\n=== Training Quality Filter Classifier ===")
from sklearn.ensemble import RandomForestClassifier

# Use original train data for classifier
features = TIMEGAN_FEATURE_COLUMNS
X_real = train[features].fillna(0)
y_real = train[CONFIG['target_column']]

print(f"Training RandomForest on {len(X_real)} real samples with balanced weights...")
quality_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
quality_clf.fit(X_real, y_real)
print("âœ“ Qualifier model trained.")

def get_oversample_factor(real_count):
    if real_count < 10: return 20.0
    if real_count < 30: return 10.0
    if real_count < 50: return 5.0
    return 3.0

# Generate synthetic samples from TimeGAN models and unroll to tabular rows
print("\n=== Generating Synthetic Samples (TimeGAN) ===")

target_col = CONFIG['target_column']
date_col = CONFIG['date_column']
entity_col = CONFIG['entity_column']
seq_len = int(CONFIG['seq_len'])

def _safe_reference_date(series):
    dates = pd.to_datetime(series, errors='coerce').dropna()
    if len(dates) == 0:
        return pd.Timestamp(datetime.now().date())
    return dates.max()

base_reference_date = _safe_reference_date(train[date_col]) if date_col in train.columns else pd.Timestamp(datetime.now().date())

generated_rows = []
generated_windows_log = []
synthetic_window_counter = 0

for cls, n_windows in sorted(windows_per_class.items()):
    if int(n_windows) <= 0:
        continue

    model_key = f'class_{int(cls)}'
    if model_key in TIMEGAN_MODELS:
        active_model = TIMEGAN_MODELS[model_key]
        model_source = model_key
    elif 'global' in TIMEGAN_MODELS:
        active_model = TIMEGAN_MODELS['global']
        model_source = 'global'
    else:
        print(f"âš  No TimeGAN model available for class {cls}, skipping")
        continue

    try:
        real_count = class_counts.get(int(cls), 0)
        oversample_factor = get_oversample_factor(real_count)
        oversampled_n_windows = int(np.ceil(n_windows * oversample_factor))
        
        print(f"Class {cls} (real={real_count}): Base target {n_windows} windows. Oversampling ({oversample_factor}x) -> {oversampled_n_windows} windows")

        sampled_norm = safe_timegan_sample(active_model, oversampled_n_windows)

        if sampled_norm.shape[0] == 0:
            print(f"âš  Class {cls}: model returned 0 windows")
            continue

        sampled_norm = sampled_norm[:, :seq_len, :len(TIMEGAN_FEATURE_COLUMNS)]
        sampled_raw = inverse_timegan_scale(sampled_norm, TIMEGAN_SCALER, TIMEGAN_FEATURE_COLUMNS)

        class_meta = TIMEGAN_SEQUENCE_META[TIMEGAN_SEQUENCE_META['window_label'] == int(cls)].copy()
        class_start_dates = pd.to_datetime(class_meta.get('start_date', pd.Series([], dtype='datetime64[ns]')), errors='coerce').dropna().tolist()
        if len(class_start_dates) == 0:
            class_start_dates = [base_reference_date]

        for win_idx in range(sampled_raw.shape[0]):
            seq_block = sampled_raw[win_idx]
            ref_date = class_start_dates[win_idx % len(class_start_dates)]
            if pd.isna(ref_date):
                ref_date = base_reference_date + pd.Timedelta(days=30 * (synthetic_window_counter + 1))

            synthetic_ticker = f"SYN_{int(cls)}_{synthetic_window_counter:07d}"

            for step_idx in range(seq_len):
                row_dict = {}
                for col_idx, col in enumerate(TIMEGAN_FEATURE_COLUMNS):
                    row_dict[col] = float(seq_block[step_idx, col_idx])

                row_dict[target_col] = int(cls)

                if entity_col in train.columns:
                    row_dict[entity_col] = synthetic_ticker
                if date_col in train.columns:
                    row_dict[date_col] = pd.to_datetime(ref_date) + pd.Timedelta(days=90 * step_idx)

                generated_rows.append(row_dict)

            synthetic_window_counter += 1

        generated_windows_log.append({
            'class_label': int(cls),
            'windows_requested': int(n_windows),
            'windows_generated': int(sampled_raw.shape[0]),
            'model_source': model_source
        })
        print(f"âœ“ Class {cls}: generated {sampled_raw.shape[0]} windows via {model_source}")
    except Exception as e:
        print(f"âš  Class {cls}: generation failed on {model_source}: {e}")

if len(generated_rows) == 0:
    synthetic_df = pd.DataFrame()
    print("\nâš  No synthetic rows generated")
else:
    synthetic_df = pd.DataFrame(generated_rows)
    
    # ----------------------------------------------------
    # Quality Filter Step
    # ----------------------------------------------------
    print("\n=== Filtering Synthetic Sequences ===")
    
    synthetic_df['predicted_class'] = quality_clf.predict(synthetic_df[TIMEGAN_FEATURE_COLUMNS].fillna(0))
    synthetic_df['is_correct'] = (synthetic_df['predicted_class'] == synthetic_df[target_col]).astype(int)
    
    seq_scores = synthetic_df.groupby(entity_col)['is_correct'].mean()
    
    rebalanced_chunks = []
    for cls, n_rows in samples_per_class.items():
        if int(n_rows) <= 0:
            continue

        cls_rows = synthetic_df[synthetic_df[target_col] == int(cls)]
        if len(cls_rows) == 0:
            continue

        cls_entities = cls_rows[entity_col].unique()
        target_seqs = int(np.ceil(n_rows / seq_len))
        
        # Get scores for these entities
        cls_scores = seq_scores[cls_entities].sort_values(ascending=False)
        
        valid_cls_entities = cls_scores[cls_scores >= 0.5].index.tolist()
        
        if len(valid_cls_entities) >= target_seqs:
            selected_entities = valid_cls_entities[:target_seqs]
            print(f"Class {cls}: Retained {target_seqs} valid sequences (from {len(valid_cls_entities)} valid pool).")
        else:
            selected_entities = cls_scores.index[:target_seqs].tolist()
            num_valid = len(valid_cls_entities)
            print(f"âš  Class {cls}: Only {num_valid} valid sequences. Padded with {len(selected_entities)-num_valid} sub-optimal ones to reach quota.")
            
        cls_rows = cls_rows[cls_rows[entity_col].isin(selected_entities)]
        
        # We must drop predicted_class and is_correct for safety
        cls_rows = cls_rows.drop(columns=['predicted_class', 'is_correct'], errors='ignore')

        rebalanced_chunks.append(cls_rows)

    if len(rebalanced_chunks) > 0:
        synthetic_df = pd.concat(rebalanced_chunks, ignore_index=True)
        synthetic_df = synthetic_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        synthetic_df = pd.DataFrame()

    if len(synthetic_df) > 0:
        synthetic_df = postprocess_synthetic_rows(
            synthetic_df,
            numeric_cols=TIMEGAN_NUMERIC_COLUMNS,
            categorical_cols=TIMEGAN_CATEGORICAL_COLUMNS
        )

    print(f"\nâœ“ Final synthetic rows generated (after quality filter & padding): {len(synthetic_df)}")
    if len(synthetic_df) > 0:
        print("Class distribution (synthetic):")
        print(synthetic_df[target_col].value_counts().sort_index())

TIMEGAN_GENERATION_LOG = generated_windows_log
