print("=" * 70)

print("LOADING SOURCE DATA AND PERFORMING STANDALONE SPLIT")

print("=" * 70)

# Load dataset from source path
data_file = INPUT_PATH / CONFIG['data_file']
df = pd.read_csv(data_file)

print(f"\nâœ“ Dataset loaded: {data_file}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {len(df.columns)}")

if CONFIG['target_column'] not in df.columns:
    raise ValueError(f"Target column '{CONFIG['target_column']}' not found in dataset")

# Parse date column
if CONFIG['date_column'] in df.columns:
    df[CONFIG['date_column']] = pd.to_datetime(df[CONFIG['date_column']], errors='coerce')
    print(f"  âœ“ Parsed {CONFIG['date_column']} to datetime")

def normalize_rating_text(value):
    """Normalize rating string for robust mapping."""
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper().replace(' ', '')

# Normalize target column for encoded labels (expected range 0-22)
if CONFIG.get('target_is_encoded_label', False):
    target_col = CONFIG['target_column']
    target_numeric = pd.to_numeric(df[target_col], errors='coerce')
    non_na_numeric = int(target_numeric.notna().sum())

    if non_na_numeric == 0:
        print("  â  Target appears to be rating text. Applying fallback mapping to labels 0-22...")

        # Full 23-class mapping observed in merged_credit_rating_common.csv (worst -> best).
        # Labels are contiguous in [0, 22].
        ordered_ratings = [
            'D', 'C', 'CC', 'CCC-', 'CCC', 'CCC+',
            'B-', 'B', 'B+',
            'BB-', 'BB', 'BB+',
            'BBB-', 'BBB', 'BBB+',
            'A-', 'A', 'A+',
            'AA-', 'AA', 'AA+', 'AAA'
        ]
        rating_to_label = {rating: idx for idx, rating in enumerate(ordered_ratings)}

        normalized_target = df[target_col].apply(normalize_rating_text)
        mapped_target = normalized_target.map(rating_to_label)

        unknown_mask = mapped_target.isna() & normalized_target.notna()
        unknown_ratings = sorted(normalized_target[unknown_mask].unique().tolist())
        if unknown_ratings:
            sample_unknown = unknown_ratings[:10]
            raise ValueError(
                "Found unmapped rating labels in target: "
                f"{sample_unknown}. Please extend rating_to_label mapping in Cell 8."
            )

        df[target_col] = mapped_target
        print(f"  âœ“ Fallback mapping completed with {int(df[target_col].notna().sum())} labeled rows")
    else:
        before_na = int(df[target_col].isna().sum())
        df[target_col] = target_numeric
        after_na = int(df[target_col].isna().sum())
        if after_na > before_na:
            print(f"  â  Coercion added {after_na - before_na} NaN values in target")

    min_label = int(CONFIG.get('target_min_label', 0))
    max_label = int(CONFIG.get('target_max_label', 22))
    in_range_mask = df[target_col].between(min_label, max_label) | df[target_col].isna()
    out_of_range = int((~in_range_mask).sum())
    if out_of_range > 0:
        print(f"  â  Found {out_of_range} rows with target outside [{min_label}, {max_label}] and will drop them")
    df = df[in_range_mask].copy()

# Remove rows with missing target
initial_rows = len(df)
df = df.dropna(subset=[CONFIG['target_column']])
print(f"  âœ“ Removed {initial_rows - len(df)} rows with missing target")

if CONFIG.get('target_is_encoded_label', False):
    df[CONFIG['target_column']] = df[CONFIG['target_column']].astype(int)
    uniq = sorted(df[CONFIG['target_column']].unique().tolist())
    print(f"  âœ“ Target converted to int labels, unique classes: {len(uniq)}")
    if len(uniq) == 0:
        raise ValueError(
            "Target became empty after normalization/filtering. "
            "Check target mapping and source data in Cell 8."
        )
    print(f"  âœ“ Label range in data: min={min(uniq)}, max={max(uniq)}")

print(f"  âœ“ Final shape: {df.shape}")

# Perform train/val/test split
print(f"\n=== Performing Stratified Split ===")
print(f"  Train: {CONFIG['train_ratio']*100:.0f}%")
print(f"  Val:   {CONFIG['val_ratio']*100:.0f}%")
print(f"  Test:  {CONFIG['test_ratio']*100:.0f}%")

stratify_series = df[CONFIG['target_column']] if CONFIG['stratify'] else None
min_class_count = int(stratify_series.value_counts().min()) if stratify_series is not None else None
if stratify_series is not None and min_class_count < 2:
    print("  â  Some classes have <2 samples, disabling stratified split")
    stratify_series = None

# First split: separate test set
train_val, test = train_test_split(
    df,
    test_size=CONFIG['test_ratio'],
    random_state=CONFIG['random_seed'],
    stratify=stratify_series
)

# Second split: separate train and val
val_ratio_adjusted = CONFIG['val_ratio'] / (1 - CONFIG['test_ratio'])
stratify_train_val = train_val[CONFIG['target_column']] if stratify_series is not None else None
if stratify_train_val is not None and int(stratify_train_val.value_counts().min()) < 2:
    stratify_train_val = None

train, val = train_test_split(
    train_val,
    test_size=val_ratio_adjusted,
    random_state=CONFIG['random_seed'],
    stratify=stratify_train_val
)

print(f"\nâœ“ Split completed:")
print(f"  Train: {len(train)} ({100*len(train)/len(df):.1f}%)")
print(f"  Val:   {len(val)} ({100*len(val)/len(df):.1f}%)")
print(f"  Test:  {len(test)} ({100*len(test)/len(df):.1f}%)")

# Check class distribution
print(f"\n=== Class Distribution ===")
for name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
    print(f"\n{name}:")
    print(split_df[CONFIG['target_column']].value_counts().sort_index())

# Save raw splits for reproducibility before preprocessing
train.to_csv(SPLITS_DIR / 'train_raw.csv', index=False)
val.to_csv(SPLITS_DIR / 'val_raw.csv', index=False)
test.to_csv(SPLITS_DIR / 'test_raw.csv', index=False)
print(f"\nâœ“ Raw splits saved to {SPLITS_DIR}")

preprocessed = False

print(f"\n{'='*70}")
print(f"Data loading complete. Preprocessed: {preprocessed}")
print(f"{'='*70}")
