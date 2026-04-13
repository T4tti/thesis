# Prepare training data for TimeGAN using entity-based sequence windows
if not TIMEGAN_AVAILABLE:
    raise ImportError(f"TimeGAN import failed: {TIMEGAN_IMPORT_ERROR}")

target_col = CONFIG['target_column']
date_col = CONFIG['date_column']
entity_col = CONFIG['entity_column']

train_for_timegan = train.copy()

# Ensure required entity/date columns exist for sequence extraction
if entity_col not in train_for_timegan.columns:
    fallback_entity_col = 'company_name' if 'company_name' in train_for_timegan.columns else None
    if fallback_entity_col is not None:
        train_for_timegan[entity_col] = train_for_timegan[fallback_entity_col].astype(str)
        print(f"â  Missing '{entity_col}', fallback to '{fallback_entity_col}'")
    else:
        train_for_timegan[entity_col] = [f"ENTITY_{i:07d}" for i in range(len(train_for_timegan))]
        print(f"â  Missing '{entity_col}' and 'company_name', generated synthetic entity IDs")

if date_col not in train_for_timegan.columns:
    train_for_timegan[date_col] = pd.Timestamp('2000-01-01')
    print(f"â  Missing '{date_col}', fallback to constant timestamp")

train_for_timegan[date_col] = pd.to_datetime(train_for_timegan[date_col], errors='coerce')
if train_for_timegan[date_col].isna().all():
    train_for_timegan[date_col] = pd.Timestamp('2000-01-01')
    print(f"â  '{date_col}' is fully NaT after parsing, fallback to constant timestamp")
else:
    fill_value = train_for_timegan[date_col].dropna().min()
    train_for_timegan[date_col] = train_for_timegan[date_col].fillna(fill_value)

# Keep only features available in current train schema
TIMEGAN_FEATURE_COLUMNS = [c for c in (FEATURE_NUMERIC + FEATURE_CATEGORICAL) if c in train_for_timegan.columns]
TIMEGAN_NUMERIC_COLUMNS = [c for c in FEATURE_NUMERIC if c in TIMEGAN_FEATURE_COLUMNS]
TIMEGAN_CATEGORICAL_COLUMNS = [c for c in FEATURE_CATEGORICAL if c in TIMEGAN_FEATURE_COLUMNS]

if len(TIMEGAN_FEATURE_COLUMNS) == 0:
    raise ValueError("No feature columns available for TimeGAN sequence modeling")

# TimeGAN uses sigmoid output; normalize all feature columns to [0, 1]
TIMEGAN_SCALER = {}
timegan_norm_frame = train_for_timegan.copy()

for col in TIMEGAN_FEATURE_COLUMNS:
    col_series = pd.to_numeric(timegan_norm_frame[col], errors='coerce').fillna(0.0)
    col_min = float(col_series.min())
    col_max = float(col_series.max())

    if not np.isfinite(col_min) or not np.isfinite(col_max):
        col_min, col_max = 0.0, 1.0

    if abs(col_max - col_min) < 1e-12:
        norm_vals = np.zeros(len(col_series), dtype=float)
    else:
        norm_vals = (col_series - col_min) / (col_max - col_min)

    timegan_norm_frame[col] = np.clip(norm_vals, 0.0, 1.0)
    TIMEGAN_SCALER[col] = {'min': col_min, 'max': col_max}

def build_entity_windows(df, feature_cols, target_col, entity_col, date_col, seq_len, stride, min_history):
    windows = []
    meta_records = []

    grouped = df.groupby(entity_col, dropna=False)
    for entity_value, group_df in grouped:
        grp = group_df.sort_values(date_col).reset_index(drop=True)
        history_len = len(grp)

        if history_len < min_history:
            continue

        if history_len >= seq_len:
            start_points = range(0, history_len - seq_len + 1, stride)
            for start_idx in start_points:
                chunk = grp.iloc[start_idx:start_idx + seq_len].copy()
                windows.append(chunk[feature_cols].to_numpy(dtype=float))

                label_mode = pd.Series(chunk[target_col]).mode(dropna=True)
                label_value = int(label_mode.iloc[0]) if len(label_mode) > 0 else int(chunk[target_col].iloc[-1])

                meta_records.append({
                    'entity': str(entity_value),
                    'start_idx': int(start_idx),
                    'end_idx': int(start_idx + seq_len - 1),
                    'start_date': pd.to_datetime(chunk[date_col].iloc[0]),
                    'end_date': pd.to_datetime(chunk[date_col].iloc[-1]),
                    'window_label': label_value,
                    'is_padded': 0
                })
        else:
            # Optional padding for short-but-usable history
            chunk = grp.copy()
            pad_count = seq_len - history_len
            pad_rows = pd.concat([chunk.iloc[[-1]].copy() for _ in range(pad_count)], ignore_index=True)
            padded = pd.concat([chunk, pad_rows], ignore_index=True)
            windows.append(padded[feature_cols].to_numpy(dtype=float))

            label_mode = pd.Series(padded[target_col]).mode(dropna=True)
            label_value = int(label_mode.iloc[0]) if len(label_mode) > 0 else int(padded[target_col].iloc[-1])

            meta_records.append({
                'entity': str(entity_value),
                'start_idx': 0,
                'end_idx': int(seq_len - 1),
                'start_date': pd.to_datetime(padded[date_col].iloc[0]),
                'end_date': pd.to_datetime(padded[date_col].iloc[-1]),
                'window_label': label_value,
                'is_padded': 1
            })

    if len(windows) > 0:
        windows_array = np.asarray(windows, dtype=np.float32)
    else:
        windows_array = np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)

    meta_df = pd.DataFrame(meta_records)
    return windows_array, meta_df

TIMEGAN_SEQUENCE_WINDOWS, TIMEGAN_SEQUENCE_META = build_entity_windows(
    df=timegan_norm_frame,
    feature_cols=TIMEGAN_FEATURE_COLUMNS,
    target_col=target_col,
    entity_col=entity_col,
    date_col=date_col,
    seq_len=int(CONFIG['seq_len']),
    stride=max(1, int(CONFIG['sequence_stride'])),
    min_history=max(1, int(CONFIG['min_history_for_company']))
)

if TIMEGAN_SEQUENCE_WINDOWS.shape[0] == 0:
    raise ValueError(
        "No sequence windows were created for TimeGAN. "
        "Check entity/date columns, seq_len, and min_history_for_company."
    )

SEQUENCE_CLASS_COUNTS = TIMEGAN_SEQUENCE_META['window_label'].value_counts().sort_index() if len(TIMEGAN_SEQUENCE_META) > 0 else pd.Series(dtype=int)

print("âœ“ TimeGAN sequence data prepared")
print(f"Entities: {timegan_norm_frame[entity_col].nunique()}")
print(f"Feature columns for TimeGAN: {len(TIMEGAN_FEATURE_COLUMNS)}")
print(f"Numeric features: {len(TIMEGAN_NUMERIC_COLUMNS)}")
print(f"Categorical features: {len(TIMEGAN_CATEGORICAL_COLUMNS)}")
print(f"Sequence windows: {TIMEGAN_SEQUENCE_WINDOWS.shape[0]}")
print(f"Window shape: {TIMEGAN_SEQUENCE_WINDOWS.shape}")
print(f"Classes in windows: {SEQUENCE_CLASS_COUNTS.to_dict()}")