# Utility helpers for TimeGAN pipeline
def inverse_timegan_scale(sequence_array, scaler_dict, feature_cols):
    """Inverse min-max scaling from [0, 1] back to train feature ranges."""
    arr = np.asarray(sequence_array, dtype=float).copy()
    for col_idx, col in enumerate(feature_cols):
        col_min = float(scaler_dict[col]['min'])
        col_max = float(scaler_dict[col]['max'])
        span = col_max - col_min
        if abs(span) < 1e-12:
            arr[:, :, col_idx] = col_min
        else:
            arr[:, :, col_idx] = arr[:, :, col_idx] * span + col_min
    return arr

def safe_timegan_sample(model, n_samples):
    """Safely generate `n_samples` sequence windows from a trained TimeGAN model."""
    raw_samples = model.generate(int(n_samples))
    arr = np.asarray(raw_samples, dtype=float)

    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])

    if arr.ndim != 3:
        return np.empty((0, int(CONFIG['seq_len']), len(TIMEGAN_FEATURE_COLUMNS)), dtype=float)

    return arr

def postprocess_synthetic_rows(df, numeric_cols, categorical_cols):
    """Clip/round generated rows to keep schema compatible with preprocessed train data."""
    out = df.copy()

    for col in numeric_cols:
        if col in out.columns and col in train.columns:
            low = float(train[col].quantile(0.001))
            high = float(train[col].quantile(0.999))
            out[col] = pd.to_numeric(out[col], errors='coerce').clip(low, high)

    for col in categorical_cols:
        if col in out.columns and col in train.columns:
            if pd.api.types.is_integer_dtype(train[col]):
                cmin = int(train[col].min())
                cmax = int(train[col].max())
                out[col] = pd.to_numeric(out[col], errors='coerce').round().clip(cmin, cmax).fillna(cmin).astype(int)
            else:
                out[col] = out[col].astype(str)

    return out

print("TimeGAN utility helpers defined")