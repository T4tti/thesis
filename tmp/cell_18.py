def build_timegan_model():
    """Create a tsgm TimeGAN instance from CONFIG model parameters."""
    return TimeGAN(
        seq_len=int(CONFIG['seq_len']),
        module=str(CONFIG.get('timegan_module', 'gru')),
        hidden_dim=int(CONFIG.get('timegan_latent_dim', 24)),
        n_features=int(len(TIMEGAN_FEATURE_COLUMNS)),
        n_layers=int(CONFIG.get('timegan_n_layers', 3)),
        batch_size=int(CONFIG['timegan_batch_size']),
        gamma=float(CONFIG['timegan_gamma'])
    )

def train_timegan_on_windows(window_data):
    """Train a TimeGAN model on pre-built sequence windows."""
    if len(window_data) == 0:
        raise ValueError("Cannot train TimeGAN on empty window data")

    model = build_timegan_model()
    model.compile()
    model.fit(data=np.asarray(window_data, dtype=np.float32), epochs=int(CONFIG['timegan_train_steps']))
    return model

print("\n=== Training TimeGAN Models ===")
print(f"Total sequence windows: {TIMEGAN_SEQUENCE_WINDOWS.shape[0]}")
print(f"Sequence length: {CONFIG['seq_len']}")
print(f"Features per step: {len(TIMEGAN_FEATURE_COLUMNS)}")
print(f"Train steps: {CONFIG['timegan_train_steps']}")

TIMEGAN_MODELS = {}
TIMEGAN_TRAINING_REGISTRY = {
    'global': {},
    'per_class': {},
    'train_steps': int(CONFIG['timegan_train_steps']),
    'seq_len': int(CONFIG['seq_len']),
    'feature_columns': TIMEGAN_FEATURE_COLUMNS,
    'window_count': int(TIMEGAN_SEQUENCE_WINDOWS.shape[0])
}

# Train fallback model on PRIORITY CLASSES ONLY
priority_classes_mask = TIMEGAN_SEQUENCE_META['window_label'].isin([0, 1, 2, 3, 4, 5, 18, 19, 20, 21])
priority_windows = TIMEGAN_SEQUENCE_WINDOWS[priority_classes_mask]
print(f"[Global] Training Fallback TimeGAN on PRIORITY classes only ({len(priority_windows)} windows)...")
try:
    global_model = train_timegan_on_windows(priority_windows)
    TIMEGAN_MODELS['global'] = global_model
    TIMEGAN_TRAINING_REGISTRY['global'] = {'status': 'trained', 'windows_used': int(priority_windows.shape[0])}
    print("[Global] Fallback TimeGAN training complete")
except Exception as e:
    TIMEGAN_TRAINING_REGISTRY['global'] = {'status': 'failed', 'error': str(e)}
    print(f"[Global] Fallback TimeGAN training failed: {e}")

# Train per-class models for classes with enough windows
# Focus ONLY on extremely rare/priority classes
min_windows = 1
eligible_classes = [0, 1, 2, 3, 4, 5, 18, 19, 20, 21]

print(f"\nEligible classes for per-class TimeGAN (min_windows={min_windows}): {eligible_classes}")

for cls in eligible_classes:
    class_mask = TIMEGAN_SEQUENCE_META['window_label'].to_numpy() == int(cls)
    class_windows = TIMEGAN_SEQUENCE_WINDOWS[class_mask]

    entry = {
        'windows_used': int(class_windows.shape[0]),
        'status': 'not_started'
    }

    try:
        print(f"[Class {cls}] Training TimeGAN on {len(class_windows)} windows...")
        cls_model = train_timegan_on_windows(class_windows)
        TIMEGAN_MODELS[f'class_{cls}'] = cls_model

        cls_model_path = MODELS_DIR / f'timegan_class_{cls}.pkl'
        save_status = 'not_saved'
        save_error = None
        try:
            cls_model.save(str(cls_model_path))
            save_status = 'saved'
        except Exception as save_e:
            save_error = str(save_e)

        entry.update({
            'status': 'trained',
            'model_path': str(cls_model_path),
            'save_status': save_status,
            'save_error': save_error
        })
        print(f"[Class {cls}] TimeGAN training complete")
    except Exception as e:
        entry.update({
            'status': 'failed',
            'error': str(e)
        })
        print(f"[Class {cls}] TimeGAN training failed: {e}")

    TIMEGAN_TRAINING_REGISTRY['per_class'][str(cls)] = entry

trained_per_class = [k for k, v in TIMEGAN_TRAINING_REGISTRY['per_class'].items() if v.get('status') == 'trained']
print(f"\nPer-class models trained: {trained_per_class}")