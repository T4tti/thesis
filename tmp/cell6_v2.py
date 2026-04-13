# Configure paths for Kaggle environment
INPUT_PATH = Path('/kaggle/input/datasets/tailength/corporate-credit-rating')
OUTPUT_PATH = Path('/kaggle/working')

SPLITS_DIR = OUTPUT_PATH / 'splits'
MODELS_DIR = OUTPUT_PATH / 'models' / 'timegan'
REPORTS_DIR = OUTPUT_PATH / 'reports'
for d in [SPLITS_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- CONFIGURATION v2 (improved augmentation strategy) ----
# Changes vs v1:
# 1. max_synthetic_ratio: 0.45 -> 0.60 (more budget for minority)
# 2. balance_strategy: 'log_ratio' (geometric mean, softer than median_flat)
# 3. priority_min_count: 220 -> 300
# 4. priority_boost_multiplier: 2.0 -> 1.5 (less aggressive, more stable)
# 5. min_windows_per_class_for_timegan: 1 -> 10 (guard against mode collapse)
# 6. enable_sector_conditional: True (sector-level fallback model)
# 7. js_divergence_threshold: 0.20 (new quality filter)
# 8. enable_ordinal_mixup_fallback: True (gap fill)
# 9. enable_hybrid_rebalancing: True (was False)
# 10. majority_cap_percentile: 80 (was 75), minority_floor: 150 (was 200)
CONFIG = {
    'random_seed': RANDOM_SEED,
    'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1, 'stratify': True,
    'data_file': 'merged_credit_rating_common.csv',
    'target_column': 'rating_detail',
    'target_is_encoded_label': True, 'target_min_label': 0, 'target_max_label': 21,
    'date_column': 'rating_date', 'entity_column': 'ticker',
    # Sequence
    'seq_len': 4, 'sequence_stride': 1, 'min_history_for_company': 2,
    'timegan_use_gpu': True,
    # Minority support v2
    'priority_class_labels': [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21],
    'priority_boost_multiplier': 1.5,
    'priority_min_count': 300,
    'priority_max_oversample_factor': 150.0,
    'max_synthetic_ratio': 0.60,
    # Balance strategy: 'log_ratio' | 'sqrt_flat' | 'median_flat'
    'balance_strategy': 'log_ratio',
    # TimeGAN hyper-params (unchanged)
    'timegan_train_steps': 500, 'timegan_batch_size': 128,
    'timegan_learning_rate': 5e-4, 'timegan_noise_dim': 32,
    'timegan_layers_dim': 128, 'timegan_latent_dim': 24,
    'timegan_gamma': 1, 'timegan_module': 'gru', 'timegan_n_layers': 3,
    # Min windows guard (was 1, now 10 to avoid mode collapse)
    'min_windows_per_class_for_timegan': 10,
    # Sector-conditional fallback
    'sector_column': 'sector', 'enable_sector_conditional': True,
    # Quality filters
    'quality_quantile_low': 0.005, 'quality_quantile_high': 0.995,
    'js_divergence_threshold': 0.20,
    'rf_quality_threshold': 0.50,
    # Ordinal MixUp fallback (new)
    'enable_ordinal_mixup_fallback': True,
    'mixup_alpha': 0.3, 'mixup_adjacent_only': True,
    # Hybrid rebalancing v2 (ENABLED by default, was False)
    'enable_hybrid_rebalancing': True,
    'majority_cap_percentile': 80,
    'minority_floor': 150,
    'timestamp': datetime.now().isoformat()
}

timegan_gpu_runtime_ok = False
timegan_gpu_runtime_note = None
if CONFIG['timegan_use_gpu'] and tf is not None:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            _ = tf.matmul(tf.random.uniform((64, 64)), tf.random.uniform((64, 64)))
            timegan_gpu_runtime_ok = True
            timegan_gpu_runtime_note = f"GPU OK: {[g.name for g in gpus]}"
        else:
            CONFIG['timegan_use_gpu'] = False
            timegan_gpu_runtime_note = "No GPU found, CPU fallback"
    except Exception as e:
        CONFIG['timegan_use_gpu'] = False
        timegan_gpu_runtime_note = f"GPU check failed: {e}"
else:
    CONFIG['timegan_use_gpu'] = False
    timegan_gpu_runtime_note = "CPU mode" if tf is not None else "TF not available"

print("Configuration v2 loaded.")
print(json.dumps({k: v for k, v in CONFIG.items() if k != 'timestamp'}, indent=2))
print(timegan_gpu_runtime_note)
if not TIMEGAN_AVAILABLE:
    print("WARNING: TimeGAN not importable.")
with open(OUTPUT_PATH / 'config_timegan.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
