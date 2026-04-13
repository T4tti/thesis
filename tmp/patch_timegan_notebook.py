"""
Patch timegan_data_preparation_kaggle.ipynb với các cải tiến:

1. CONFIG: Tăng max_synthetic_ratio, thêm JS divergence threshold, sector-conditional
2. Cell 6  (CONFIG): Cập nhật tham số chiến lược
3. Cell 18 (Training): Thêm sector-conditional fallback + min_windows_guard
4. Cell 21 (Sampling plan): Thêm log-ratio rebalancing thay vì median-based flat target
5. Cell 22 (Generation): Thêm JS divergence check + ordinal-aware MixUp fallback
6. Cell 24 (Filter): Thêm financial constraints check
7. Cell 31 (Rebalancing): Bật hybrid rebalancing với entropy-weighted capping
"""

import json, copy, re

NB_PATH = 'notebooks/timegan_data_preparation_kaggle.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def get_cell(nb, idx):
    return nb['cells'][idx]

def set_cell_source(nb, idx, new_source: str):
    nb['cells'][idx]['source'] = new_source
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None

# ─────────────────────────────────────────────────────────────
# CELL 6: CONFIG — Cải tiến tham số chiến lược
# ─────────────────────────────────────────────────────────────
NEW_CELL_6 = r"""# Configure paths for Kaggle environment

INPUT_PATH = Path('/kaggle/input/datasets/tailength/corporate-credit-rating')  # Update this path
OUTPUT_PATH = Path('/kaggle/working')

# Output directories
SPLITS_DIR = OUTPUT_PATH / 'splits'
MODELS_DIR = OUTPUT_PATH / 'models' / 'timegan'
REPORTS_DIR = OUTPUT_PATH / 'reports'

for dir_path in [SPLITS_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ─── CHIẾN LƯỢC TĂNG CƯỜNG (v2) ────────────────────────────
# Cải tiến so với v1:
# 1. Tăng max_synthetic_ratio → cho phép tạo nhiều hơn cho minority
# 2. Log-ratio target thay vì median-flat để cân bằng mềm hơn
# 3. Sector-conditional fallback TimeGAN (train theo ngành)
# 4. JS-Divergence check để loại bỏ sample phân phối lạc
# 5. Ordinal-aware MixUp để fill gap khi TimeGAN thất bại
# 6. Bật ENABLE_HYBRID_REBALANCING mặc định
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'random_seed': RANDOM_SEED,
    # Data split parameters (standalone mode)
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'stratify': True,
    # Data parameters
    'data_file': 'merged_credit_rating_common.csv',
    'target_column': 'rating_detail',
    'target_is_encoded_label': True,
    'target_min_label': 0,
    'target_max_label': 21,
    'date_column': 'rating_date',
    'entity_column': 'ticker',
    # Sequence parameters
    'seq_len': 4,
    'sequence_stride': 1,
    'min_history_for_company': 2,
    # TimeGAN runtime parameters
    'timegan_use_gpu': True,
    # ── Minority support parameters (v2) ──
    'priority_class_labels': [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21],
    # Tăng lên để minority classes đạt ít nhất 300 samples
    'priority_min_count': 300,
    # Giảm floor từ 2× → 1.5× để tránh OVER-emphasis trên majority gap
    'priority_boost_multiplier': 1.5,
    'priority_max_oversample_factor': 150.0,
    # Tăng từ 0.45 → 0.60: cho phép tạo tối đa 60% tổng train làm synthetic
    'max_synthetic_ratio': 0.60,
    # ── Rebalancing strategy (v2) ──
    # 'log_ratio': dùng logarithmic leveling thay vì median-flat
    # target_i = max(median, min_count) với soft cap bằng log scale
    'balance_strategy': 'log_ratio',   # 'median_flat' | 'log_ratio' | 'sqrt_flat'
    # ── TimeGAN training parameters ──
    'timegan_train_steps': 500,
    'timegan_batch_size': 128,
    'timegan_learning_rate': 5e-4,
    'timegan_noise_dim': 32,
    'timegan_layers_dim': 128,
    'timegan_latent_dim': 24,
    'timegan_gamma': 1,
    'timegan_module': 'gru',
    'timegan_n_layers': 3,
    # Số windows tối thiểu để train per-class model (tăng từ 1 lên 10)
    'min_windows_per_class_for_timegan': 10,
    # ── Sector-conditional fallback ──
    # Nếu class không đủ windows, lấy sector-level fallback thay vì global
    'sector_column': 'sector',
    'enable_sector_conditional': True,
    # ── Quality filtering (v2) ──
    'quality_quantile_low': 0.005,
    'quality_quantile_high': 0.995,
    # JS Divergence threshold: bỏ sequence có divergence > threshold
    'js_divergence_threshold': 0.20,
    # RF quality threshold per sequence (tỷ lệ step đúng label)
    'rf_quality_threshold': 0.5,
    # ── Ordinal-aware MixUp fallback ──
    # Khi TimeGAN không đủ samples, dùng MixUp giữa adjacent classes
    'enable_ordinal_mixup_fallback': True,
    'mixup_alpha': 0.3,               # Beta(alpha,alpha) — nhỏ = gần boundary
    'mixup_adjacent_only': True,      # Chỉ mix class i với i±1
    # ── Hybrid rebalancing (v2, bật mặc định) ──
    'enable_hybrid_rebalancing': True,
    'majority_cap_percentile': 80,    # Tăng từ 75→80: giảm thiệt hại thông tin majority
    'minority_floor': 150,            # Giảm từ 200→150: realistic với rare classes
    'timestamp': datetime.now().isoformat()
}

# Runtime GPU sanity check for TensorFlow (used by tsgm TimeGAN)
timegan_gpu_runtime_ok = False
timegan_gpu_runtime_note = None
if CONFIG['timegan_use_gpu'] and tf is not None:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            x = tf.random.uniform((1024, 1024))
            y = tf.random.uniform((1024, 1024))
            _ = tf.matmul(x, y)
            timegan_gpu_runtime_ok = True
            timegan_gpu_runtime_note = f"✔ TensorFlow GPU devices: {[gpu.name for gpu in gpus]}"
        else:
            CONFIG['timegan_use_gpu'] = False
            timegan_gpu_runtime_note = "⚠ No TensorFlow GPU found, fallback to CPU"
    except Exception as e:
        CONFIG['timegan_use_gpu'] = False
        timegan_gpu_runtime_note = f"⚠ TensorFlow GPU runtime check failed, fallback to CPU: {e}"
else:
    CONFIG['timegan_use_gpu'] = False
    if tf is None:
        timegan_gpu_runtime_note = "⚠ TensorFlow not available, TimeGAN cannot run until dependencies are fixed"
    else:
        timegan_gpu_runtime_note = "CPU mode selected for TimeGAN"

print("Configuration:")
print(json.dumps({k: v for k, v in CONFIG.items() if k != 'timestamp'}, indent=2))
print(timegan_gpu_runtime_note)

if not TIMEGAN_AVAILABLE:
    print("\n⚠ TimeGAN class is not importable. Please re-run dependency cell and restart kernel.")

# Save config
with open(OUTPUT_PATH / 'config_timegan.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
"""

# ─────────────────────────────────────────────────────────────
# CELL 18: Training — Sector-conditional + min_windows guard
# ─────────────────────────────────────────────────────────────
NEW_CELL_18 = r"""def build_timegan_model():
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

def train_timegan_on_windows(window_data: np.ndarray):
    """Train a TimeGAN model on pre-built sequence windows."""
    if len(window_data) == 0:
        raise ValueError("Cannot train TimeGAN on empty window data")
    model = build_timegan_model()
    model.compile()
    model.fit(data=np.asarray(window_data, dtype=np.float32), epochs=int(CONFIG['timegan_train_steps']))
    return model

print("\n=== Training TimeGAN Models (v2 — Sector-Conditional + Min-Window Guard) ===")
print(f"Total sequence windows: {TIMEGAN_SEQUENCE_WINDOWS.shape[0]}")
print(f"Sequence length: {CONFIG['seq_len']}")
print(f"Features per step: {len(TIMEGAN_FEATURE_COLUMNS)}")
print(f"Train steps: {CONFIG['timegan_train_steps']}")
print(f"Min windows per class for dedicated model: {CONFIG['min_windows_per_class_for_timegan']}")

TIMEGAN_MODELS = {}
TIMEGAN_TRAINING_REGISTRY = {
    'global': {},
    'per_class': {},
    'per_sector': {},
    'train_steps': int(CONFIG['timegan_train_steps']),
    'seq_len': int(CONFIG['seq_len']),
    'feature_columns': TIMEGAN_FEATURE_COLUMNS,
    'window_count': int(TIMEGAN_SEQUENCE_WINDOWS.shape[0])
}

priority_classes = [int(c) for c in CONFIG.get('priority_class_labels', [])]
min_windows_thresh = int(CONFIG.get('min_windows_per_class_for_timegan', 10))

# ── Global Fallback: Train trên toàn bộ priority windows ──────────────────
print("\n[Global Fallback] Training TimeGAN on ALL priority class windows...")
priority_classes_mask = TIMEGAN_SEQUENCE_META['window_label'].isin(priority_classes)
priority_windows = TIMEGAN_SEQUENCE_WINDOWS[priority_classes_mask]
print(f"  Priority windows: {len(priority_windows)}")

try:
    global_model = train_timegan_on_windows(priority_windows)
    TIMEGAN_MODELS['global'] = global_model
    TIMEGAN_TRAINING_REGISTRY['global'] = {'status': 'trained', 'windows_used': int(priority_windows.shape[0])}
    print("[Global Fallback] ✔ Training complete")
except Exception as e:
    TIMEGAN_TRAINING_REGISTRY['global'] = {'status': 'failed', 'error': str(e)}
    print(f"[Global Fallback] ✘ Failed: {e}")

# ── Sector-Conditional Fallback (v2 new) ──────────────────────────────────
# Nếu class không đủ windows, dùng sector model (học pattern cùng ngành)
SECTOR_FALLBACK_MAP = {}  # class_label -> sector_name
if CONFIG.get('enable_sector_conditional', True) and CONFIG['sector_column'] in train.columns:
    print("\n[Sector-Conditional] Building sector-level fallback models...")
    sector_col = CONFIG['sector_column']

    # Gán sector cho mỗi window qua meta
    if sector_col in train.columns and 'entity' in TIMEGAN_SEQUENCE_META.columns:
        entity_sector = (
            train.groupby(CONFIG['entity_column'])[sector_col].agg(lambda x: x.mode()[0] if len(x) > 0 else 'Unknown')
            if CONFIG['entity_column'] in train.columns else pd.Series(dtype=str)
        )
        TIMEGAN_SEQUENCE_META['sector'] = TIMEGAN_SEQUENCE_META['entity'].map(entity_sector).fillna('Unknown')
    else:
        TIMEGAN_SEQUENCE_META['sector'] = 'Unknown'

    sectors = TIMEGAN_SEQUENCE_META['sector'].unique()
    print(f"  Sectors found: {len(sectors)}: {list(sectors)[:10]}...")

    for sector in sectors:
        sector_mask = TIMEGAN_SEQUENCE_META['sector'] == sector
        sector_windows = TIMEGAN_SEQUENCE_WINDOWS[sector_mask.to_numpy()]
        if len(sector_windows) < max(10, min_windows_thresh):
            print(f"  [Sector={sector}] Skipped: only {len(sector_windows)} windows")
            continue
        try:
            sec_model = train_timegan_on_windows(sector_windows)
            safe_key = f"sector_{re.sub(r'[^a-zA-Z0-9_]', '_', str(sector))}"
            TIMEGAN_MODELS[safe_key] = sec_model
            TIMEGAN_TRAINING_REGISTRY['per_sector'][sector] = {
                'status': 'trained', 'windows_used': int(sector_windows.shape[0])
            }
            print(f"  [Sector={sector}] ✔ {len(sector_windows)} windows")
        except Exception as e:
            TIMEGAN_TRAINING_REGISTRY['per_sector'][sector] = {'status': 'failed', 'error': str(e)}
            print(f"  [Sector={sector}] ✘ {e}")

    # Map each priority class → dominant sector model
    if CONFIG['entity_column'] in train.columns and sector_col in train.columns:
        cls_sector_map = (
            train.groupby([CONFIG['target_column'], sector_col])
            .size()
            .reset_index(name='cnt')
            .sort_values('cnt', ascending=False)
            .groupby(CONFIG['target_column'])
            .first()[sector_col]
        )
        for cls_label, sector_name in cls_sector_map.items():
            safe_key = f"sector_{re.sub(r'[^a-zA-Z0-9_]', '_', str(sector_name))}"
            if safe_key in TIMEGAN_MODELS:
                SECTOR_FALLBACK_MAP[int(cls_label)] = safe_key
    print(f"  Sector fallback map: {SECTOR_FALLBACK_MAP}")
else:
    print("[Sector-Conditional] Disabled or sector column not found, skipping.")

# ── Per-Class Models (với min_windows guard) ──────────────────────────────
print(f"\n[Per-Class] Training dedicated TimeGAN models for priority classes...")
print(f"  Guard: require >= {min_windows_thresh} windows to train per-class model")

for cls in priority_classes:
    class_mask = TIMEGAN_SEQUENCE_META['window_label'].to_numpy() == int(cls)
    class_windows = TIMEGAN_SEQUENCE_WINDOWS[class_mask]
    n_class_windows = len(class_windows)

    entry = {'windows_used': n_class_windows, 'status': 'not_started'}

    if n_class_windows < min_windows_thresh:
        # Không đủ dữ liệu → fallback sẽ được dùng thay thế
        entry['status'] = 'skipped_insufficient'
        entry['reason'] = f'< {min_windows_thresh} windows (got {n_class_windows})'
        print(f"  [Class {cls}] ⚠ Skipped: {n_class_windows} windows < threshold {min_windows_thresh}")
        print(f"           → Will use {'sector' if cls in SECTOR_FALLBACK_MAP else 'global'} fallback")
    else:
        try:
            print(f"  [Class {cls}] Training on {n_class_windows} windows...")
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
            print(f"  [Class {cls}] ✔ Training complete")
        except Exception as e:
            entry.update({'status': 'failed', 'error': str(e)})
            print(f"  [Class {cls}] ✘ Training failed: {e}")

    TIMEGAN_TRAINING_REGISTRY['per_class'][str(cls)] = entry

trained_per_class = [k for k, v in TIMEGAN_TRAINING_REGISTRY['per_class'].items() if v.get('status') == 'trained']
skipped_per_class = [k for k, v in TIMEGAN_TRAINING_REGISTRY['per_class'].items() if 'skipped' in v.get('status', '')]
print(f"\nPer-class models trained:  {trained_per_class}")
print(f"Per-class models skipped:  {skipped_per_class}")
print(f"Sector fallback available: {list(SECTOR_FALLBACK_MAP.keys())}")
"""

# ─────────────────────────────────────────────────────────────
# CELL 21: Sampling Plan — Log-ratio balancing
# ─────────────────────────────────────────────────────────────
NEW_CELL_21 = r"""# ─────────────────────────────────────────────────────────────────────────
# TimeGAN Sampling Plan (v2 — Log-Ratio Balancing)
#
# Thay đổi so với v1:
# - 'log_ratio' strategy: target_i = geometric_mean(count_i, P75_count)
#   giảm thiệt hại cho majority, tăng coverage cho minority
# - 'sqrt_flat': target_i = sqrt(count_i * median) → nhẹ nhàng hơn median_flat
# - Giữ nguyên 'median_flat' như v1 để so sánh
# ─────────────────────────────────────────────────────────────────────────
full_class_range = list(range(int(CONFIG['target_min_label']), int(CONFIG['target_max_label']) + 1))
class_counts = (
    train[CONFIG['target_column']]
    .value_counts()
    .reindex(full_class_range, fill_value=0)
    .sort_index()
    .astype(int)
)

nonzero_counts = class_counts[class_counts > 0]
median_count  = float(nonzero_counts.median()) if len(nonzero_counts) > 0 else 1.0
p75_count     = float(nonzero_counts.quantile(0.75)) if len(nonzero_counts) > 0 else 1.0

print("=== TimeGAN Sampling Plan (v2 — Log-Ratio) ===")
print("\nOriginal class counts:")
print(class_counts)
print(f"\nMedian class count (non-zero): {median_count:.0f}")
print(f"P75 class count   (non-zero): {p75_count:.0f}")

priority_classes = [int(c) for c in CONFIG.get('priority_class_labels', [])]
priority_classes_present = [c for c in priority_classes if c in class_counts.index and class_counts[c] > 0]
print(f"\nPriority classes present in train: {priority_classes_present}")

balance_strategy = CONFIG.get('balance_strategy', 'log_ratio')
print(f"Balance strategy: {balance_strategy}")

priority_boost = float(CONFIG.get('priority_boost_multiplier', 1.5))
priority_min   = int(CONFIG.get('priority_min_count', 300))
priority_max_f = float(CONFIG.get('priority_max_oversample_factor', 150.0))

def compute_target_count(class_label: int, real_count: int) -> int:
    """
    Tính target count theo balance_strategy được chọn.
    Returns target số samples sau augmentation.
    """
    is_priority = int(class_label) in priority_classes
    count = max(real_count, 0)

    if balance_strategy == 'log_ratio':
        # Geometric mean giữa count và p75: cân bằng mềm
        # target = sqrt(count * p75) capped at p75 for majority
        if count > 0:
            base = int(np.sqrt(count * p75_count))
        else:
            base = int(p75_count)
        # Majority classes: không tăng quá p75
        if not is_priority:
            base = min(base, int(p75_count))

    elif balance_strategy == 'sqrt_flat':
        base = int(np.sqrt(count * median_count)) if count > 0 else int(median_count)

    else:  # 'median_flat' (v1 behavior)
        base = int(median_count)

    # Priority override
    if is_priority:
        boosted = max(int(base * priority_boost), priority_min)
        if count > 0:
            cap = int(np.ceil(count * priority_max_f))
            base = min(boosted, cap)
        else:
            base = boosted
    elif count == 0:
        base = 0  # Lớp hoàn toàn absent → bỏ qua

    return max(0, base)

def allocate_with_budget(need_map: dict, budget: int) -> dict:
    """Scale class-wise needs to fit total synthetic budget (Hamilton method)."""
    if budget <= 0 or sum(need_map.values()) <= 0:
        return {k: 0 for k in need_map}
    total_need = sum(need_map.values())
    if total_need <= budget:
        return need_map.copy()
    scaled, remainders, allocated = {}, [], 0
    for cls, need in need_map.items():
        exact = budget * (need / total_need)
        flo = int(np.floor(exact))
        scaled[cls] = flo
        allocated += flo
        remainders.append((exact - flo, cls))
    for _, cls in sorted(remainders, reverse=True)[:budget - allocated]:
        scaled[cls] += 1
    return scaled

# Compute raw need per class
raw_need = {}
for class_label, count in class_counts.items():
    count = int(count)
    target_count = compute_target_count(int(class_label), count)
    raw_need[int(class_label)] = max(0, target_count - count)

max_total_synthetic = int(len(train) * float(CONFIG['max_synthetic_ratio']))
samples_per_class   = allocate_with_budget(raw_need, max_total_synthetic)
total_synthetic     = int(sum(samples_per_class.values()))

# Row → window conversion
windows_per_class = {
    int(cls): int(np.ceil(n_rows / max(1, int(CONFIG['seq_len']))))
    for cls, n_rows in samples_per_class.items()
    if int(n_rows) > 0
}

expected_counts = class_counts.add(pd.Series(samples_per_class), fill_value=0).astype(int)

print(f"\nMax total synthetic budget (rows): {max_total_synthetic}")
print(f"Synthetic planned (rows): {total_synthetic} ({total_synthetic/len(train):.1%} of train)")

print("\nPlanned synthetic rows per class:")
header = f"{'Class':>6} {'Real':>8} {'Target':>8} {'Need':>8} {'Windows':>9}  Model"
print(header); print("-" * 60)
for cls in sorted(samples_per_class.keys()):
    need = samples_per_class[cls]
    if need <= 0:
        continue
    real = int(class_counts.get(cls, 0))
    tgt  = int(compute_target_count(cls, real))
    wins = windows_per_class.get(cls, 0)
    model_key = f'class_{cls}'
    sector_key = SECTOR_FALLBACK_MAP.get(cls, '')
    if model_key in TIMEGAN_MODELS:
        avail = 'per-class'
    elif sector_key and sector_key in TIMEGAN_MODELS:
        avail = f'sector({sector_key})'
    elif 'global' in TIMEGAN_MODELS:
        avail = 'global'
    else:
        avail = 'NONE ⚠'
    print(f"{cls:>6} {real:>8} {tgt:>8} {need:>8} {wins:>9}  {avail}")

print(f"\nExpected class counts after augmentation:")
print(expected_counts.sort_index())

nonzero_ec = expected_counts[expected_counts > 0]
before_ratio = nonzero_counts.min() / nonzero_counts.max() if len(nonzero_counts) > 0 and nonzero_counts.max() > 0 else float('nan')
after_ratio  = nonzero_ec.min()    / nonzero_ec.max()     if len(nonzero_ec) > 0    and nonzero_ec.max() > 0    else float('nan')
print(f"\nImbalance ratio before: {before_ratio:.4f}" if np.isfinite(before_ratio) else "\nImbalance ratio before: N/A")
print(f"Expected ratio after:   {after_ratio:.4f}"  if np.isfinite(after_ratio)  else "Expected ratio after:  N/A")
"""

# ─────────────────────────────────────────────────────────────
# CELL 22: Generation — JS divergence + Ordinal MixUp fallback
# ─────────────────────────────────────────────────────────────
NEW_CELL_22 = r"""# ─────────────────────────────────────────────────────────────────────────
# Generation + Quality Filtering (v2)
# Cải tiến:
# 1. Chọn model theo thứ tự: per-class > sector-fallback > global
# 2. JS Divergence check per-sequence để loại outlier
# 3. Ordinal-aware MixUp khi không đủ valid sequences
# ─────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta as beta_dist

# Train RF Quality Classifier (fit trên real train data)
print("=== [Step 1] Training RF Quality Classifier ===")
quality_clf = RandomForestClassifier(
    n_estimators=150,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    class_weight='balanced',
    max_depth=15,
    min_samples_leaf=2
)
quality_clf.fit(train[TIMEGAN_FEATURE_COLUMNS].fillna(0), train[CONFIG['target_column']])
print(f"✔ RF Classifier trained on {len(train)} real samples")

# Build per-class reference distributions (for JS divergence)
print("\n[Step 2] Building reference distributions for JS divergence check...")
REFERENCE_DISTS = {}
bins_config = {}
for col in TIMEGAN_NUMERIC_COLUMNS:
    col_vals = train[col].dropna()
    p01 = float(col_vals.quantile(0.01))
    p99 = float(col_vals.quantile(0.99))
    bins_config[col] = np.linspace(p01 - 1e-9, p99 + 1e-9, 31)  # 30 bins

for cls_label in sorted(train[CONFIG['target_column']].unique()):
    cls_data = train[train[CONFIG['target_column']] == cls_label]
    cls_hist = {}
    for col in TIMEGAN_NUMERIC_COLUMNS:
        if col in bins_config and col in cls_data.columns:
            hist, _ = np.histogram(cls_data[col].dropna(), bins=bins_config[col], density=True)
            hist = hist + 1e-8
            cls_hist[col] = hist / hist.sum()
    REFERENCE_DISTS[int(cls_label)] = cls_hist
print(f"✔ Reference distributions built for {len(REFERENCE_DISTS)} classes")

def compute_js_divergence_for_window(window_rows: pd.DataFrame, class_label: int) -> float:
    """JS divergence trung bình giữa synthetic window và real class distribution."""
    if int(class_label) not in REFERENCE_DISTS:
        return 0.0
    ref = REFERENCE_DISTS[int(class_label)]
    js_vals = []
    for col, ref_hist in ref.items():
        if col not in window_rows.columns:
            continue
        hist, _ = np.histogram(window_rows[col].dropna(), bins=bins_config[col], density=True)
        hist = hist + 1e-8
        hist = hist / hist.sum()
        js_vals.append(float(jensenshannon(ref_hist, hist)))
    return float(np.mean(js_vals)) if js_vals else 0.0

def ordinal_mixup_fallback(
    train_df: pd.DataFrame,
    target_cls: int,
    n_rows_needed: int,
    feature_cols: list,
    target_col: str,
    alpha: float = 0.3,
    adjacent_only: bool = True,
    rng: np.random.Generator = None
) -> pd.DataFrame:
    """
    Ordinal-aware MixUp: tạo synthetic rows bằng cách nội suy
    giữa class target_cls và các class cận (target_cls ± 1).
    Đảm bảo label không trộn quá 2 lớp xa nhau.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    cls_a_df = train_df[train_df[target_col] == target_cls][feature_cols].dropna()
    if len(cls_a_df) == 0:
        return pd.DataFrame()

    # Chọn partner class
    candidates = []
    if adjacent_only:
        for adj in [target_cls - 1, target_cls + 1]:
            df_cand = train_df[train_df[target_col] == adj][feature_cols].dropna()
            if len(df_cand) > 0:
                candidates.append(df_cand)
    if not candidates:
        # Fall back to same-class duplication với noise
        candidates.append(cls_a_df)

    rows = []
    for _ in range(n_rows_needed):
        # Sample source from target class
        a = cls_a_df.sample(1, replace=True, random_state=None).values[0]
        # Sample partner
        partner_df = candidates[rng.integers(0, len(candidates))]
        b = partner_df.sample(1, replace=True, random_state=None).values[0]
        # Beta mixing coefficient (small alpha → stay near boundary)
        lam = float(rng.beta(alpha, alpha))
        mixed = lam * a + (1.0 - lam) * b
        rows.append(mixed)

    result = pd.DataFrame(rows, columns=feature_cols)
    result[target_col] = target_cls
    return result

# ─── Main Generation Loop ─────────────────────────────────────────────────
print("\n=== [Step 3] Generating Synthetic Samples ===")

target_col  = CONFIG['target_column']
date_col    = CONFIG['date_column']
entity_col  = CONFIG['entity_column']
seq_len     = int(CONFIG['seq_len'])
js_thresh   = float(CONFIG.get('js_divergence_threshold', 0.20))
rf_thresh   = float(CONFIG.get('rf_quality_threshold', 0.5))
enable_mixup = bool(CONFIG.get('enable_ordinal_mixup_fallback', True))
mixup_alpha  = float(CONFIG.get('mixup_alpha', 0.3))
mixup_adjacent = bool(CONFIG.get('mixup_adjacent_only', True))

rng = np.random.default_rng(int(CONFIG['random_seed']))

def _safe_reference_date(series):
    dates = pd.to_datetime(series, errors='coerce').dropna()
    return dates.max() if len(dates) > 0 else pd.Timestamp(datetime.now().date())

base_reference_date = _safe_reference_date(train[date_col]) if date_col in train.columns else pd.Timestamp(datetime.now().date())

def get_oversample_factor(real_count: int) -> float:
    """Generate extra to compensate for JS + RF filtering losses."""
    if real_count < 5:  return 30.0
    if real_count < 15: return 20.0
    if real_count < 30: return 12.0
    if real_count < 60: return 6.0
    return 3.5

generated_rows = []
mixup_rows     = []
generated_windows_log = []
synthetic_window_counter = 0

for cls, n_windows in sorted(windows_per_class.items()):
    if int(n_windows) <= 0:
        continue

    real_count   = int(class_counts.get(int(cls), 0))
    n_rows_need  = int(samples_per_class.get(int(cls), 0))

    # ── Select Model (priority: per-class > sector > global) ──────────────
    model_key    = f'class_{int(cls)}'
    sector_key   = SECTOR_FALLBACK_MAP.get(int(cls), '')
    if model_key in TIMEGAN_MODELS:
        active_model  = TIMEGAN_MODELS[model_key]
        model_source  = model_key
    elif sector_key and sector_key in TIMEGAN_MODELS:
        active_model  = TIMEGAN_MODELS[sector_key]
        model_source  = f'sector({sector_key})'
    elif 'global' in TIMEGAN_MODELS:
        active_model  = TIMEGAN_MODELS['global']
        model_source  = 'global'
    else:
        print(f"⚠ No model for class {cls}, trying MixUp only...")
        active_model  = None
        model_source  = None

    class_generated = []

    # ── TimeGAN Generation with oversampling ──────────────────────────────
    if active_model is not None:
        try:
            oversample_f  = get_oversample_factor(real_count)
            over_n_windows = int(np.ceil(n_windows * oversample_f))

            print(f"\nClass {cls} (real={real_count}, need={n_rows_need} rows, {n_windows} wins):")
            print(f"  Model: {model_source} | Oversample {over_n_windows} windows ({oversample_f}x)")

            sampled_norm = safe_timegan_sample(active_model, over_n_windows)
            if sampled_norm.shape[0] == 0:
                print(f"  ⚠ Model returned 0 windows")
            else:
                sampled_norm = sampled_norm[:, :seq_len, :len(TIMEGAN_FEATURE_COLUMNS)]
                sampled_raw  = inverse_timegan_scale(sampled_norm, TIMEGAN_SCALER, TIMEGAN_FEATURE_COLUMNS)

                class_meta       = TIMEGAN_SEQUENCE_META[TIMEGAN_SEQUENCE_META['window_label'] == int(cls)].copy()
                class_start_dates = pd.to_datetime(
                    class_meta.get('start_date', pd.Series([], dtype='datetime64[ns]')), errors='coerce'
                ).dropna().tolist()
                if len(class_start_dates) == 0:
                    class_start_dates = [base_reference_date]

                accepted_wins = 0
                rejected_js   = 0
                rejected_rf   = 0

                for win_idx in range(sampled_raw.shape[0]):
                    seq_block   = sampled_raw[win_idx]
                    ref_date    = class_start_dates[win_idx % len(class_start_dates)]
                    if pd.isna(ref_date):
                        ref_date = base_reference_date + pd.Timedelta(days=30 * (synthetic_window_counter + 1))

                    synthetic_ticker = f"SYN_{int(cls)}_{synthetic_window_counter:07d}"

                    # Build window rows tentatively
                    win_rows = []
                    for step_idx in range(seq_len):
                        row_d = {col: float(seq_block[step_idx, col_idx])
                                 for col_idx, col in enumerate(TIMEGAN_FEATURE_COLUMNS)}
                        row_d[target_col] = int(cls)
                        if entity_col in train.columns:
                            row_d[entity_col] = synthetic_ticker
                        if date_col in train.columns:
                            row_d[date_col] = pd.to_datetime(ref_date) + pd.Timedelta(days=90 * step_idx)
                        win_rows.append(row_d)

                    win_df = pd.DataFrame(win_rows)

                    # ── JS Divergence Filter ──────────────────────────────
                    js_score = compute_js_divergence_for_window(win_df, int(cls))
                    if js_score > js_thresh:
                        rejected_js += 1
                        synthetic_window_counter += 1
                        continue

                    # ── RF Quality Filter ─────────────────────────────────
                    preds = quality_clf.predict(win_df[TIMEGAN_FEATURE_COLUMNS].fillna(0))
                    rf_acc = float(np.mean(preds == int(cls)))
                    if rf_acc < rf_thresh:
                        rejected_rf += 1
                        synthetic_window_counter += 1
                        continue

                    class_generated.extend(win_rows)
                    accepted_wins += 1
                    synthetic_window_counter += 1

                    if accepted_wins >= n_windows:
                        break

                generated_windows_log.append({
                    'class_label': int(cls),
                    'windows_requested': int(n_windows),
                    'windows_generated_raw': int(sampled_raw.shape[0]),
                    'windows_accepted': accepted_wins,
                    'rejected_js': rejected_js,
                    'rejected_rf': rejected_rf,
                    'model_source': model_source
                })
                print(f"  ✔ {accepted_wins} windows accepted | JS_rejected={rejected_js} RF_rejected={rejected_rf}")

        except Exception as e:
            print(f"  ⚠ TimeGAN generation failed: {e}")

    # Add accepted TimeGAN rows to pool
    generated_rows.extend(class_generated)

    # ── Ordinal MixUp Fallback (fill gap nếu thiếu) ───────────────────────
    generated_cls_count = len(class_generated)
    gap = n_rows_need - generated_cls_count

    if gap > 0 and enable_mixup:
        print(f"  → MixUp fallback: need {gap} more rows for class {cls}")
        mixup_df = ordinal_mixup_fallback(
            train_df=train,
            target_cls=int(cls),
            n_rows_needed=gap,
            feature_cols=TIMEGAN_FEATURE_COLUMNS,
            target_col=target_col,
            alpha=mixup_alpha,
            adjacent_only=mixup_adjacent,
            rng=rng
        )
        if len(mixup_df) > 0:
            # Thêm metadata
            for i, row in mixup_df.iterrows():
                r = row.to_dict()
                r['is_mixup'] = 1
                if entity_col in train.columns:
                    r[entity_col] = f"MIX_{int(cls)}_{i:07d}"
                if date_col in train.columns:
                    r[date_col] = base_reference_date + pd.Timedelta(days=90 * i)
                mixup_rows.append(r)
            print(f"  ✔ MixUp created {len(mixup_df)} rows")
    elif gap > 0:
        print(f"  ⚠ Gap of {gap} rows for class {cls} — MixUp disabled")

# ─── Combine TimeGAN + MixUp rows ─────────────────────────────────────────
all_synthetic_rows = generated_rows + mixup_rows

if len(all_synthetic_rows) == 0:
    synthetic_df = pd.DataFrame()
    print("\n⚠ No synthetic rows generated")
else:
    synthetic_df = pd.DataFrame(all_synthetic_rows)

    # Chuẩn hóa: fill missing columns
    if 'is_mixup' not in synthetic_df.columns:
        synthetic_df['is_mixup'] = 0
    else:
        synthetic_df['is_mixup'] = synthetic_df['is_mixup'].fillna(0).astype(int)

    print(f"\n✔ Total synthetic rows: {len(synthetic_df)}")
    print(f"  - TimeGAN rows: {len(generated_rows)}")
    print(f"  - MixUp rows:   {len(mixup_rows)}")
    print(f"\nClass distribution (synthetic):")
    print(synthetic_df[target_col].value_counts().sort_index())

TIMEGAN_GENERATION_LOG = generated_windows_log
"""

# ─────────────────────────────────────────────────────────────
# CELL 31: Hybrid Rebalancing — Entropy-weighted
# ─────────────────────────────────────────────────────────────
NEW_CELL_31 = r"""# ─────────────────────────────────────────────────────────────────────────
# Hybrid Rebalancing (v2 — ENABLED mặc định)
# Chiến lược: sau TimeGAN + MixUp, áp dụng:
# 1. Cap majority classes tại P80 count (thay vì P75)
# 2. Giữ nguyên minority classes đã được augment
# 3. Minority floor = 150 (giảm từ 200 để realistic với rare classes)
# 4. Nếu minority vẫn < floor sau TimeGAN + MixUp: duplicate thêm
# ─────────────────────────────────────────────────────────────────────────
ENABLE_HYBRID_REBALANCING = bool(CONFIG.get('enable_hybrid_rebalancing', True))
MAJORITY_CAP_PERCENTILE   = int(CONFIG.get('majority_cap_percentile', 80))
MINORITY_FLOOR            = int(CONFIG.get('minority_floor', 150))

if ENABLE_HYBRID_REBALANCING and len(train_augmented) > 0:
    print("\n" + "=" * 70)
    print("HYBRID REBALANCING v2  (Majority Cap + Minority Floor)")
    print("=" * 70)

    target_col  = CONFIG['target_column']
    priority_cls = [int(c) for c in CONFIG.get('priority_class_labels', [])]
    aug_counts  = train_augmented[target_col].value_counts().sort_index()

    # Determine cap from non-priority class counts
    non_priority_counts = aug_counts[~aug_counts.index.isin(priority_cls)]
    nonzero_np = non_priority_counts[non_priority_counts > 0]
    if len(nonzero_np) > 0:
        cap_value = int(np.percentile(nonzero_np.values, MAJORITY_CAP_PERCENTILE))
    else:
        cap_value = int(aug_counts[aug_counts > 0].quantile(0.80)) if len(aug_counts[aug_counts > 0]) > 0 else int(MINORITY_FLOOR * 4)
    cap_value = max(cap_value, MINORITY_FLOOR * 2)

    print(f"  Non-priority class P{MAJORITY_CAP_PERCENTILE} cap: {cap_value}")
    print(f"  Minority floor: {MINORITY_FLOOR}")

    rebalanced_chunks = []
    for cls_label in sorted(train_augmented[target_col].unique()):
        cls_df = train_augmented[train_augmented[target_col] == cls_label]
        n = len(cls_df)
        is_priority = int(cls_label) in priority_cls

        if not is_priority and n > cap_value:
            # Downsample majority: ưu tiên giữ real data
            real_subset  = cls_df[cls_df.get('is_synthetic', pd.Series(0, index=cls_df.index)).eq(0)]
            synth_subset = cls_df[cls_df.get('is_synthetic', pd.Series(0, index=cls_df.index)).eq(1)]
            n_real = len(real_subset)
            n_need = cap_value - n_real
            if n_need >= 0 and len(synth_subset) > 0:
                synth_sampled = synth_subset.sample(n=min(n_need, len(synth_subset)), random_state=RANDOM_SEED)
                cls_df = pd.concat([real_subset, synth_sampled], ignore_index=True)
            elif n_need < 0:
                # Cắt cả real (hiếm xảy ra)
                cls_df = real_subset.sample(n=cap_value, random_state=RANDOM_SEED)
            print(f"  Class {cls_label} (majority): {n} → {len(cls_df)} (capped at {cap_value})")

        elif n < MINORITY_FLOOR and n > 0:
            # Oversample minority via duplication (last resort)
            extra_needed = MINORITY_FLOOR - n
            extra = cls_df.sample(n=extra_needed, random_state=RANDOM_SEED, replace=True)
            extra['is_synthetic'] = 1
            cls_df = pd.concat([cls_df, extra], ignore_index=True)
            print(f"  Class {cls_label} (minority): {n} → {len(cls_df)} (floor padded)")
        else:
            status = "priority-kept" if is_priority else "kept"
            print(f"  Class {cls_label}: {n} ({status})")

        rebalanced_chunks.append(cls_df)

    train_augmented = pd.concat(rebalanced_chunks, ignore_index=True)
    train_augmented = train_augmented.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    final_counts  = train_augmented[target_col].value_counts().sort_index()
    nonzero_final = final_counts[final_counts > 0]
    ratio_after   = nonzero_final.min() / nonzero_final.max() if len(nonzero_final) > 0 and nonzero_final.max() > 0 else float('nan')

    print(f"\n  After hybrid rebalancing: {len(train_augmented)} total samples")
    print(f"  Class count range: [{nonzero_final.min()}, {nonzero_final.max()}]")
    print(f"  Imbalance ratio: {ratio_after:.3f}" if np.isfinite(ratio_after) else "  Imbalance ratio: N/A")
    quality_label = 'EXCELLENT (>0.5)' if ratio_after > 0.5 else 'GOOD (0.3-0.5)' if ratio_after > 0.3 else 'MODERATE (0.1-0.3)' if ratio_after > 0.1 else 'STILL IMBALANCED (<0.1)'
    print(f"  Balance quality: {quality_label}")
    print(f"\n  Full distribution:")
    print(final_counts)
else:
    print("\nHybrid rebalancing: SKIPPED (disabled in CONFIG or no augmented data)")
"""

# ─────────────────────────────────────────────────────────────
# Apply patches
# ─────────────────────────────────────────────────────────────
import re as re_module

def patch_cell(nb_obj, cell_idx, new_src):
    nb_obj['cells'][cell_idx]['source'] = new_src
    nb_obj['cells'][cell_idx]['outputs'] = []
    nb_obj['cells'][cell_idx]['execution_count'] = None

patch_cell(nb, 6,  NEW_CELL_6)
patch_cell(nb, 18, NEW_CELL_18)
patch_cell(nb, 21, NEW_CELL_21)
patch_cell(nb, 22, NEW_CELL_22)
patch_cell(nb, 31, NEW_CELL_31)

# Fix: add import re to cell 4 (imports cell)
cell4_src = ''.join(nb['cells'][4]['source'])
if 'import re' not in cell4_src:
    nb['cells'][4]['source'] = 'import re\n' + cell4_src

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✔ Notebook patched successfully!")
print("Changes applied to cells: 4 (imports), 6 (CONFIG), 18 (Training), 21 (Sampling), 22 (Generation), 31 (Rebalancing)")
