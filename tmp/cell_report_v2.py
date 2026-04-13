# ==============================================================================
# CELL: Generate Comprehensive TimeGAN Augmentation Report
# Fixed version — kiểm tra defensive cho mọi biến có thể chưa tồn tại
# ==============================================================================

import pathlib
from datetime import datetime

# ── 1. Safe-fallback cho mọi biến có thể chưa tồn tại ──────────────────────
generation_log = globals().get('TIMEGAN_GENERATION_LOG', [])

_registry: dict = globals().get('TIMEGAN_TRAINING_REGISTRY', {})
trained_class_models = [
    cls for cls, info in _registry.get('per_class', {}).items()
    if info.get('status') == 'trained'
]

# DataFrames — kiểm tra cả tồn tại lẫn không rỗng
_train             = globals().get('train')
_val               = globals().get('val')
_test              = globals().get('test')
_synthetic_df      = globals().get('synthetic_df')
_synthetic_filtered= globals().get('synthetic_df_filtered')
_train_aug         = globals().get('train_augmented')

def _safe_len(obj) -> int:
    """Trả về len(obj) nếu obj không phải None, ngược lại trả 0."""
    return len(obj) if obj is not None else 0

n_train          = _safe_len(_train)
n_val            = _safe_len(_val)
n_test           = _safe_len(_test)
n_synthetic_raw  = _safe_len(_synthetic_df)
n_synthetic_filt = _safe_len(_synthetic_filtered)
n_train_aug      = _safe_len(_train_aug)

_synth_ratio = (
    100 * n_synthetic_filt / n_train_aug
    if (n_synthetic_filt > 0 and n_train_aug > 0)
    else 0.0
)

# ── 2. Đảm bảo thư mục output tồn tại ─────────────────────────────────────
REPORTS_DIR = globals().get('REPORTS_DIR', pathlib.Path('reports'))
REPORTS_DIR = pathlib.Path(REPORTS_DIR)   # đảm bảo là Path object
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 3. Lấy CONFIG an toàn ──────────────────────────────────────────────────
CONFIG: dict = globals().get('CONFIG', {})

# ── 4. Xây dựng report ─────────────────────────────────────────────────────
report = f"""# TimeGAN Augmentation Report (Standalone Pipeline)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- **Random seed:** {CONFIG.get('random_seed', 'N/A')}
- **Target column:** {CONFIG.get('target_column', 'N/A')}
- **Data split:** Train {CONFIG.get('train_ratio', 0)*100:.0f}% / Val {CONFIG.get('val_ratio', 0)*100:.0f}% / Test {CONFIG.get('test_ratio', 0)*100:.0f}%
- **Sequence length (`seq_len`):** {CONFIG.get('seq_len', 'N/A')}
- **Sequence stride:** {CONFIG.get('sequence_stride', 'N/A')}
- **TimeGAN train steps:** {CONFIG.get('timegan_train_steps', 'N/A')}
- **Batch size:** {CONFIG.get('timegan_batch_size', 'N/A')}
- **Learning rate:** {CONFIG.get('timegan_learning_rate', 'N/A')}
- **Use GPU (requested):** {CONFIG.get('timegan_use_gpu', False)}

## Pipeline Mode
✔ Standalone mode: loaded source dataset and split inside notebook
✔ Sequence-aware mode: TimeGAN windows grouped by entity and ordered by time

## Data Summary
- **Original train size:** {n_train}
- **Val size:** {n_val}
- **Test size:** {n_test}
- **Synthetic rows generated (raw):** {n_synthetic_raw}
- **Synthetic rows after filtering:** {n_synthetic_filt}
- **Final augmented train size:** {n_train_aug}
- **Synthetic ratio:** {_synth_ratio:.1f}%

## TimeGAN Training Summary
- **Global model status:** {_registry.get('global', {}).get('status', 'unknown')}
- **Global windows used:** {_registry.get('global', {}).get('windows_used', 'N/A')}
- **Per-class trained models:** {trained_class_models if len(trained_class_models) > 0 else 'None'}

## Generation Log
"""

if len(generation_log) == 0:
    report += "\n- No generation log entries available."
else:
    for item in generation_log:
        report += (
            f"\n- Class {item.get('class_label', '?')}: "
            f"requested {item.get('windows_requested', '?')} windows, "
            f"generated {item.get('windows_generated', '?')} windows, "
            f"model source = {item.get('model_source', '?')}"
        )

# ── Class Distribution ──────────────────────────────────────────────────────
report += "\n\n## Class Distribution\n\n### Before Augmentation\n"

_target_col = CONFIG.get('target_column', 'rating_class')

if _train is not None and _target_col in _train.columns:
    before_dist = _train[_target_col].value_counts().sort_index()
    for class_label, count in before_dist.items():
        report += f"\n- Class {class_label}: {count}"
else:
    before_dist = None
    report += "\n- (train DataFrame not available)"

report += "\n\n### After Augmentation\n"

if _train_aug is not None and _target_col in _train_aug.columns:
    after_dist = _train_aug[_target_col].value_counts().sort_index()
    for class_label, count in after_dist.items():
        # FIX: pd.Series.get() với int index đôi khi không hoạt động đúng
        # → dùng at[] / loc[] hoặc kiểm tra index trước
        if before_dist is not None and class_label in before_dist.index:
            original = int(before_dist.loc[class_label])
        else:
            original = 0
        increase = int(count) - original
        pct = 100 * increase / original if original > 0 else 0.0
        report += f"\n- Class {class_label}: {int(count)} (+{increase}, {pct:.1f}%)"
else:
    report += "\n- (train_augmented DataFrame not available)"

# ── Phần còn lại của report ────────────────────────────────────────────────
_q_low  = CONFIG.get('quality_quantile_low',  0.005)
_q_high = CONFIG.get('quality_quantile_high', 0.995)
_seed   = CONFIG.get('random_seed', 'N/A')

report += f"""

## Preprocessing Applied
✔ Full preprocessing pipeline executed (imputation, log transform, scaling, encoding) with fit-on-train only.
✔ TimeGAN-specific min-max scaling applied on model feature columns before sequence training.
✔ Synthetic sequences inverse-scaled and mapped back to tabular rows.

## Synthetic Data Quality
- Quantile-based clipping applied at ({_q_low}, {_q_high})
- Duplicate synthetic rows removed
- See `quality_analysis.json` for statistical comparison between real and synthetic sets

## Anti-Leakage Protocol
✔ All transformations fitted only on train set
✔ TimeGAN trained only on train-derived windows
✔ Synthetic samples added only to train set
✔ Val/test sets remain real data for evaluation

## Output Files
- `splits/train_raw.csv` - Raw training split
- `splits/val_raw.csv` - Raw validation split
- `splits/test_raw.csv` - Raw test split
- `splits/train.csv` - Training set (preprocessed)
- `splits/val.csv` - Validation set (preprocessed)
- `splits/test.csv` - Test set (preprocessed)
- `splits/train_augmented_timegan.csv` - Augmented training set (primary)
- `splits/train_augmented_ctgan.csv` - Compatibility alias
- `models/timegan/timegan_training_registry.json` - TimeGAN model/training metadata
- `models/timegan/quality_analysis.json` - Quality metrics
- `config_timegan.json` - Configuration file
- `reports/class_distribution_timegan_comparison.png` - Visualization
- `reports/timegan_generation_summary.csv` - Per-class generation summary

## Next Steps
1. Train downstream classification model on augmented training set
2. Evaluate on unchanged val/test sets
3. Compare baseline vs TimeGAN augmentation metrics (balanced accuracy, macro-F1, ordinal metrics)
4. Analyze minority-class gains and calibration effects

## Reproducibility
✔ Fixed random seed ({_seed})
✔ Config and generation metadata saved
✔ Standalone split and processing protocol kept deterministic where possible
"""

# ── 5. Lưu report với encoding tường minh ──────────────────────────────────
report_path = REPORTS_DIR / 'timegan_augmentation_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print("✔ Report generated")
print(f"✔ Report saved to: {report_path}")
print(report)
