# Generate comprehensive report
generation_log = TIMEGAN_GENERATION_LOG if 'TIMEGAN_GENERATION_LOG' in globals() else []
trained_class_models = [
    cls for cls, info in TIMEGAN_TRAINING_REGISTRY.get('per_class', {}).items()
    if info.get('status') == 'trained'
]

report = f"""# TimeGAN Augmentation Report (Standalone Pipeline)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- **Random seed:** {CONFIG['random_seed']}
- **Target column:** {CONFIG['target_column']}
- **Data split:** Train {CONFIG['train_ratio']*100:.0f}% / Val {CONFIG['val_ratio']*100:.0f}% / Test {CONFIG['test_ratio']*100:.0f}%
- **Sequence length (`seq_len`):** {CONFIG['seq_len']}
- **Sequence stride:** {CONFIG['sequence_stride']}
- **TimeGAN train steps:** {CONFIG['timegan_train_steps']}
- **Batch size:** {CONFIG['timegan_batch_size']}
- **Learning rate:** {CONFIG['timegan_learning_rate']}
- **Use GPU (requested):** {CONFIG.get('timegan_use_gpu', False)}

## Pipeline Mode
âœ“ Standalone mode: loaded source dataset and split inside notebook
âœ“ Sequence-aware mode: TimeGAN windows grouped by entity and ordered by time

## Data Summary
- **Original train size:** {len(train)}
- **Val size:** {len(val)}
- **Test size:** {len(test)}
- **Synthetic rows generated (raw):** {len(synthetic_df) if len(synthetic_df) > 0 else 0}
- **Synthetic rows after filtering:** {len(synthetic_df_filtered) if len(synthetic_df_filtered) > 0 else 0}
- **Final augmented train size:** {len(train_augmented)}
- **Synthetic ratio:** {100*len(synthetic_df_filtered)/len(train_augmented) if len(synthetic_df_filtered) > 0 else 0:.1f}%

## TimeGAN Training Summary
- **Global model status:** {TIMEGAN_TRAINING_REGISTRY.get('global', {}).get('status', 'unknown')}
- **Global windows used:** {TIMEGAN_TRAINING_REGISTRY.get('global', {}).get('windows_used', 'N/A')}
- **Per-class trained models:** {trained_class_models if len(trained_class_models) > 0 else 'None'}

## Generation Log
"""

if len(generation_log) == 0:
    report += "\n- No generation log entries available."
else:
    for item in generation_log:
        report += (
            f"\n- Class {item.get('class_label')}: "
            f"requested {item.get('windows_requested')} windows, "
            f"generated {item.get('windows_generated')} windows, "
            f"model source = {item.get('model_source')}"
        )

report += """

## Class Distribution

### Before Augmentation
"""

before_dist = train[CONFIG['target_column']].value_counts().sort_index()
for class_label, count in before_dist.items():
    report += f"\n- Class {class_label}: {count}"

report += "\n\n### After Augmentation\n"

after_dist = train_augmented[CONFIG['target_column']].value_counts().sort_index()
for class_label, count in after_dist.items():
    original = before_dist.get(class_label, 0)
    increase = count - original
    pct = 100 * increase / original if original > 0 else 0
    report += f"\n- Class {class_label}: {count} (+{increase}, {pct:.1f}%)"

report += f"""

## Preprocessing Applied
âœ“ Full preprocessing pipeline executed (imputation, log transform, scaling, encoding) with fit-on-train only.
âœ“ TimeGAN-specific min-max scaling applied on model feature columns before sequence training.
âœ“ Synthetic sequences inverse-scaled and mapped back to tabular rows.

## Synthetic Data Quality
- Quantile-based clipping applied at ({CONFIG.get('quality_quantile_low', 0.005)}, {CONFIG.get('quality_quantile_high', 0.995)})
- Duplicate synthetic rows removed
- See `quality_analysis.json` for statistical comparison between real and synthetic sets

## Anti-Leakage Protocol
âœ“ All transformations fitted only on train set
âœ“ TimeGAN trained only on train-derived windows
âœ“ Synthetic samples added only to train set
âœ“ Val/test sets remain real data for evaluation

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
âœ“ Fixed random seed ({CONFIG['random_seed']})
âœ“ Config and generation metadata saved
âœ“ Standalone split and processing protocol kept deterministic where possible
"""

# Save report
report_path = REPORTS_DIR / 'timegan_augmentation_report.md'
with open(report_path, 'w') as f:
    f.write(report)

print("âœ“ Report generated")
print(f"âœ“ Report saved to: {report_path}")
print(report)