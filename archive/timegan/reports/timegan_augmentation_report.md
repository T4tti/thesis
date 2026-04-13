# TimeGAN Augmentation Report (Standalone Pipeline)

**Generated:** 2026-04-05 08:22:15

## Configuration
- **Random seed:** 42
- **Target column:** rating_detail
- **Split strategy:** Walk-forward expanding window | active_fold=3 / 3 | train_years=[2005, 2009, 2010, 2011, 2012, 2013, 2014] | val_years=[2015] | test_years=[2016]
- **Sequence length (`seq_len`):** 8
- **Sequence stride:** 1
- **TimeGAN train steps:** 150
- **Batch size:** 64
- **Learning rate:** 0.0005
- **Use GPU (requested):** False

## Pipeline Mode
 Standalone mode: loaded source dataset and split inside notebook
 Sequence-aware mode: TimeGAN windows grouped by entity and ordered by time

## Data Summary
- **Original train size:** 5250
- **Val size:** 2250
- **Test size:** 1180
- **Synthetic rows generated (raw):** 1837
- **Synthetic rows after filtering:** 1837
- **Final augmented train size:** 7087
- **Synthetic rows in final augmented set:** 1837
- **Synthetic ratio:** 25.9%

## TimeGAN Training Summary
- **Global model status:** trained
- **Global windows used:** 195
- **Per-class trained models:** ['4', '5', '6', '18', '19', '20']

## Generation Log

- Class 0: requested 24 windows, generated 0 windows, model source = global, JS rejected = 192, RF rejected = 0
- Class 1: requested 24 windows, generated 0 windows, model source = global, JS rejected = 144, RF rejected = 0
- Class 2: requested 23 windows, generated 0 windows, model source = global, JS rejected = 138, RF rejected = 0
- Class 3: requested 24 windows, generated 0 windows, model source = global, JS rejected = 144, RF rejected = 0
- Class 4: requested 19 windows, generated 0 windows, model source = class_4, JS rejected = 38, RF rejected = 0
- Class 5: requested 17 windows, generated 0 windows, model source = class_5, JS rejected = 34, RF rejected = 0
- Class 6: requested 16 windows, generated 0 windows, model source = class_6, JS rejected = 32, RF rejected = 0
- Class 7: requested 4 windows, generated 0 windows, model source = global, JS rejected = 8, RF rejected = 0
- Class 8: requested 4 windows, generated 0 windows, model source = global, JS rejected = 8, RF rejected = 0
- Class 9: requested 4 windows, generated 0 windows, model source = global, JS rejected = 8, RF rejected = 0
- Class 11: requested 3 windows, generated 0 windows, model source = global, JS rejected = 6, RF rejected = 0
- Class 17: requested 3 windows, generated 0 windows, model source = global, JS rejected = 6, RF rejected = 0
- Class 18: requested 16 windows, generated 0 windows, model source = class_18, JS rejected = 32, RF rejected = 0
- Class 19: requested 15 windows, generated 0 windows, model source = class_19, JS rejected = 30, RF rejected = 0
- Class 20: requested 21 windows, generated 0 windows, model source = class_20, JS rejected = 63, RF rejected = 0
- Class 21: requested 20 windows, generated 0 windows, model source = global, JS rejected = 60, RF rejected = 0

## Class Distribution

### Before Augmentation

- Class 0: 3
- Class 1: 6
- Class 2: 13
- Class 3: 6
- Class 4: 67
- Class 5: 87
- Class 6: 177
- Class 7: 271
- Class 8: 256
- Class 9: 270
- Class 10: 383
- Class 11: 296
- Class 12: 409
- Class 13: 763
- Class 14: 543
- Class 15: 458
- Class 16: 588
- Class 17: 297
- Class 18: 154
- Class 19: 110
- Class 20: 44
- Class 21: 49

### After Augmentation

- Class 0: 191 (+188, 6266.7%)
- Class 1: 192 (+186, 3100.0%)
- Class 2: 194 (+181, 1392.3%)
- Class 3: 192 (+186, 3100.0%)
- Class 4: 214 (+147, 219.4%)
- Class 5: 222 (+135, 155.2%)
- Class 6: 304 (+127, 71.8%)
- Class 7: 297 (+26, 9.6%)
- Class 8: 286 (+30, 11.7%)
- Class 9: 296 (+26, 9.6%)
- Class 10: 383 (+0, 0.0%)
- Class 11: 316 (+20, 6.8%)
- Class 12: 409 (+0, 0.0%)
- Class 13: 763 (+0, 0.0%)
- Class 14: 543 (+0, 0.0%)
- Class 15: 458 (+0, 0.0%)
- Class 16: 588 (+0, 0.0%)
- Class 17: 316 (+19, 6.4%)
- Class 18: 279 (+125, 81.2%)
- Class 19: 230 (+120, 109.1%)
- Class 20: 206 (+162, 368.2%)
- Class 21: 208 (+159, 324.5%)

## Preprocessing Applied
 Full preprocessing pipeline executed (imputation, log transform, scaling, encoding) with fit-on-train only.
 TimeGAN-specific min-max scaling applied on model feature columns before sequence training.
 Synthetic sequences inverse-scaled and mapped back to tabular rows.

## Synthetic Data Quality
- Quantile-based clipping applied at (0.005, 0.995)
- Duplicate synthetic rows are not removed (disabled to preserve sequence integrity)
- See `quality_analysis.json` for statistical comparison between real and synthetic sets

## Anti-Leakage Protocol
 All transformations fitted only on train set
 TimeGAN trained only on train-derived windows
 Synthetic samples added only to train set
 Val/test sets remain real data for evaluation

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
- `reports/walk_forward_folds_summary.csv` - Walk-forward fold summary
- `reports/split_metadata.json` - Split metadata

## Next Steps
1. Train downstream classification model on augmented training set
2. Evaluate on unchanged val/test sets
3. Compare baseline vs TimeGAN augmentation metrics (balanced accuracy, macro-F1, ordinal metrics)
4. Analyze minority-class gains and calibration effects

## Reproducibility
 Fixed random seed (42)
 Config and generation metadata saved
 Standalone split and processing protocol kept deterministic where possible
