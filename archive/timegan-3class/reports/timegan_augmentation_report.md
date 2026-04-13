# TimeGAN Augmentation Report (Standalone Pipeline)

**Generated:** 2026-04-07 03:34:17

## Configuration
- **Random seed:** 42
- **Target column:** rating_detail
- **Split strategy:** Walk-forward expanding window | active_fold=3 / 3 | train_years=[2005, 2009, 2010, 2011, 2012, 2013, 2014] | val_years=[2015] | test_years=[2016]
- **Sequence length (`seq_len`):** 8
- **Sequence stride:** 1
- **TimeGAN train steps:** 140
- **Batch size:** 24
- **Learning rate:** 0.0005
- **Use GPU (requested):** False

## Pipeline Mode
 Standalone mode: loaded source dataset and split inside notebook
 Sequence-aware mode: TimeGAN windows grouped by entity and ordered by time

## Data Summary
- **Original train size:** 5250
- **Val size:** 2250
- **Test size:** 1180
- **Synthetic rows generated (raw):** 2887
- **Synthetic rows after filtering:** 2887
- **Final augmented train size:** 8137
- **Synthetic rows in final augmented set:** 2887
- **Synthetic ratio:** 35.5%

## TimeGAN Training Summary
- **Global model status:** trained
- **Global windows used:** 473
- **Per-class trained models:** ['0', '1']

## Generation Log

- Class 0: requested 214 windows, generated 0 windows, model source = class_0, JS rejected = 384, RF rejected = 0
- Class 1: requested 148 windows, generated 0 windows, model source = class_1, JS rejected = 256, RF rejected = 0

## Class Distribution

### Before Augmentation

- Class 0: 182
- Class 1: 1653
- Class 2: 3415

### After Augmentation

- Class 0: 1888 (+1706, 937.4%)
- Class 1: 2834 (+1181, 71.4%)
- Class 2: 3415 (+0, 0.0%)

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
