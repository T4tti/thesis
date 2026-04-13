from pathlib import Path

path = Path('e:/thesis/notebooks/transformer-lstm.ipynb')
text = path.read_text(encoding='utf-8')

replacements = [
    ('"SAMPLER_WEIGHT_POWER = 0.5\\n",', '"SAMPLER_WEIGHT_POWER = 0.75\\n",'),
    ('"EARLY_STOP_METRIC = \'val_acc\'\\n",', '"EARLY_STOP_METRIC = \'val_f1\'\\n",'),
    ('"MIXUP_PROB = 0.10\\n",', '"MIXUP_PROB = 0.05\\n",'),
    ('"MIN_UPLIFT_TO_SAVE = -0.020\\n",', '"MIN_UPLIFT_TO_SAVE = -0.020\\n",\n                "ENABLE_UPLIFT_GATE = False\\n",'),
    (
        '"    current_metric = vl_acc if EARLY_STOP_METRIC == \'val_acc\' else vl_f1w\\n",',
        '"    current_metric = vl_acc if EARLY_STOP_METRIC == \'val_acc\' else (vl_f1 if EARLY_STOP_METRIC == \'val_f1\' else vl_f1w)\\n",',
    ),
    (
        '"    meets_uplift_gate = bool(np.isnan(val_persistence_acc)) or (val_uplift_vs_persistence >= MIN_UPLIFT_TO_SAVE)\\n",',
        '"    meets_uplift_gate = (not ENABLE_UPLIFT_GATE) or bool(np.isnan(val_persistence_acc)) or (val_uplift_vs_persistence >= MIN_UPLIFT_TO_SAVE)\\n",',
    ),
    (
        '"    if epoch + 1 >= MIN_EPOCHS_BEFORE_UPLIFT_STOP and no_uplift_counter >= NO_UPLIFT_PATIENCE:\\n",',
        '"    if ENABLE_UPLIFT_GATE and epoch + 1 >= MIN_EPOCHS_BEFORE_UPLIFT_STOP and no_uplift_counter >= NO_UPLIFT_PATIENCE:\\n",',
    ),
    (
        '"    print(\'Warning: no checkpoint met uplift gate; loaded best raw metric model instead.\')\\n",',
        '"    print(\'Warning: no checkpoint met save gate; loaded best raw metric model instead.\')\\n",',
    ),
    (
        '"print(f\'Min uplift required to save best checkpoint: {MIN_UPLIFT_TO_SAVE:+.3f}\')\\n",',
        '"print(f\'Uplift gate enabled: {ENABLE_UPLIFT_GATE}\')\\n",\n                "print(f\'Min uplift required to save best checkpoint: {MIN_UPLIFT_TO_SAVE:+.3f}\')\\n",',
    ),
    (
        '"**Training:** Mixed precision (fp16 on CUDA) + AdamW + OneCycleLR (`max_lr` tăng lên khoảng `1e-3`) + số epoch tăng gấp đôi + early stopping theo `val_f1_weighted`\\n",',
        '"**Training:** Mixed precision (fp16 on CUDA) + AdamW + OneCycleLR (`max_lr` tăng lên khoảng `1e-3`) + số epoch tăng gấp đôi + early stopping theo `val_f1` (macro)\\n",',
    ),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f'Missing pattern: {old[:120]}')
    text = text.replace(old, new)

path.write_text(text, encoding='utf-8')
print('Patched notebook successfully')
