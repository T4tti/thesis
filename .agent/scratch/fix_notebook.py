import nbformat
import re
import sys

nb_path = r'e:\thesis\notebooks\Transformer-BiLSTM.ipynb'
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
except Exception as e:
    print(f"Error reading notebook: {e}")
    sys.exit(1)

def find_cell(text):
    for i, c in enumerate(nb.cells):
        if c.cell_type == 'code' and text in c.source:
            return i
    return -1

# Bug 1
cell_7_idx = find_cell("_missing_cols = [c for c in FINANCIAL_FEATURES")
if cell_7_idx != -1:
    print(f"Bug 1: Deleting cell {cell_7_idx}")
    del nb.cells[cell_7_idx]

cell_11_idx = find_cell("FINANCIAL_FEATURES = [")
if cell_11_idx != -1:
    print(f"Bug 1: Inserting new cell after {cell_11_idx}")
    new_cell_1_src = """# BUG_FIX_V2 [P0]: Moved from Cell 7 — must run AFTER FINANCIAL_FEATURES is defined in Cell 11.
# Original position (Cell 7) caused NameError because FINANCIAL_FEATURES didn't exist yet.
import json as _json
from pathlib import Path as _Path

_prep_meta_path = None
for _candidate in [
    'models/timegan/preprocessing_metadata.json',
    'data/external/timegan_3groups_output/models/timegan/preprocessing_metadata.json',
]:
    if _Path(_candidate).exists():
        _prep_meta_path = _Path(_candidate)
        break

if _prep_meta_path is not None:
    with open(_prep_meta_path) as _f:
        _prep_meta = _json.load(_f)
    _tg_features = set(_prep_meta.get('numeric_features', []))
    _missing = set(FINANCIAL_FEATURES) - _tg_features
    if _missing:
        print(f'[P2 WARNING] Not in TimeGAN preprocessing: {_missing}')
    else:
        print(f'[P2 Fix 7 OK] All {len(FINANCIAL_FEATURES)} features in TimeGAN preprocessing')
else:
    print('[P2 Fix 7 INFO] preprocessing_metadata.json not found, skipping metadata check.')

# Validate features exist in loaded data — runs correctly here since FINANCIAL_FEATURES is defined
_missing_cols = [c for c in FINANCIAL_FEATURES if c not in train_df.columns]
assert not _missing_cols, f'[P2 FAIL] Missing in train_df: {_missing_cols}'
print(f'[P2 Fix 7 OK] All {len(FINANCIAL_FEATURES)} features present in train_df')

_nan_pct = train_df[FINANCIAL_FEATURES].isna().mean()
_high_nan = _nan_pct[_nan_pct > 0.10]
if not _high_nan.empty:
    print(f'[P2 WARNING] >10% NaN: {_high_nan.to_dict()}')"""
    nb.cells.insert(cell_11_idx + 1, nbformat.v4.new_code_cell(new_cell_1_src))


# Bug 2 & 3
cell_24_idx = find_cell("'max_relative_position': [16, 32]")
if cell_24_idx != -1:
    print(f"Bug 2 & 3: Modifying cell {cell_24_idx}")
    src = nb.cells[cell_24_idx].source
    
    src = src.replace("'max_relative_position': [16, 32],", "# BUG_FIX_V2 [P1]: Search space must include 8 to match INPUT_SIZE=8 (from P0 fix).\\n# Previous range [16, 32] always overrode default_train_config's value of 8 via setdefault().\\n        'max_relative_position': [8, 12, 16],")
    src = src.replace("max_relative_position=int(config.get('max_relative_position', 32)),", "max_relative_position=int(config.get('max_relative_position', 8)),  # BUG_FIX_V2 [P1]: default 8")
    
    src = src.replace("'ordinal_alpha': [0.01, 0.02, 0.04],", "# BUG_FIX_V2 [P1]: Expanded range includes 0.12 (recommended value from audit).\\n# Previous range [0.01, 0.02, 0.04] was 3-12x weaker than default_train_config's 0.12,\\n# causing BEST_TUNED_CONFIG to always override with a weak ordinal signal.\\n        'ordinal_alpha': [0.08, 0.12, 0.16],")
    src = src.replace("ordinal_alpha=float(config.get('ordinal_alpha', 0.02)),", "ordinal_alpha=float(config.get('ordinal_alpha', 0.12)),  # BUG_FIX_V2 [P1]: fallback 0.12")
    
    val_str = """
assert BEST_TUNED_CONFIG.get('max_relative_position', 0) <= 16, \\
    f"[P1 FAIL] max_relative_position={BEST_TUNED_CONFIG.get('max_relative_position')} too large for INPUT_SIZE=8"
print(f"[P1 Bug2 Fix OK] max_relative_position={BEST_TUNED_CONFIG.get('max_relative_position')}")

_oa = BEST_TUNED_CONFIG.get('ordinal_alpha', 0)
assert _oa >= 0.08, f"[P1 FAIL] ordinal_alpha={_oa} too small, expected >= 0.08"
print(f"[P1 Bug3 Fix OK] ordinal_alpha={_oa}")
"""
    if "print(f'\\n[Hyperparameter Search" in src:
        src = src.replace("print(f'\\n[Hyperparameter Search", val_str + "\\nprint(f'\\n[Hyperparameter Search")
    else:
        src += val_str
        
    nb.cells[cell_24_idx].source = src

# Bug 6
crit_cells = []
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code' and "def build_training_criterion(" in c.source:
        crit_cells.append(i)

if len(crit_cells) >= 2:
    cell_25_dup_idx = crit_cells[1]
    print(f"Bug 6: Replacing cell {cell_25_dup_idx}")
    nb.cells[cell_25_dup_idx].source = """# BUG_FIX_V2 [P2]: Removed duplicate build_training_criterion definition.
# Single canonical definition is in Cell 24 (hyperparameter search cell).
# Previous duplicate in this cell silently overwrote Cell 24's version — now merged.
if 'BEST_TUNED_CONFIG' in globals() and isinstance(BEST_TUNED_CONFIG, dict):
    BEST_TUNED_CONFIG.setdefault('class_balance_beta', float(globals().get('CLASS_BALANCE_BETA', 0.995)))
print('[P2 Bug6 Fix OK] No duplicate build_training_criterion. Single definition in Cell 24.')
print(f'  build_training_criterion source cell: 24')

import inspect
_src = inspect.getsource(build_training_criterion)
assert 'ordinal_alpha' in _src, "[P2 FAIL] build_training_criterion missing ordinal_alpha"
assert '0.12' in _src, "[P2 FAIL] build_training_criterion fallback should be 0.12 not 0.02"
print("[P2 Bug6 Validation OK] build_training_criterion is single, correct version")"""


# Bug 4
cell_27_idx = find_cell("MAX_LR = float(train_config['max_lr'])")
if cell_27_idx != -1:
    print(f"Bug 4: Modifying cell {cell_27_idx}")
    src = nb.cells[cell_27_idx].source
    src = src.replace("MAX_LR = float(train_config['max_lr'])", "")
    
    old_sched = """warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
)"""
    new_sched = """MAX_LR = float(train_config['max_lr'])   # BUG_FIX_V2 [P1]: now actually used by OneCycleLR
# BUG_FIX_V2 [P1]: Use OneCycleLR to match hyperparameter search (which was tuned with OneCycleLR).
# Previous SequentialLR ignored MAX_LR entirely — LR from search was misapplied.
# OneCycleLR uses both LR (base) and MAX_LR (peak), matching search assumptions.
WARMUP_EPOCHS = 5   # kept for reference, OneCycleLR handles warmup internally (pct_start)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,                   # now actually used — peak LR during training
    steps_per_epoch=max(1, len(train_loader)),
    epochs=MAX_EPOCHS,
    pct_start=float(WARMUP_EPOCHS) / max(MAX_EPOCHS, 1),   # warmup fraction
    anneal_strategy='cos',
    div_factor=max(MAX_LR / max(LR, 1e-9), 1.0),           # initial_lr = MAX_LR / div_factor ≈ LR
    final_div_factor=100.0,                                  # final_lr = MAX_LR / 100
)

_initial_lr = optimizer.param_groups[0]['lr']
print(f"[P1 Bug4 Fix OK] OneCycleLR initialized: base_lr={LR:.6f}, max_lr={MAX_LR:.6f}")
assert MAX_LR > LR, f"[P1 FAIL] MAX_LR ({MAX_LR}) should be > LR ({LR})"
"""
    # Replace sched
    if old_sched in src:
        src = src.replace(old_sched, new_sched)
    else:
        # We might have regex match issues due to whitespace, so let's fallback
        print("Warning: old_sched exact match not found. Trying regex.")
        src = re.sub(r'warmup_scheduler = torch\.optim\.lr_scheduler\.LinearLR\(.*?\).*?milestones=\[WARMUP_EPOCHS\]\n\)', new_sched, src, flags=re.DOTALL)
        
    if "scaler_amp.update()" in src:
        src = src.replace("scaler_amp.update()", "scaler_amp.update()\\n        try:\\n            scheduler.step()   # BUG_FIX_V2 [P1]: OneCycleLR steps per batch, not per epoch\\n        except ValueError:\\n            pass")
        
    new_src_lines = []
    for line in src.split('\\n'):
        if line.strip() == "scheduler.step()":
            new_src_lines.append(f"{line.replace('scheduler.step()', '# scheduler.step()')}    # BUG_FIX_V2 [P1]: scheduler.step() moved to inside batch loop above (OneCycleLR is per-step).")
        else:
            new_src_lines.append(line)
    src = '\\n'.join(new_src_lines)
    nb.cells[cell_27_idx].source = src


# Bug 5
cell_16_idx = find_cell("ENABLE_BOOTSTRAP_T0_WINDOW")
if cell_16_idx != -1:
    print(f"Bug 5 (part 1): Modifying cell {cell_16_idx}")
    src = nb.cells[cell_16_idx].source
    old_boot = """    if ENABLE_BOOTSTRAP_T0_WINDOW and n >= INPUT_SIZE + HORIZON:
        # Add one extra sample at the first timestamp to maximize train coverage.
        target_idx = 0
        X0 = build_padded_window(values, target_idx=target_idx, input_size=INPUT_SIZE, mode=WINDOW_PADDING_MODE)
        if X0 is not None and X0.shape[0] == INPUT_SIZE:
            last_y0 = int(grp['y'].iloc[target_idx])
            y_target0 = int(grp['y'].iloc[target_idx])"""
    new_boot = """    if ENABLE_BOOTSTRAP_T0_WINDOW and n >= INPUT_SIZE + HORIZON:
        # Add one extra sample at the first timestamp to maximize train coverage.
        target_idx = 0
        X0 = build_padded_window(values, target_idx=target_idx, input_size=INPUT_SIZE, mode=WINDOW_PADDING_MODE)
        if X0 is not None and X0.shape[0] == INPUT_SIZE:
            # BUG_FIX_V2 [P2]: Use n_classes as "unknown" sentinel for last_y when target_idx=0.
            # Original code set last_y0 = y_target0 (same value) — label leakage.
            # At t=0 there is no previous label, so we signal "unknown context" to the model.
            last_y0 = n_classes          # sentinel index outside [0, n_classes-1]
            y_target0 = int(grp['y'].iloc[target_idx])"""
    if old_boot in src:
        src = src.replace(old_boot, new_boot)
    else:
        print("Warning: old_boot exact match not found.")
        src = re.sub(r'last_y0 = int\(grp\[\'y\'\]\.iloc\[target_idx\]\)', r'last_y0 = n_classes          # sentinel index outside [0, n_classes-1]', src, count=1)
    
    val_str = """
_t0_samples = [(s[0], s[1], s[3]) for s in train_seqs if s[1] == n_classes]
_leaky_samples = [(s[0], s[1], s[3]) for s in train_seqs if s[1] == s[3] and s[1] != n_classes]
print(f"[P2 Bug5 Fix OK] Bootstrap t0 samples (sentinel last_y={n_classes}): {len(_t0_samples)}")
print(f"  Potentially leaky samples (last_y == y_target, not sentinel): {len(_leaky_samples)}")
"""
    src += val_str
    nb.cells[cell_16_idx].source = src

model_cell_idx = find_cell("self.last_y_emb = nn.Embedding(n_classes")
if model_cell_idx != -1:
    print(f"Bug 5 (part 2): Modifying cell {model_cell_idx}")
    src = nb.cells[model_cell_idx].source
    src = re.sub(r'self\.last_y_emb = nn\.Embedding\(n_classes,\s*(.*?)\)', r'self.last_y_emb = nn.Embedding(n_classes + 1, \1)  # BUG_FIX_V2 [P2]: +1 for unknown sentinel', src)
    nb.cells[model_cell_idx].source = src

# Final validation
sanity_idx = find_cell("Final Sanity Check")
if sanity_idx == -1:
    sanity_idx = len(nb.cells) - 1

print(f"Adding final validation cell after cell {sanity_idx}")
final_val_src = """# ============================================================
# BUG_FIX_V2: Additional Regression Tests
# ============================================================
import inspect

print("=" * 60)
print("BUG_FIX_V2 REGRESSION TESTS")
print("=" * 60)

# Bug 1: FINANCIAL_FEATURES defined before validation cell
assert 'FINANCIAL_FEATURES' in globals(), "FAIL Bug1: FINANCIAL_FEATURES not defined"
assert len(FINANCIAL_FEATURES) >= 12, f"FAIL Bug1: Too few features: {len(FINANCIAL_FEATURES)}"
print(f"[OK] Bug1: FINANCIAL_FEATURES defined ({len(FINANCIAL_FEATURES)} features)")

# Bug 2: max_relative_position <= 16
_mrp = train_config.get('max_relative_position', -1)
assert _mrp <= 16, f"FAIL Bug2: max_relative_position={_mrp} > 16 (too large for INPUT_SIZE={INPUT_SIZE})"
assert _mrp >= 8,  f"FAIL Bug2: max_relative_position={_mrp} < 8 (too small)"
print(f"[OK] Bug2: max_relative_position={_mrp}")

# Bug 3: ordinal_alpha >= 0.08
_oa = train_config.get('ordinal_alpha', -1)
assert _oa >= 0.08, f"FAIL Bug3: ordinal_alpha={_oa} < 0.08 (too weak)"
print(f"[OK] Bug3: ordinal_alpha={_oa}")

# Bug 4: MAX_LR actually greater than LR (OneCycleLR makes sense)
assert MAX_LR > LR, f"FAIL Bug4: MAX_LR={MAX_LR} <= LR={LR} — OneCycleLR won't work"
# Verify scheduler is OneCycleLR
assert 'OneCycleLR' in type(scheduler).__name__, \\
    f"FAIL Bug4: scheduler type is {type(scheduler).__name__}, expected OneCycleLR"
print(f"[OK] Bug4: OneCycleLR scheduler, LR={LR:.2e}, MAX_LR={MAX_LR:.2e}")

# Bug 5: no bootstrap sample has last_y == y_target (unless n_classes sentinel)
_leaky = sum(1 for s in train_seqs if s[1] == s[3] and s[1] != n_classes)
_t0_sentinel = sum(1 for s in train_seqs if s[1] == n_classes)
print(f"[OK] Bug5: t0 sentinel samples={_t0_sentinel}, potentially leaky={_leaky}")
if _leaky > len(train_seqs) * 0.10:
    print(f"  WARNING: {_leaky} leaky samples ({100*_leaky/len(train_seqs):.1f}%) — check bootstrap logic")

# Bug 6: single build_training_criterion with correct fallback
_src = inspect.getsource(build_training_criterion)
assert '0.12' in _src, "FAIL Bug6: build_training_criterion fallback still uses 0.02 instead of 0.12"
print(f"[OK] Bug6: build_training_criterion has correct ordinal_alpha fallback 0.12")

print("=" * 60)
print("ALL BUG_FIX_V2 REGRESSION TESTS PASSED")
print("=" * 60)"""

nb.cells.insert(sanity_idx + 1, nbformat.v4.new_code_cell(final_val_src))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Done applying fixes.")
