"""
Apply Strategy C, D, E, F to Transformer-BiLSTM notebook.

  C: Reduce model complexity (params 1.18M -> ~300K)
  D: Disable persistence calibration blend
  E: Increase ordinal_alpha (0.01 -> 0.08)
  F: Stronger regularization (weight_decay, label_smoothing)

How to Run:
  python e:/thesis/.agent/scratch/apply_strategy_cdef.py

Expected Output:
  Confirmation messages for each strategy applied.
"""
import json
from pathlib import Path

NB_PATH = Path("e:/thesis/notebooks/Transformer-BiLSTM.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = []

# ============================================================
# STRATEGY C + E + F: Update default_train_config AND add
# strategy override block in Cell 26
# ============================================================
cell26 = nb["cells"][26]
src26 = "".join(cell26["source"])

# --- C1: Update default_train_config values ---
replacements_config = {
    "'hidden_size': 128,":   "'hidden_size': 64,       # Strategy C: reduced from 128",
    "'dropout': 0.15,":      "'dropout': 0.25,         # Strategy C: increased from 0.15 for regularization",
    "'d_model': 128,":       "'d_model': 64,           # Strategy C: reduced from 128",
    "'n_layers': 3,":        "'n_layers': 2,           # Strategy C: reduced from 3",
    "'ticker_emb_dim': 32,": "'ticker_emb_dim': 4,     # Strategy C: reduced from 32 to limit memorization",
    "'company_emb_dim': 32,":"'company_emb_dim': 4,    # Strategy C: reduced from 32 to limit memorization",
    "'weight_decay': 2e-3,": "'weight_decay': 5e-3,    # Strategy F: increased from 2e-3 for stronger L2",
    "'label_smoothing': 0.0,": "'label_smoothing': 0.05,  # Strategy F: increased from 0.0",
}

for old, new in replacements_config.items():
    if old in src26:
        src26 = src26.replace(old, new)
        changes.append(f"Config: {old.strip()} -> {new.strip()}")

# --- C2/E/F: Add strategy override block after setdefault merge ---
# This ensures our strategy values override BEST_TUNED_CONFIG
old_merge = """train_config = dict(globals().get('BEST_TUNED_CONFIG', {}))
for k, v in default_train_config.items():
    train_config.setdefault(k, v)"""

new_merge = """train_config = dict(globals().get('BEST_TUNED_CONFIG', {}))
for k, v in default_train_config.items():
    train_config.setdefault(k, v)

# Strategy C+E+F overrides: force these values over BEST_TUNED_CONFIG
# Rationale: search was run with INPUT_SIZE=1 (degenerate); these values
# are tuned for the new INPUT_SIZE=4 regime with temporal modeling active.
_strategy_overrides = {
    'hidden_size': 64,        # C: reduce overfitting (1.18M -> ~300K params)
    'd_model': 64,            # C: match reduced hidden_size
    'n_layers': 2,            # C: fewer transformer blocks
    'dropout': 0.25,          # C: stronger dropout
    'ticker_emb_dim': 4,      # C: limit entity memorization
    'company_emb_dim': 4,     # C: limit entity memorization
    'ordinal_alpha': 0.08,    # E: stronger ordinal signal (was 0.01)
    'weight_decay': 5e-3,     # F: stronger L2 regularization
    'label_smoothing': 0.05,  # F: prevent overconfident predictions
}
train_config.update(_strategy_overrides)
print(f"[Strategy C+E+F] Applied {len(_strategy_overrides)} config overrides over BEST_TUNED_CONFIG")"""

if old_merge in src26:
    src26 = src26.replace(old_merge, new_merge)
    changes.append("Strategy C+E+F: Added _strategy_overrides block after config merge")
else:
    raise ValueError("Cannot find merge block in Cell 26")

# --- C3: Update fallback defaults in variable extraction lines ---
fallback_replacements = {
    "int(train_config.get('d_model', 128))":     "int(train_config.get('d_model', 64))",
    "int(train_config.get('n_layers', 3))":      "int(train_config.get('n_layers', 2))",
    "int(train_config.get('ticker_emb_dim', 32))": "int(train_config.get('ticker_emb_dim', 4))",
    "int(train_config.get('company_emb_dim', 32))": "int(train_config.get('company_emb_dim', 4))",
    "int(train_config.get('max_relative_position', 32))": "int(train_config.get('max_relative_position', 8))",
}

for old, new in fallback_replacements.items():
    if old in src26:
        src26 = src26.replace(old, new)

# Write Cell 26 back
cell26["source"] = [line + "\n" for line in src26.splitlines()]
changes.append("Strategy C: Updated fallback defaults in variable extraction")


# ============================================================
# STRATEGY D: Disable persistence calibration (Cell 29)
# ============================================================
cell29 = nb["cells"][29]
src29 = "".join(cell29["source"])

# D1: Disable calibrated class threshold
old_cal = "USE_CALIBRATED_CLASS_THRESHOLD = True"
new_cal = "USE_CALIBRATED_CLASS_THRESHOLD = False  # Strategy D: disabled to prevent false positives from persistence blend"

if old_cal in src29:
    src29 = src29.replace(old_cal, new_cal)
    changes.append("Strategy D1: USE_CALIBRATED_CLASS_THRESHOLD = False")

# D2: Reduce the class-conditional blend weights
old_blend_w = "_class_blend_w = torch.tensor([0.0, 0.30, 0.70]"
new_blend_w = "_class_blend_w = torch.tensor([0.0, 0.0, 0.0]  # Strategy D: zeroed out to disable persistence blend"

if old_blend_w in src29:
    src29 = src29.replace(old_blend_w, new_blend_w)
    changes.append("Strategy D2: Zeroed persistence blend weights [0.0, 0.0, 0.0]")

# Write Cell 29 back
cell29["source"] = [line + "\n" for line in src29.splitlines()]


# ============================================================
# Save notebook
# ============================================================
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("=" * 60)
print("CHANGES APPLIED:")
for c in changes:
    print(f"  [OK] {c}")
print("=" * 60)

# ============================================================
# Verification
# ============================================================
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb_v = json.load(f)

cell26_v = "".join(nb_v["cells"][26]["source"])
cell29_v = "".join(nb_v["cells"][29]["source"])

checks = {
    # Strategy C
    "C: hidden_size=64 in config": "'hidden_size': 64," in cell26_v,
    "C: d_model=64 in config":     "'d_model': 64," in cell26_v,
    "C: n_layers=2 in config":     "'n_layers': 2," in cell26_v,
    "C: dropout=0.25 in config":   "'dropout': 0.25," in cell26_v,
    "C: ticker_emb_dim=4":         "'ticker_emb_dim': 4," in cell26_v,
    "C: company_emb_dim=4":        "'company_emb_dim': 4," in cell26_v,
    "C: strategy_overrides block": "_strategy_overrides" in cell26_v,
    # Strategy D
    "D: calibration disabled":     "USE_CALIBRATED_CLASS_THRESHOLD = False" in cell29_v,
    "D: blend weights zeroed":     "[0.0, 0.0, 0.0]" in cell29_v,
    # Strategy E
    "E: ordinal_alpha=0.08":       "'ordinal_alpha': 0.08," in cell26_v,
    # Strategy F
    "F: weight_decay=5e-3":        "'weight_decay': 5e-3," in cell26_v,
    "F: label_smoothing=0.05":     "'label_smoothing': 0.05," in cell26_v,
}

print("\nVERIFICATION:")
all_ok = True
for name, ok in checks.items():
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        all_ok = False

if all_ok:
    print("\nSUCCESS: All Strategy C+D+E+F changes verified!")
else:
    print("\nFAILED: Some checks failed!")
