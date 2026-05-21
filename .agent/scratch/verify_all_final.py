"""
Final comprehensive verification of ALL strategies A through I.

How to Run:
  python e:/thesis/.agent/scratch/verify_all_final.py

Expected Output:
  All checks PASS.
"""
import json
from pathlib import Path

path = Path("e:/thesis/notebooks/Transformer-BiLSTM.ipynb")
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell15 = "".join(nb["cells"][15]["source"])
cell22 = "".join(nb["cells"][22]["source"])
cell26 = "".join(nb["cells"][26]["source"])
cell29 = "".join(nb["cells"][29]["source"])
cell31 = "".join(nb["cells"][31]["source"])

code_lines_26 = [l for l in cell26.split("\n") if not l.strip().startswith("#")]
seq_in_code = any("SequentialLR" in l for l in code_lines_26)

checks = [
    # Strategy A
    ("A: INPUT_SIZE_DEFAULT = 4", "INPUT_SIZE_DEFAULT = 4" in cell15),
    # Strategy B
    ("B: OneCycleLR active", "OneCycleLR(" in cell26),
    ("B: SequentialLR removed from code", not seq_in_code),
    ("B: Per-batch stepping", "SCHEDULER_IS_PER_BATCH = True" in cell26),
    # Strategy C
    ("C: hidden_size = 64", "'hidden_size': 64," in cell26),
    ("C: d_model = 64", "'d_model': 64," in cell26),
    ("C: n_layers = 2", "'n_layers': 2," in cell26),
    ("C: dropout = 0.25", "'dropout': 0.25," in cell26),
    ("C: ticker_emb_dim = 4", "'ticker_emb_dim': 4," in cell26),
    ("C: company_emb_dim = 4", "'company_emb_dim': 4," in cell26),
    # Strategy D
    ("D: Calibration disabled", "USE_CALIBRATED_CLASS_THRESHOLD = False" in cell29),
    ("D: Blend weights zeroed", "[0.0, 0.0, 0.0]" in cell29),
    # Strategy E
    ("E: ordinal_alpha = 0.08", "'ordinal_alpha': 0.08," in cell26),
    # Strategy F
    ("F: weight_decay = 5e-3", "'weight_decay': 5e-3," in cell26),
    ("F: label_smoothing = 0.05", "'label_smoothing': 0.05," in cell26),
    # Strategy G
    ("G: Sentinel at t=0", "last_y_val = n_classes" in cell15),
    ("G: Conditional last_y", "if target_idx == 0:" in cell15),
    ("G: Embedding n_classes+1", "n_classes + 1, hidden_size)" in cell22),
    ("G: Old leakage removed", "last_y_idx = max(0, target_idx - 1)" not in cell15),
    # Strategy H
    ("H: DISTRESSED_THRESHOLD", "DISTRESSED_THRESHOLD = 0.55" in cell31),
    ("H: False positive mask", "false_positive_mask" in cell31),
    # Strategy I
    ("I: focal_gamma = 2.5", "'focal_gamma': 2.5," in cell26),
]

print("=" * 60)
print("FINAL VERIFICATION: ALL STRATEGIES A-I")
print("=" * 60)

all_ok = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        all_ok = False

passed = sum(1 for _, ok in checks if ok)
total = len(checks)

print()
if all_ok:
    print(f"ALL {passed}/{total} CHECKS PASSED")
    print("Strategies A through I applied successfully!")
else:
    print(f"FAILED: {total - passed}/{total} checks failed!")
print("=" * 60)
