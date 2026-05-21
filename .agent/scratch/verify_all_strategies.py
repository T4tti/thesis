"""
Final comprehensive verification of ALL strategies A-F.

How to Run:
  python e:/thesis/.agent/scratch/verify_all_strategies.py

Expected Output:
  17/17 checks should PASS.
"""
import json
from pathlib import Path

path = Path("e:/thesis/notebooks/Transformer-BiLSTM.ipynb")
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell15 = "".join(nb["cells"][15]["source"])
cell26 = "".join(nb["cells"][26]["source"])
cell29 = "".join(nb["cells"][29]["source"])

print("=" * 60)
print("COMPREHENSIVE VERIFICATION: ALL STRATEGIES A-F")
print("=" * 60)

# Filter out comment lines from cell26 for SequentialLR check
code_lines_26 = [l for l in cell26.split("\n") if not l.strip().startswith("#")]
seq_in_code = any("SequentialLR" in l for l in code_lines_26)

checks = [
    # Strategy A
    ("A: INPUT_SIZE_DEFAULT = 4", "INPUT_SIZE_DEFAULT = 4" in cell15),

    # Strategy B
    ("B: OneCycleLR active", "scheduler = torch.optim.lr_scheduler.OneCycleLR(" in cell26),
    ("B: SequentialLR removed from code", not seq_in_code),
    ("B: Per-batch step enabled", "SCHEDULER_IS_PER_BATCH = True" in cell26),

    # Strategy C
    ("C: hidden_size = 64", "'hidden_size': 64," in cell26),
    ("C: d_model = 64", "'d_model': 64," in cell26),
    ("C: n_layers = 2", "'n_layers': 2," in cell26),
    ("C: dropout = 0.25", "'dropout': 0.25," in cell26),
    ("C: ticker_emb_dim = 4", "'ticker_emb_dim': 4," in cell26),
    ("C: company_emb_dim = 4", "'company_emb_dim': 4," in cell26),
    ("C: Strategy overrides block", "_strategy_overrides" in cell26 and "train_config.update(_strategy_overrides)" in cell26),

    # Strategy D
    ("D: Calibration disabled", "USE_CALIBRATED_CLASS_THRESHOLD = False" in cell29),
    ("D: Blend weights zeroed", "[0.0, 0.0, 0.0]" in cell29),
    ("D: Blend line valid syntax", "torch.tensor([0.0, 0.0, 0.0], device=" in cell29),

    # Strategy E
    ("E: ordinal_alpha = 0.08 in overrides", "'ordinal_alpha': 0.08," in cell26),

    # Strategy F
    ("F: weight_decay = 5e-3", "'weight_decay': 5e-3," in cell26),
    ("F: label_smoothing = 0.05", "'label_smoothing': 0.05," in cell26),
]

all_ok = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print(f"ALL {len(checks)}/{len(checks)} CHECKS PASSED")
    print("Strategies A through F applied successfully!")
else:
    failed = sum(1 for _, ok in checks if not ok)
    print(f"FAILED: {failed}/{len(checks)} checks failed!")
print("=" * 60)
