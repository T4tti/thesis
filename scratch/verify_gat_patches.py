"""
Verify the GAT notebook patches were applied correctly.

How to Run: python scratch/verify_gat_patches.py
Expected Output: Report showing each change was applied.
"""
import json
from pathlib import Path

notebook_path = Path(r'e:\thesis\notebooks\gat-baseline.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

checks = {
    "use_class_weights_true": False,
    "focal_gamma_2": False,
    "focal_weight_05": False,
    "ordinal_weight_005": False,
    "warmup_20": False,
    "scheduler_defined": False,
    "scheduler_step": False,
    "selection_accuracy_040": False,
    "selection_class0_025": False,
    "selection_chgacc_010": False,
    "class_balanced_knn": False,
}

for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])

    if "'use_class_weights': True," in source:
        checks["use_class_weights_true"] = True
    if "'focal_gamma': 2.0," in source:
        checks["focal_gamma_2"] = True
    if "'focal_weight': 0.5," in source:
        checks["focal_weight_05"] = True
    if "'ordinal_weight': 0.05," in source:
        checks["ordinal_weight_005"] = True
    if "'warmup_epochs': 20," in source:
        checks["warmup_20"] = True
    if "CosineAnnealingLR" in source:
        checks["scheduler_defined"] = True
    if "scheduler.step()" in source:
        checks["scheduler_step"] = True
    if "0.40 * metrics['Accuracy']" in source:
        checks["selection_accuracy_040"] = True
    if "0.25 * metrics['Class0_F2']" in source:
        checks["selection_class0_025"] = True
    if "0.10 * chg_acc" in source:
        checks["selection_chgacc_010"] = True
    if "Class-balanced KNN" in source:
        checks["class_balanced_knn"] = True

print("=" * 60)
print("  GAT Baseline Notebook - Patch Verification")
print("=" * 60)
all_passed = True
for name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"  [{status}] {name}")

print()
if all_passed:
    print("  All 11 checks passed!")
else:
    print("  WARNING: Some checks failed!")
print("=" * 60)
