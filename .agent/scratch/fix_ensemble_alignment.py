"""
Fix row misalignment in all 6 ensemble notebooks (KB7-KB12).

Problem: Ensemble notebooks generate y_val/y_test from CSV (raw row order),
but proba .npy files are in DataLoader order (grouped by ticker, sorted by date).

Fix: Load y_val/y_test from {model}_y_val.npy files (aligned with probas)
instead of generating from CSV.

How to Run:
  python e:/thesis/.agent/scratch/fix_ensemble_alignment.py

Expected Output:
  All 6 notebooks updated to load aligned y_val/y_test from .npy files.
"""
import json
from pathlib import Path

NB_DIR = Path("e:/thesis/notebooks")
ENSEMBLE_NBS = [
    "KB7_FI-TTX.ipynb",
    "KB8_FI-PLL.ipynb",
    "KB9_FI-TTLPXL.ipynb",
    "KB10_FR-TTX.ipynb",
    "KB11_FR-PLL.ipynb",
    "KB12_FR-TTLPXL.ipynb",
]

changes_log = []

for nb_name in ENSEMBLE_NBS:
    nb_path = NB_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    nb_changes = []

    # ================================================================
    # Fix Cell 6: Load y_val/y_test from aligned .npy instead of CSV
    # ================================================================
    cell6 = nb["cells"][6]
    src6 = "".join(cell6["source"])

    # Find the block that creates y_val/y_test from CSV and replace it
    old_label_block = """le = LabelEncoder().fit(df_train[TARGET_COL].astype(str))
y_train = le.transform(df_train[TARGET_COL].astype(str))
y_val   = le.transform(df_val[TARGET_COL].astype(str))
y_test  = le.transform(df_test[TARGET_COL].astype(str))
n_classes   = len(le.classes_)
class_names = list(le.classes_)

np.save(str(ARTIFACT_DIR/'y_val.npy'),  y_val)
np.save(str(ARTIFACT_DIR/'y_test.npy'), y_test)"""

    new_label_block = """le = LabelEncoder().fit(df_train[TARGET_COL].astype(str))
y_train = le.transform(df_train[TARGET_COL].astype(str))
n_classes   = len(le.classes_)
class_names = list(le.classes_)

# ── Load y_val / y_test aligned with proba .npy files ──────────────
# Proba files are generated in DataLoader order (ticker-grouped, date-sorted),
# NOT in CSV row order. We must use the y_val/y_test that were saved
# alongside the probas to ensure row alignment.
_ref_model_key = ART_MAP[MODEL_KEYS[0]]
_y_val_path  = PROBA_INPUT_DIR / f'{_ref_model_key}_y_val.npy'
_y_test_path = PROBA_INPUT_DIR / f'{_ref_model_key}_y_test.npy'

if _y_val_path.exists() and _y_test_path.exists():
    y_val  = np.load(str(_y_val_path)).astype(int)
    y_test = np.load(str(_y_test_path)).astype(int)
    print(f'[OK] Loaded aligned y_val from {_y_val_path.name} ({len(y_val)} rows)')
    print(f'[OK] Loaded aligned y_test from {_y_test_path.name} ({len(y_test)} rows)')
else:
    # Fallback to CSV-based labels (will work only if probas are also CSV-ordered)
    y_val  = le.transform(df_val[TARGET_COL].astype(str))
    y_test = le.transform(df_test[TARGET_COL].astype(str))
    print('[WARN] Aligned y_val/y_test .npy not found, using CSV-order labels.')
    print('       This may cause row misalignment with proba files!')

np.save(str(ARTIFACT_DIR/'y_val.npy'),  y_val)
np.save(str(ARTIFACT_DIR/'y_test.npy'), y_test)"""

    if old_label_block in src6:
        src6 = src6.replace(old_label_block, new_label_block)
        cell6["source"] = [line + "\n" for line in src6.splitlines()]
        nb_changes.append("Cell 6: Fixed y_val/y_test to load from aligned .npy files")
    else:
        # Check if already fixed
        if "_ref_model_key" in src6:
            nb_changes.append("Cell 6: Already fixed")
        else:
            print(f"  [WARN] {nb_name}: Cannot find old label block in Cell 6")
            # Print what we have for debugging
            for i, line in enumerate(src6.split("\n")):
                if "y_val" in line and "le.transform" in line:
                    print(f"    Found at line {i}: {line.strip()[:120]}")

    # ================================================================
    # Ensure MODEL_KEYS and ART_MAP are accessible from Cell 6
    # Since Cell 6 runs BEFORE the cell with MODEL_KEYS (Cell 8/9),
    # we need to move the MODEL_KEYS/ART_MAP definitions earlier
    # OR reference them in Cell 6.
    # ================================================================
    # Find the cell with MODEL_KEYS
    model_keys_cell_idx = None
    model_keys_src = None
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            s = "".join(cell.get("source", []))
            if "MODEL_KEYS" in s and "ART_MAP" in s and "_proba.npy" in s:
                model_keys_cell_idx = idx
                model_keys_src = s
                break

    if model_keys_cell_idx and model_keys_cell_idx > 6:
        # Extract MODEL_KEYS and ART_MAP lines from that cell
        mk_lines = []
        for line in model_keys_src.split("\n"):
            if line.startswith("MODEL_KEYS") or line.startswith("MODEL_NAMES") or line.startswith("ART_MAP"):
                mk_lines.append(line)
            elif mk_lines and (line.startswith("           ") or line.startswith("          ")):
                mk_lines.append(line)  # continuation lines of ART_MAP

        if mk_lines:
            # Add MODEL_KEYS/ART_MAP to Cell 6, before the label loading block
            src6_current = "".join(cell6["source"])
            if "MODEL_KEYS" not in src6_current:
                # Find position: right before TARGET_COL definition
                target_marker = "TARGET_COL = 'rating_detail'"
                if target_marker in src6_current:
                    mk_block = "\n".join(mk_lines) + "\n\n"
                    src6_current = src6_current.replace(
                        target_marker,
                        mk_block + target_marker
                    )
                    cell6["source"] = [line + "\n" for line in src6_current.splitlines()]
                    nb_changes.append(f"Cell 6: Added MODEL_KEYS/ART_MAP from Cell {model_keys_cell_idx}")

    # Save
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    changes_log.append((nb_name, nb_changes))

# ================================================================
# Report
# ================================================================
print("=" * 60)
print("CHANGES APPLIED:")
print("=" * 60)
for nb_name, changes in changes_log:
    print(f"\n  {nb_name}:")
    for c in changes:
        print(f"    [OK] {c}")

# ================================================================
# Verification
# ================================================================
print("\n" + "=" * 60)
print("VERIFICATION:")
print("=" * 60)

all_ok = True
for nb_name in ENSEMBLE_NBS:
    nb_path = NB_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell6_v = "".join(nb["cells"][6]["source"])
    has_aligned_load = "_ref_model_key" in cell6_v and "_y_val_path" in cell6_v
    has_model_keys = "MODEL_KEYS" in cell6_v and "ART_MAP" in cell6_v

    ok = has_aligned_load and has_model_keys
    status = "PASS" if ok else "FAIL"
    details = []
    if not has_aligned_load:
        details.append("missing aligned y_val load")
    if not has_model_keys:
        details.append("missing MODEL_KEYS in Cell 6")

    detail_str = f" ({', '.join(details)})" if details else ""
    print(f"  [{status}] {nb_name}{detail_str}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print(f"SUCCESS: All {len(ENSEMBLE_NBS)} ensemble notebooks fixed!")
else:
    print("PARTIAL: Some notebooks may need manual review.")
print("=" * 60)
