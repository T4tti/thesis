"""
Update ensemble notebooks (KB7-KB12) to load .npy proba files from
Kaggle input dataset path instead of ARTIFACT_DIR.

Input path: /kaggle/input/datasets/tailength/model-proba/proba/
Files expected: {model}_val_proba.npy, {model}_test_proba.npy

How to Run:
  python e:/thesis/.agent/scratch/update_ensemble_paths.py

Expected Output:
  Confirmation for each of the 6 notebooks updated.
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

KAGGLE_PROBA_PATH = "/kaggle/input/datasets/tailength/model-proba/proba"

changes_log = []

for nb_name in ENSEMBLE_NBS:
    nb_path = NB_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    nb_changes = []

    # ================================================================
    # 1. Add PROBA_INPUT_DIR to Cell 2 (path setup)
    # ================================================================
    cell2 = nb["cells"][2]
    src2 = "".join(cell2["source"])

    if "PROBA_INPUT_DIR" not in src2:
        # Add after ARTIFACT_DIR line
        old_art = "ARTIFACT_DIR = PROJECT_ROOT / 'credit_rating_artifacts'\nARTIFACT_DIR.mkdir(parents=True, exist_ok=True)"
        new_art = f"""ARTIFACT_DIR = PROJECT_ROOT / 'credit_rating_artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle input dataset for pre-computed model probabilities (.npy)
PROBA_INPUT_DIR = Path('{KAGGLE_PROBA_PATH}') if IN_KAGGLE else ARTIFACT_DIR"""

        if old_art in src2:
            src2 = src2.replace(old_art, new_art)
            cell2["source"] = [line + "\n" for line in src2.splitlines()]
            nb_changes.append("Cell 2: Added PROBA_INPUT_DIR")
        else:
            print(f"  [WARN] {nb_name}: Cannot find ARTIFACT_DIR block in Cell 2")
    else:
        nb_changes.append("Cell 2: PROBA_INPUT_DIR already present")

    # ================================================================
    # 2. Update proba loading cell to use PROBA_INPUT_DIR
    # ================================================================
    # Find the cell that loads _val_proba.npy / _test_proba.npy
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "_val_proba.npy" in src and "_test_proba.npy" in src and "MODEL_KEYS" in src:
                # Replace ARTIFACT_DIR with PROBA_INPUT_DIR for reading npy
                old_vp = "vp_path = ARTIFACT_DIR / f'{art}_val_proba.npy'"
                new_vp = "vp_path = PROBA_INPUT_DIR / f'{art}_val_proba.npy'"
                old_tp = "tp_path = ARTIFACT_DIR / f'{art}_test_proba.npy'"
                new_tp = "tp_path = PROBA_INPUT_DIR / f'{art}_test_proba.npy'"

                if old_vp in src:
                    src = src.replace(old_vp, new_vp)
                    src = src.replace(old_tp, new_tp)
                    cell["source"] = [line + "\n" for line in src.splitlines()]
                    nb_changes.append(f"Cell {idx}: Changed proba loading to PROBA_INPUT_DIR")
                elif "PROBA_INPUT_DIR" in src:
                    nb_changes.append(f"Cell {idx}: Already using PROBA_INPUT_DIR")
                else:
                    print(f"  [WARN] {nb_name} Cell {idx}: Unexpected proba path format")
                break

    # ================================================================
    # 3. Save notebook
    # ================================================================
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

    cell2_v = "".join(nb["cells"][2]["source"])
    has_proba_dir = "PROBA_INPUT_DIR" in cell2_v
    has_kaggle_path = KAGGLE_PROBA_PATH in cell2_v

    # Find proba loading cell
    proba_ok = False
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "_val_proba.npy" in src and "PROBA_INPUT_DIR" in src:
                proba_ok = True
                break

    ok = has_proba_dir and has_kaggle_path and proba_ok
    status = "PASS" if ok else "FAIL"
    details = []
    if not has_proba_dir:
        details.append("missing PROBA_INPUT_DIR")
    if not has_kaggle_path:
        details.append("missing Kaggle path")
    if not proba_ok:
        details.append("proba loading not updated")

    detail_str = f" ({', '.join(details)})" if details else ""
    print(f"  [{status}] {nb_name}{detail_str}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print(f"SUCCESS: All {len(ENSEMBLE_NBS)} ensemble notebooks updated!")
else:
    print("FAILED: Some notebooks not updated correctly!")
print("=" * 60)
