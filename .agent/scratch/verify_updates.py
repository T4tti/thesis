import json
from pathlib import Path

notebooks = [
    "notebooks/lstm-baseline.ipynb",
    "notebooks/tcn-baseline.ipynb",
    "notebooks/patchtst-baseline.ipynb",
    "notebooks/xgboost-baseline.ipynb",
    "notebooks/lightgbm-baseline.ipynb"
]

all_passed = True

for nb_path in notebooks:
    path = Path("e:/thesis") / nb_path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    found_old = False
    found_new = False
    
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "if n >= input_size + horizon:" in src:
                found_old = True
            if "last_y_idx = max(0, t - 1)" in src:
                found_new = True
                
    print(f"=== {nb_path} ===")
    print(f"  Old logic ('if n >= input_size + horizon:'): {'FOUND' if found_old else 'NOT FOUND (OK)'}")
    print(f"  New logic ('last_y_idx = max(0, t - 1)'): {'FOUND (OK)' if found_new else 'NOT FOUND'}")
    
    if found_old or not found_new:
        print("  FAIL: Verification failed for this notebook!")
        all_passed = False
    else:
        print("  PASS: Verification succeeded!")

if all_passed:
    print("\nSUCCESS: All notebooks verified successfully!")
else:
    print("\nFAILURE: One or more notebooks failed verification!")
