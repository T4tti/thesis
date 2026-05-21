import json
from pathlib import Path

notebooks = [
    "notebooks/lstm-baseline.ipynb",
    "notebooks/tcn-baseline.ipynb",
    "notebooks/patchtst-baseline.ipynb",
    "notebooks/xgboost-baseline.ipynb",
    "notebooks/lightgbm-baseline.ipynb"
]

for nb_path in notebooks:
    path = Path("e:/thesis") / nb_path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cell = data["cells"][2] # Cell 2
    src = "".join(cell.get("source", []))
    print(f"=== {nb_path} ===")
    # Print the part where build_sequences or build_tabular_samples is defined
    lines = src.split("\n")
    start_idx = -1
    for i, line in enumerate(lines):
        if "def build_sequences" in line or "def build_tabular_samples" in line or "def build_padded_window" in line:
            start_idx = i
            break
    if start_idx != -1:
        print("\n".join(lines[start_idx-2:start_idx+35]))
    else:
        print("Not found function in source lines")
    print("\n" + "="*40 + "\n")
