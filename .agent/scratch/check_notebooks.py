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
    if not path.exists():
        print(f"Path does not exist: {path}")
        continue
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    found = False
    for idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "def build_sequences" in src or "def build_tabular_samples" in src:
                print(f"=== {nb_path} (Cell {idx}) ===")
                print(src[:400] + "\n...")
                found = True
                break
    if not found:
        print(f"!!! Function not found in {nb_path}")
