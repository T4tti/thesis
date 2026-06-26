# How to Run: python scratch/get_cell6.py
# Expected Output: Prints cell 6 of the timegan-3groups.ipynb notebook

import json
from pathlib import Path

def print_cell_content(notebook_path: Path, cell_idx: int) -> None:
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
        
    idx = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            idx += 1
            if idx == cell_idx:
                print(f"\n================ CODE CELL #{idx} ================")
                print("".join(cell.get("source", [])))

if __name__ == "__main__":
    print_cell_content(Path("notebooks/timegan-3groups.ipynb"), 6)
