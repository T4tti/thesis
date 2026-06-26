# How to Run: python scratch/get_cells.py
# Expected Output: Prints cells 6, 8, and 9 of the timegan-3groups.ipynb notebook

import json
from pathlib import Path

def print_cell_content(notebook_path: Path, cell_indices: list) -> None:
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
        
    cell_idx = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell_idx += 1
            if cell_idx in cell_indices:
                print(f"\n================ CODE CELL #{cell_idx} ================")
                print("".join(cell.get("source", [])))

if __name__ == "__main__":
    print_cell_content(Path("notebooks/timegan-3groups.ipynb"), [6, 8, 9])
