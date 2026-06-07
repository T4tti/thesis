import json
from pathlib import Path

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells_to_extract = [2, 5, 6, 8, 13]
for cell_idx in cells_to_extract:
    cell = nb["cells"][cell_idx]
    output_path = Path(f"e:/thesis/scratch/cell_{cell_idx}.py")
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("".join(cell["source"]))
    print(f"Extracted Cell {cell_idx} to {output_path}")
