# How to Run: python scratch/inspect_tcn_notebook.py
# Expected Output: Prints all markdown cells and code cell structures related to interpretability/xAI.

import json
import sys
from pathlib import Path

# Force stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = Path("e:/thesis/notebooks/tcn-baseline.ipynb")

if not notebook_path.exists():
    print(f"File not found: {notebook_path}")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("\n--- Markdown Cells & Headings ---")
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"]).strip()
        if src.startswith("#"):
            print(f"Cell {idx} [MD]: {src.splitlines()[0]}")
        elif "xai" in src.lower() or "lime" in src.lower() or "shap" in src.lower() or "interpret" in src.lower() or "explain" in src.lower():
            print(f"Cell {idx} [MD] (Relevant): {src[:150]}...")

print("\n--- Detailed Content of xAI Cells (7, 8, 9) ---")
for idx in [7, 8, 9]:
    if idx < len(nb["cells"]):
        cell = nb["cells"][idx]
        print(f"\n================ CELL {idx} ({cell['cell_type'].upper()}) ================")
        print("".join(cell["source"]))

