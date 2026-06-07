import json
from pathlib import Path
import sys

# Set standard output encoding to utf-8 for console printing
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

keywords = [
    "CHECKPOINT_CONFIG",
    "scheduled_context_mask",
    "SPARSE_GRAPH_HIDDEN",
    "SPARSE_GRAPH_LAYERS",
    "PERSISTENCE_PRIOR_SCALE",
    "CONTEXT_MASK",
    "apply_dropedge",
    "benchmark_ce",
    "ordinal_ce_emd",
    "optimizer",
    "scheduler",
    "LABEL_SMOOTHING",
    "dropout",
]

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        found = []
        for kw in keywords:
            if kw in source:
                found.append(kw)
        if found:
            print(f"Cell {idx}: contains {found}")
            # Print first 8 lines of the cell for context
            lines = cell["source"][:8]
            print("--- First few lines ---")
            for line in lines:
                print(f"  {line.rstrip()}")
            print("-----------------------")
