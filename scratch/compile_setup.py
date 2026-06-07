import json
from pathlib import Path
import sys
import copy

# Ensure output is printed in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Concatenate setup code from cells 1 to 7
setup_code_lines = []
for idx in range(1, 8):
    cell = nb["cells"][idx]
    if cell["cell_type"] == "code":
        # Skip the model instantiation and configuration at the end of cell 6
        # because we will instantiate them dynamically in our ablation loop
        source_lines = cell["source"]
        source_code = "".join(source_lines)
        if idx == 6:
            # We want to keep the classes, but remove the global model/optimizer creation at the bottom
            lines = source_code.splitlines()
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith("model =") or line.strip().startswith("LOSS_CONFIG =") or line.strip().startswith("criterion =") or line.strip().startswith("OPTIMIZER_CONFIG =") or line.strip().startswith("optimizer =") or line.strip().startswith("scheduler =") or line.strip().startswith("print("):
                    if "def summarize_sparse_graph" not in line and "summarize_sparse_graph(" not in line:
                        continue
                cleaned_lines.append(line)
            source_code = "\n".join(cleaned_lines)
        setup_code_lines.append(source_code)

setup_code = "\n\n# --- CELL BOUNDARY ---\n\n".join(setup_code_lines)

# Write the concatenated setup code to a python file for inspection/execution
with open("e:/thesis/scratch/setup_compiled.py", "w", encoding="utf-8") as f_out:
    f_out.write(setup_code)

print("Setup code compiled.")
