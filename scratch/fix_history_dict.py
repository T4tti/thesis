import json
from pathlib import Path

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_8 = nb["cells"][8]
source_8 = "".join(cell_8["source"])

# Define target and replacement blocks for the history dict in Cell 8
target_dict = """    'train_Accuracy': [], 'val_Accuracy': [],
    'train_Macro_F1': [], 'val_Macro_F1': [],"""

replacement_dict = """    'train_Accuracy': [], 'val_Accuracy': [],
    'train_Balanced_Accuracy': [], 'val_Balanced_Accuracy': [],
    'train_Macro_F1': [], 'val_Macro_F1': [],"""

if target_dict in source_8:
    source_8 = source_8.replace(target_dict, replacement_dict)
    print("Found and replaced history dictionary keys in Cell 8.")
else:
    # Fallback replacement if formatting is slightly different
    source_8 = source_8.replace(
        "'train_Accuracy': [], 'val_Accuracy': [],",
        "'train_Accuracy': [], 'val_Accuracy': [],\\n    'train_Balanced_Accuracy': [], 'val_Balanced_Accuracy': [],"
    )
    print("Applied fallback replacement for history dictionary keys.")

cell_8["source"] = [line + "\n" for line in source_8.splitlines()]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated and saved successfully.")
