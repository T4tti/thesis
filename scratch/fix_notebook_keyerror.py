import json
from pathlib import Path

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_8 = nb["cells"][8]
source_8 = "".join(cell_8["source"])

# Define target and replacement blocks for metric logging in Cell 8
target_logging = """    for metric_name in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Class0_Precision', 'Class0_Recall', 'Class0_F1', 'Class0_F2', 'ChgAcc', 'Ordinal_MAE', 'AUC', 'QWK']:
        history[f'train_{metric_name}'].append(float(tr[metric_name]) if not (isinstance(tr[metric_name], float) and tr[metric_name] != tr[metric_name]) else float('nan'))
        history[f'val_{metric_name}'].append(float(va[metric_name]) if not (isinstance(va[metric_name], float) and va[metric_name] != va[metric_name]) else float('nan'))"""

replacement_logging = """    for metric_name in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Class0_Precision', 'Class0_Recall', 'Class0_F1', 'Class0_F2', 'ChgAcc', 'Ordinal_MAE', 'AUC', 'QWK']:
        val_tr = tr.get(metric_name, tr.get('Accuracy', float('nan')))
        val_va = va.get(metric_name, va.get('Accuracy', float('nan')))
        history[f'train_{metric_name}'].append(float(val_tr) if not (isinstance(val_tr, float) and val_tr != val_tr) else float('nan'))
        history[f'val_{metric_name}'].append(float(val_va) if not (isinstance(val_va, float) and val_va != val_va) else float('nan'))"""

if target_logging in source_8:
    source_8 = source_8.replace(target_logging, replacement_logging)
    print("Found and replaced target logging in Cell 8.")
else:
    # Let's try matching with different spacing/newlines if any
    source_8 = source_8.replace("tr[metric_name]", "tr.get(metric_name, tr.get('Accuracy', float('nan')))")
    source_8 = source_8.replace("va[metric_name]", "va.get(metric_name, va.get('Accuracy', float('nan')))")
    print("Applied fallback replacements for tr[metric_name] and va[metric_name].")

cell_8["source"] = [line + "\n" for line in source_8.splitlines()]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated and saved successfully.")
