"""
Patch the learning rate scheduler in notebooks/gat-baseline.ipynb from CosineAnnealingWarmRestarts to CosineAnnealingLR.

How to Run:
    python scratch/patch_gat_scheduler.py

Expected Output:
    Successfully updated scheduler to CosineAnnealingLR in notebooks/gat-baseline.ipynb
"""

import json
from pathlib import Path

notebook_path = Path(r'e:\thesis\notebooks\gat-baseline.ipynb')

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

modified = False

# Search and replace
old_scheduler_lines = (
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(\n"
    "    optimizer, T_0=20, T_mult=2, eta_min=1e-5,\n"
    ")"
)

new_scheduler_lines = (
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n"
    "    optimizer, T_max=100, eta_min=1e-5,\n"
    ")"
)

for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell['source'])
    if old_scheduler_lines in source:
        source = source.replace(old_scheduler_lines, new_scheduler_lines)
        cell['source'] = source.splitlines(True)
        if cell['source'] and not cell['source'][-1].endswith('\n'):
            cell['source'][-1] += '\n'
        cell['outputs'] = []
        cell['execution_count'] = None
        modified = True
        break

if modified:
    # Save the notebook back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print("Successfully updated scheduler to CosineAnnealingLR in notebooks/gat-baseline.ipynb")
else:
    print("WARNING: CosineAnnealingWarmRestarts pattern not found in notebooks/gat-baseline.ipynb. Already updated?")
