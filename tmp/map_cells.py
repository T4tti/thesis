import json, copy

# Load notebook
with open('notebooks/timegan_data_preparation_kaggle.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Print cell map for patching
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    print(f"Cell {i} [{cell['cell_type']}]: {src[:80].strip()!r}")
