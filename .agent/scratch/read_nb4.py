import json

# Get detailed structure of first cells for data loading
with open('e:/thesis/notebooks/Transformer-BiLSTM.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells'][:6]):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    print(f"\n--- CELL {i} [{ctype}] ---")
    print(src[:3000])
