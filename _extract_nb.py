import json

with open('notebooks/transformer-lstm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('_nb_code.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            out.write(f'# === Cell {i} ===\n')
            out.write(src)
            out.write('\n\n')

print(f"Extracted {sum(1 for c in nb['cells'] if c['cell_type']=='code')} code cells")
