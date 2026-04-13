import json
nb = json.load(open(r'notebooks/timegan_data_preparation_kaggle.ipynb', encoding='utf-8'))
# Print full content of all code cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        with open(f'tmp/cell_{i}.py', 'w', encoding='utf-8') as f:
            f.write(src)
        print(f'Cell {i}: {len(src)} chars')
