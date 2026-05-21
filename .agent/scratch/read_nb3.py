import json

# Read the last portion of each baseline notebook to understand final sections
for nb_name in ['Transformer-BiLSTM.ipynb', 'tcn-baseline.ipynb', 'xgboost-baseline.ipynb']:
    print(f"\n{'='*70}")
    print(f"NOTEBOOK: {nb_name}")
    print('='*70)
    with open(f'e:/thesis/notebooks/{nb_name}', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    print(f"Total cells: {len(cells)}")
    
    # Show last 8 cells (evaluation section)
    for i, cell in enumerate(cells[-8:]):
        src = ''.join(cell['source'])
        ctype = cell['cell_type']
        actual_idx = len(cells) - 8 + i
        print(f"\n--- CELL {actual_idx} [{ctype}] ---")
        print(src[:2000])
