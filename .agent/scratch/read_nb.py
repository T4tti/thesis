import json

for nb_name in ['Transformer-BiLSTM.ipynb', 'tcn-baseline.ipynb', 'xgboost-baseline.ipynb']:
    print(f"\n{'='*60}")
    print(f"NOTEBOOK: {nb_name}")
    print('='*60)
    with open(f'e:/thesis/notebooks/{nb_name}', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells'][:25]):
        src = ''.join(cell['source'])
        ctype = cell['cell_type']
        print(f"\n--- CELL {i} [{ctype}] ---")
        print(src[:1200])
