import json

# Read key sections from all 6 baseline notebooks
notebooks = {
    'tBiLSTM': 'Transformer-BiLSTM.ipynb',
    'tcn': 'tcn-baseline.ipynb',
    'xgboost': 'xgboost-baseline.ipynb',
    'lstm': 'lstm-baseline.ipynb',
    'patchtst': 'patchtst-baseline.ipynb',
    'lgbm': 'lightgbm-baseline.ipynb',
}

for key, nb_name in notebooks.items():
    print(f"\n{'='*70}")
    print(f"NOTEBOOK: {key} ({nb_name})")
    print('='*70)
    with open(f'e:/thesis/notebooks/{nb_name}', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    print(f"Total cells: {len(cells)}")
    
    # Show last cells (training + evaluation section)
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        ctype = cell['cell_type']
        # Print markdown cells as section markers
        if ctype == 'markdown' and len(src) < 300:
            print(f"\n--- CELL {i} [MD] ---")
            print(src[:300])
        # Print code cells that have key sections
        elif ctype == 'code' and any(kw in src for kw in [
            'ARTIFACT_DIR', 'best_model', 'val_acc', 'test_acc', 'f1', 'confusion',
            'torch.save', 'model.save', 'joblib', 'classification_report',
            'BEST_MODEL_PATH', 'best_val', 'evaluate', 'OOF', 'oof_probs',
            'checkpoint'
        ]):
            print(f"\n--- CELL {i} [CODE - key] ---")
            print(src[:1500])
