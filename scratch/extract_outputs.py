"""Extract output from specific notebook cells."""
import json

nb_path = 'notebooks/Transformer-LSTM.ipynb'
target_cells = [35]  # Evaluation metrics cell

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

for idx in target_cells:
    c = cells[idx]
    print(f"\n{'='*80}")
    print(f"CELL {idx} OUTPUTS:")
    print(f"{'='*80}")
    for out in c.get('outputs', []):
        if 'text' in out:
            text = out['text']
            if isinstance(text, list):
                text = ''.join(text)
            print(text[:3000])
        elif 'data' in out and 'text/plain' in out['data']:
            text = out['data']['text/plain']
            if isinstance(text, list):
                text = ''.join(text)
            print(text[:3000])
