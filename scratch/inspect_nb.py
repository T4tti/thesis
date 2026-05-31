"""Inspect notebook cells — print index, type, and first 200 chars of source."""
import json
import sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/Transformer-LSTM.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}\n")

for i, c in enumerate(cells):
    src = c['source'][:200].replace('\n', '\\n')
    # Sanitize for console output
    src = src.encode('ascii', errors='replace').decode('ascii')
    print(f"[{i:3d}] {c['cell_type']:10s} | {src}")
    print()
