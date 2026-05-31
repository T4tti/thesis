"""Extract specific cell source code from the notebook, write to UTF-8 file."""
import json

nb_path = 'notebooks/Transformer-LSTM.ipynb'
out_path = 'scratch/extracted_cells_utf8.txt'
target_cells = [11, 16, 21, 23, 24, 25, 27, 30, 32, 34, 35]

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

with open(out_path, 'w', encoding='utf-8') as out:
    for idx in target_cells:
        c = cells[idx]
        src = c['source']
        out.write(f"\n{'='*80}\n")
        out.write(f"CELL {idx} ({c['cell_type']})\n")
        out.write(f"{'='*80}\n")
        out.write(src)
        out.write('\n\n')

print(f"Done. Written to {out_path}")
