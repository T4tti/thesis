# How to Run: python scratch/inspect_metrics_format.py
# Expected Output: Prints metric capitalization and names populated in training history and used in plotting/evaluation in both notebooks.

import json
from pathlib import Path

def inspect_notebook(nb_name):
    nb_path = Path('notebooks') / nb_name
    print(f"=== {nb_name} ===")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Let's search for "history" or "train_loss" or similar in all code cells
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if 'history' in source and ('loss' in source.lower() or 'acc' in source.lower()):
                # Print lines containing history or train_ or val_
                lines = source.splitlines()
                matching_lines = []
                for line in lines:
                    if any(term in line.lower() for term in ['history[', 'history.append', 'history_df =', 'history_plot_df', '.columns', 'train_loss', 'val_loss', 'macro_f1', 'qwk', 'chgacc']):
                        matching_lines.append(line.strip())
                if matching_lines:
                    print(f"  [Cell {idx}] key matching lines:")
                    for ml in matching_lines[:15]:
                        print(f"    {ml}")
                    if len(matching_lines) > 15:
                        print(f"    ... [truncated {len(matching_lines)-15} lines]")
                    print()

if __name__ == '__main__':
    inspect_notebook('lstm-baseline.ipynb')
    inspect_notebook('transformer-lstm.ipynb')
