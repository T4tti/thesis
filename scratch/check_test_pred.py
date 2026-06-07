# How to Run: python scratch/check_test_pred.py
# Expected Output: Prints all occurrences of y_test_pred in notebooks/Sparse-Graph-baseline.ipynb cells.

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def main():
    nb_path = Path('notebooks/Sparse-Graph-baseline.ipynb')
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    
    for idx, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        if 'y_test_pred' in source:
            print(f"Cell {idx} ({cell.get('cell_type')}):")
            for line_no, line in enumerate(source.splitlines(), 1):
                if 'y_test_pred' in line:
                    print(f"  Line {line_no}: {line.strip()}")

if __name__ == '__main__':
    main()
