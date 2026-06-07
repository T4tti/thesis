# How to Run: python scratch/inspect_cell25_details.py
# Expected Output: Prints key lines of Cell 25 in notebooks/transformer-lstm.ipynb to plan the replacement.

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def main():
    nb_path = Path('notebooks/transformer-lstm.ipynb')
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    cell = nb['cells'][25]
    lines = cell.get('source', [])
    
    print(f"Total lines in Cell 25: {len(lines)}")
    
    print("\n--- LINES 1110 - 1140 ---")
    for idx in range(1110, min(1140, len(lines))):
        print(f"{idx+1:04d}: {lines[idx]}", end='')
        
    print("\n--- LAST 20 LINES ---")
    start_last = max(0, len(lines) - 20)
    for idx in range(start_last, len(lines)):
        print(f"{idx+1:04d}: {lines[idx]}", end='')

if __name__ == '__main__':
    main()
