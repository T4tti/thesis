# How to Run: python scratch/inspect_transformer_lstm.py
# Expected Output: Prints selected cell sources and metadata related to batch loading, validation, training loops, and loss function definition from notebooks/transformer-lstm.ipynb.

import json
from pathlib import Path

def main():
    nb_path = Path('notebooks/transformer-lstm.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            # Let's check for keywords that are key to the training loop, loss, data unpacking, or evaluation
            keywords = ['DataLoader', 'class TLSTMFuzzyClassifier', 'train_one_epoch', 'evaluate', 'optimizer', 'FocalLoss', 'history_df']
            matched = [k for k in keywords if k in source]
            if matched:
                print(f"=== Cell {idx} (matched: {matched}) ===")
                # Print first 20 lines
                lines = source.splitlines()
                for line in lines[:40]:
                    print(f"  {line}")
                if len(lines) > 40:
                    print(f"  ... [truncated {len(lines)-40} lines]")
                print()

if __name__ == '__main__':
    main()
