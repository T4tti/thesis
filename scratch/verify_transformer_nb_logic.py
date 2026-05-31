# How to Run: python scratch/verify_transformer_nb_logic.py
# Expected Output: Prints compilation verification result, critical variable sequences, and all file paths read or written by the notebook.

import json
import re
import sys
from pathlib import Path

def main():
    nb_path = Path('notebooks/transformer-lstm.ipynb')
    if not nb_path.exists():
        print(f"Error: {nb_path} does not exist!")
        sys.exit(1)

    print(f"Analyzing {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    print(f"Total cells: {len(cells)}")

    code_cells = []
    for idx, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            code_cells.append((idx, source))

    print(f"Code cells found: {len(code_cells)}")

    # 1. Compile each cell to verify syntax
    syntax_errors = 0
    for idx, code in code_cells:
        if not code.strip():
            continue
        try:
            compile(code, filename=f"cell_{idx}", mode="exec")
        except SyntaxError as e:
            print(f"[SYNTAX ERROR] Cell {idx} has a syntax error at line {e.lineno}: {e.msg}")
            syntax_errors += 1

    if syntax_errors > 0:
        print(f"[FAILURE] Found {syntax_errors} cells with syntax errors.")
    else:
        print("[SUCCESS] All cells compile successfully without syntax errors.")

    # 2. Extract and print all file paths referenced
    print("\n=== EXTRACTING REFERENCED FILE PATHS ===")
    all_code = "\n".join(code for _, code in code_cells)
    
    # Match any string literal ending with common formats
    file_pattern = re.compile(r'[\'"]([^\'"]+\.(?:csv|xlsx|npy|pth|png|json))[\'"]')
    matches = file_pattern.findall(all_code)
    
    # Also find pathlib Path constructions or saves
    path_pattern = re.compile(r'(?:Path|ARTIFACT_DIR)\s*/\s*[\'"]([^\'"]+)[\'"]')
    path_matches = path_pattern.findall(all_code)
    
    unique_paths = sorted(list(set(matches + path_matches)))
    print("Found files/paths referenced in code:")
    for path in unique_paths:
        print(f"  - {path}")

    # 3. Check for specific imports/classes
    print("\n=== SPECIAL CHECKS ===")
    has_seaborn = 'sns.set_theme' in all_code
    print(f"Set theme using seaborn present: {has_seaborn}")

if __name__ == '__main__':
    main()
