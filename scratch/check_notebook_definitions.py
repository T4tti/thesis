# How to Run: python scratch/check_notebook_definitions.py
# Expected Output: Prints defined variables, imports, and check results for cross-cell reference soundness.

import json
from pathlib import Path

def main():
    nb_path = Path('notebooks/transformer-lstm.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Track defined variables
    defined_vars = set()
    # Pre-populate with typical Jupyter notebook variables and modules
    defined_vars.update(['__name__', '__doc__', '__package__', 'display', 'get_ipython'])
    
    # We will also parse cells and find all variable definitions (simple assignment, import, def, class)
    # and then check cell by cell if any variable used is not yet defined.
    # Note: We must be careful not to flag false positives for common modules or external constants,
    # so we'll focus on notebook-specific keys like ARTIFACT_DIR, history_df, model, etc.
    
    critical_variables = ['ARTIFACT_DIR', 'history_df', 'history', 'best_epoch', 'device', 'PIN_MEMORY', 'MAX_EPOCHS']
    
    cell_definitions = []
    cell_usages = []
    
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            # Find definitions
            defs = []
            uses = []
            for var in critical_variables:
                # Simple heuristic: var is defined if we see "var =" or "var =" with spaces
                if f"{var} =" in source or f"{var}=" in source or f"def {var}" in source or f"class {var}" in source or f"import {var}" in source or f"from {var}" in source:
                    defs.append(var)
                # var is used if it appears in the source but is not defined
                elif var in source:
                    uses.append(var)
            cell_definitions.append((idx, defs))
            cell_usages.append((idx, uses))

    # Print definitions and usages of critical variables per cell
    print("=== CRITICAL VARIABLE FLOW ===")
    for idx in range(len(cell_definitions)):
        cell_idx, defs = cell_definitions[idx]
        _, uses = cell_usages[idx]
        if defs or uses:
            print(f"Cell {cell_idx:2d}:")
            if defs:
                print(f"  Defines: {defs}")
            if uses:
                print(f"  Uses:    {uses}")

    # Check if a variable is used before it is defined in any cell
    print("\n=== SEQUENCE CHECK ===")
    active_defs = set()
    errors_found = 0
    for idx in range(len(cell_definitions)):
        cell_idx, defs = cell_definitions[idx]
        _, uses = cell_usages[idx]
        
        for use in uses:
            if use not in active_defs:
                # Check if it might be defined as a builtin or we just missed it,
                # but if it is never defined in any cell before this, it's a hazard.
                # Let's see if it is defined in subsequent cells.
                future_defs = []
                for f_idx in range(idx + 1, len(cell_definitions)):
                    if use in cell_definitions[f_idx][1]:
                        future_defs.append(cell_definitions[f_idx][0])
                if future_defs:
                    print(f"[WARNING] Cell {cell_idx:2d} uses '{use}' which is defined later in cell(s) {future_defs}!")
                    errors_found += 1
                else:
                    # Never defined in the notebook? Let's check if it is defined in the cell itself via global/locals
                    # or if it's external.
                    pass
        
        active_defs.update(defs)
        
    if errors_found == 0:
        print("[SUCCESS] No out-of-order critical variable usages detected.")
    else:
        print(f"[WARNING] Found {errors_found} out-of-order variable usage(s).")

if __name__ == '__main__':
    main()
