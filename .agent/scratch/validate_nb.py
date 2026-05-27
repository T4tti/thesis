import json

for f in ['KB7_FI-TTX.ipynb','KB8_FI-PLL.ipynb','KB9_FI-TTLPXL.ipynb']:
    nb = json.load(open(f'e:/thesis/notebooks/{f}', encoding='utf-8'))
    print(f"\n{f}: {len(nb['cells'])} cells")
    syntax_errors = []
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        ctype = cell['cell_type']
        head = src.splitlines()[0][:60] if src else ''
        print(f"  Cell {i:02d} [{ctype}]: {head}")
        if ctype == 'code' and src.strip():
            try:
                compile(src, f'<cell_{i}>', 'exec')
            except SyntaxError as e:
                syntax_errors.append((i, str(e)))
                print(f"    *** SYNTAX ERROR: {e}")

    if not syntax_errors:
        print("  [OK] All code cells pass syntax check")
    else:
        print(f"  [FAIL] {len(syntax_errors)} syntax errors")
