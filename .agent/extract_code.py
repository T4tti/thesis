import json

with open('e:/thesis/notebooks/Transformer-BiLSTM.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            source = "".join(source)
        code_cells.append(source)

with open('e:/thesis/.agent/notebook_code.py', 'w', encoding='utf-8') as f:
    f.write("\n\n# --- CELL ---\n\n".join(code_cells))
