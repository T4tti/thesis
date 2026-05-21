import nbformat

nb_path = r'e:\thesis\notebooks\Transformer-BiLSTM.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for c in nb.cells:
    if c.cell_type == 'code':
        if '\\n' in c.source:
            # We must be careful because some strings might naturally contain \n.
            # But the specific one we added was 'scaler_amp.update()\\n        try:\\n...'
            if 'scaler_amp.update()\\n' in c.source:
                print("Fixing literal \\n in Bug 4 replacement")
                c.source = c.source.replace('\\n', '\n')

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Done fixing syntax error.")
