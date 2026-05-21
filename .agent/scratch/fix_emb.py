import nbformat
import re

nb_path = r'e:\thesis\notebooks\Transformer-BiLSTM.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for c in nb.cells:
    if c.cell_type == 'code' and 'TLSTMFuzzyClassifier' in c.source:
        if 'self.last_y_embed = nn.Embedding(n_classes' in c.source:
            # We want to replace it only if it hasn't been modified yet
            if 'n_classes +' not in c.source.split('self.last_y_embed')[1].split('\n')[0]:
                print("Fixing self.last_y_embed")
                # Wait, the exact string is "self.last_y_embed = nn.Embedding(n_classes, hidden_size)"
                c.source = re.sub(r'self\.last_y_embed = nn\.Embedding\(n_classes,\s*(.*?)\)', r'self.last_y_embed = nn.Embedding(n_classes + 1, \1)  # BUG_FIX_V2 [P2]: +1 for unknown sentinel', c.source)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Done fixing embedding.")
