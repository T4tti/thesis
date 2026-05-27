import nbformat

nb_path = r'e:\thesis\notebooks\Transformer-BiLSTM.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

found = False
for c in nb.cells:
    if c.cell_type == 'code' and 'TLSTMFuzzyClassifier' in c.source:
        print("Found TLSTMFuzzyClassifier")
        if 'prev_rating_emb' in c.source:
            print("Found prev_rating_emb")
            found = True
        elif 'last_y' in c.source:
            print("Found last_y")
            found = True
        elif 'Embedding(' in c.source:
            print("Found Embedding(")
            found = True
            
        # Try to modify whatever embedding there is for rating
        src = c.source
        
        # We need to find `nn.Embedding(n_classes, ` or similar and replace it.
        # Let's see what embeddings are defined.
        lines = [line for line in src.split('\n') if 'Embedding' in line]
        for line in lines:
            print(f"Embedding line: {line.strip()}")

if not found:
    print("Not found")
