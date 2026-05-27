import nbformat

nb_path = r'e:\thesis\notebooks\timegan_data_preparation_3groups.ipynb'
out_path = r'e:\thesis\.agent\scratch\split_logic.txt'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, c in enumerate(nb.cells):
            if c.cell_type == 'code':
                if 'split' in c.source.lower() or 'train' in c.source.lower() or 'test' in c.source.lower() or 'val' in c.source.lower():
                    fout.write(f"--- Cell {i} ---\n")
                    fout.write(c.source)
                    fout.write("\n\n")
    print("Done")
except Exception as e:
    print(f"Error: {e}")
