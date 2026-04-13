"""Patch timegan notebook -- cells stored in separate files."""
import json

NB_PATH = 'notebooks/timegan_data_preparation_kaggle.ipynb'
with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def patch(nb_obj, idx, src_file):
    with open(src_file, encoding='utf-8') as fh:
        src = fh.read()
    nb_obj['cells'][idx]['source'] = src
    nb_obj['cells'][idx]['outputs'] = []
    nb_obj['cells'][idx]['execution_count'] = None
    print(f"  Cell {idx} patched from {src_file}")

# Add import re to cell 4
cell4 = ''.join(nb['cells'][4]['source'])
if 'import re' not in cell4:
    nb['cells'][4]['source'] = 'import re\n' + cell4
    nb['cells'][4]['outputs'] = []
    print("  Cell 4: added 'import re'")

patch(nb, 6,  'tmp/cell6_v2.py')
patch(nb, 18, 'tmp/cell18_v2.py')
patch(nb, 21, 'tmp/cell21_v2.py')
patch(nb, 22, 'tmp/cell22_v2.py')
patch(nb, 31, 'tmp/cell31_v2.py')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nNotebook saved.")
