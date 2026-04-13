import json, sys
nb = json.load(open(r'notebooks/timegan_data_preparation_kaggle.ipynb', encoding='utf-8'))
out = []
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    out.append(f'=== CELL {i} [{cell["cell_type"]}] ===\n{src[:800]}\n')
with open('tmp/nb_content.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Done, cells:', len(nb['cells']))
