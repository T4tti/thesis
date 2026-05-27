import json
import os

def process_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                
                # For XGBoost and LightGBM baseline notebooks, we need to add the import
                # and MultipleLocator(20) right after ax.set_xlabel('Epoch')
                if "ax.set_xlabel('Epoch')" in line:
                    if "MultipleLocator" not in "".join(cell['source']):
                        new_source.append("    from matplotlib.ticker import MultipleLocator\n")
                        new_source.append("    ax.xaxis.set_major_locator(MultipleLocator(20))\n")
                        changed = True

            if new_source != cell['source']:
                cell['source'] = new_source
                changed = True
                
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Updated {filepath}")
    else:
        print(f"No changes for {filepath}")

notebooks = [
    r"e:\thesis\notebooks\xgboost-baseline.ipynb",
    r"e:\thesis\notebooks\lightgbm-baseline.ipynb"
]

for nb in notebooks:
    if os.path.exists(nb):
        process_notebook(nb)
    else:
        print(f"File not found: {nb}")
