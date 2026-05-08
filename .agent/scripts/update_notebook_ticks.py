import json
import os
import glob

def process_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell['source']:
                if "ax.xaxis.set_major_locator(MultipleLocator(2))" in line:
                    line = line.replace("MultipleLocator(2)", "MultipleLocator(10)")
                    changed = True
                elif "ax.xaxis.set_major_locator(MaxNLocator(integer=True))" in line:
                    line = line.replace("MaxNLocator(integer=True)", "MultipleLocator(10)")
                    new_source.append("    from matplotlib.ticker import MultipleLocator\n")
                    changed = True
                
                new_source.append(line)
                
                # For lstm-baseline.ipynb
                if "ax.set_xlabel('Epoch')" in line and filepath.endswith('lstm-baseline.ipynb'):
                    new_source.append("    from matplotlib.ticker import MultipleLocator\n")
                    new_source.append("    ax.xaxis.set_major_locator(MultipleLocator(10))\n")
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
    r"e:\thesis\notebooks\patchtst-baseline.ipynb",
    r"e:\thesis\notebooks\tcn-baseline.ipynb",
    r"e:\thesis\notebooks\lstm-baseline.ipynb",
    r"e:\thesis\notebooks\Transformer-BiLSTM.ipynb"
]

for nb in notebooks:
    if os.path.exists(nb):
        process_notebook(nb)
    else:
        print(f"File not found: {nb}")
