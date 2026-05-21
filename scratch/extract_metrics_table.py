import os
import json
import sys

def extract_detailed_metrics(filepath, out_f):
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        out_f.write(f"Error reading {filepath}: {e}\n")
        return

    out_f.write(f"\n=========================================\n")
    out_f.write(f"DETAILED METRICS FOR: {filename}\n")
    out_f.write(f"=========================================\n")
    
    for idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        if cell_type != 'code':
            continue
            
        outputs = cell.get('outputs', [])
        source = "".join(cell.get('source', []))
        
        has_metrics = False
        text_outputs = []
        for out in outputs:
            text = ""
            if out.get('output_type') == 'stream':
                text = "".join(out.get('text', []))
            elif out.get('output_type') == 'execute_result':
                data = out.get('data', {})
                if 'text/plain' in data:
                    text = "".join(data['text/plain'])
            
            if text:
                text_outputs.append(text)
                if any(kw in text.lower() for kw in ["accuracy", "macro avg", "weighted avg", "auc", "qwk"]):
                    has_metrics = True
                    
        if has_metrics:
            out_f.write(f"\n[Cell {idx}] Source code starts with: {source.strip().split(chr(10))[0]}\n")
            out_f.write("--- Output ---\n")
            for t in text_outputs:
                out_f.write(t.strip() + "\n")
            out_f.write("-" * 30 + "\n")

if __name__ == '__main__':
    notebooks_dir = r"e:\thesis\notebooks"
    target_notebooks = [
        "Transformer-BiLSTM.ipynb",
        "kb7-fi-ttx.ipynb",
        "kb8-fi-pll.ipynb",
        "kb9-fi-ttlpxl.ipynb",
        "kb10-fr-ttx.ipynb",
        "kb11-fr-pll.ipynb",
        "kb12-fr-ttlpxl.ipynb",
        "lightgbm-baseline.ipynb",
        "lstm-baseline.ipynb",
        "patchtst-baseline.ipynb",
        "tcn-baseline.ipynb",
        "xgboost-baseline.ipynb"
    ]
    output_path = r"e:\thesis\scratch\output_metrics.txt"
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for nb in target_notebooks:
            path = os.path.join(notebooks_dir, nb)
            if os.path.exists(path):
                extract_detailed_metrics(path, out_f)
            else:
                out_f.write(f"File not found: {path}\n")
    print(f"Metrics written to {output_path}")
