import os
import json
import re
import sys

# Ensure UTF-8 stdout encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

def search_notebook_for_metrics(filepath):
    """
    Parses a Jupyter Notebook file and searches for model metrics on the test set.
    """
    filename = os.path.basename(filepath)
    print(f"=== Reading {filename} ===")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    metrics_found = []
    
    # Iterate through notebook cells
    for idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        if cell_type != 'code':
            continue
            
        source = "".join(cell.get('source', []))
        outputs = cell.get('outputs', [])
        
        for out in outputs:
            text = ""
            if out.get('output_type') == 'stream':
                text = "".join(out.get('text', []))
            elif out.get('output_type') == 'execute_result':
                data = out.get('data', {})
                if 'text/plain' in data:
                    text = "".join(data['text/plain'])
            
            if not text:
                continue
                
            lines = text.split('\n')
            for line in lines:
                # Search for classification report header, averages, accuracy, auc-roc, f1-score, etc.
                line_lower = line.lower()
                if any(kw in line_lower for kw in [
                    "accuracy:", "precision:", "recall:", "auc-roc", "roc auc", "roc_auc", "auc:",
                    "f1-score", "macro avg", "weighted avg", "accuracy  "
                ]):
                    # Clean Vietnamese character issues if any and print safely
                    cleaned_line = line.strip()
                    metrics_found.append((idx, source[:80], cleaned_line))
                    
    # Print matches
    if metrics_found:
        print(f"Found {len(metrics_found)} potential metric lines in outputs:")
        for idx, src, val in metrics_found:
            # Clean up the output string to show clearly
            # e.g., if it's classification report headers, it might be spaced
            print(f"  Cell {idx}: Output: {val}")
    else:
        print("No metrics found in this notebook.")
    print("-" * 50)

if __name__ == '__main__':
    notebooks_dir = r"e:\thesis\notebooks"
    notebooks = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]
    # Let's filter out backup files (.bak, copy, etc.) to focus on main notebooks, but keep copy if needed.
    # Actually, let's process all main notebooks.
    filtered_notebooks = [
        f for f in notebooks 
        if not f.endswith('.bak') and 'copy' not in f.lower()
    ]
    for nb in sorted(filtered_notebooks):
        search_notebook_for_metrics(os.path.join(notebooks_dir, nb))
