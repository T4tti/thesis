import json
import re
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
notebook_files = [
    "Transformer-LSTM.ipynb",
    "tcn-baseline.ipynb",
    "lstm-baseline.ipynb",
    "patchtst-baseline.ipynb",
    "xgboost-baseline.ipynb",
    "lightgbm-baseline.ipynb",
    "kb7-fi-ttx.ipynb",
    "kb8-fi-pll.ipynb",
    "kb9-fi-ttlpxl.ipynb",
    "kb10-fr-ttx.ipynb",
    "kb11-fr-pll.ipynb",
    "kb12-fr-ttlpxl.ipynb",
]

def search_notebook_outputs(nb_path, out_file):
    out_file.write(f"\n========================================\nAnalyzing {nb_path.name}\n========================================\n")
    if not nb_path.exists():
        out_file.write("File does not exist.\n")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception as e:
            out_file.write(f"Error loading JSON: {e}\n")
            return

    # Look for outputs that contain test set metrics
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        
        # Check source code
        source = "".join(cell.get("source", []))
        
        # Check outputs
        outputs = cell.get("outputs", [])
        for out in outputs:
            text = ""
            if "text" in out:
                text = "".join(out["text"])
            elif "data" in out and "text/plain" in out["data"]:
                text = "".join(out["data"]["text/plain"])
            
            # Search for test evaluation patterns
            if ("test" in text.lower() or "evaluation" in text.lower() or "accuracy" in text.lower() or "accuracy:" in text.lower() or "val:" in text.lower()) and ("0." in text):
                # Print code snippet
                out_file.write(f"Cell #{idx} Source Code Snippet:\n")
                out_file.write(source[:300] + ("..." if len(source) > 300 else "") + "\n")
                out_file.write(f"Cell #{idx} Output:\n")
                out_file.write(text[:1500] + "\n")
                out_file.write("-" * 50 + "\n")

if __name__ == "__main__":
    out_path = Path("scratch/notebooks_metrics.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for nb_name in notebook_files:
            search_notebook_outputs(NOTEBOOKS_DIR / nb_name, out_file)
    print(f"Done! Written output to {out_path}")
