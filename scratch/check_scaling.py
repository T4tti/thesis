# How to Run: python scratch/check_scaling.py
# Expected Output: Prints lines of code related to scaling/normalization from both notebooks

import json
from pathlib import Path
from typing import List, Dict, Any

def search_notebook_cells(notebook_path: Path, keywords: List[str]) -> None:
    print(f"\n=== Searching {notebook_path.name} ===")
    if not notebook_path.exists():
        print("File does not exist.")
        return
        
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
            
        cell_idx = 0
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell_idx += 1
                source_lines = cell.get("source", [])
                # Combine source lines into a single string for matching
                cell_text = "".join(source_lines)
                
                # Check if any keyword matches
                matches = [kw for kw in keywords if kw.lower() in cell_text.lower()]
                if matches:
                    print(f"\n[Code Cell #{cell_idx}] (Matched keywords: {matches})")
                    for line_no, line in enumerate(source_lines, 1):
                        if any(kw.lower() in line.lower() for kw in keywords):
                            print(f"  Line {line_no}: {line.strip()}")
                            
    except Exception as e:
        print(f"Error parsing notebook: {e}")

if __name__ == "__main__":
    notebooks_dir = Path("notebooks")
    keywords = ["scaler", "scale", "normalize", "standard", "minmax", "robust", "transform"]
    
    search_notebook_cells(notebooks_dir / "timegan-3groups.ipynb", keywords)
    search_notebook_cells(notebooks_dir / "timegan_data_preparation_3groups.ipynb", keywords)
