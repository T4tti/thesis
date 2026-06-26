# -*- coding: utf-8 -*-
"""
How to Run:
.venv\\Scripts\\python.exe scratch/check_gat_auc.py

Expected Output:
In ra thông tin các từ khóa liên quan đến ROC/AUC trong GraphSAGE-baseline.ipynb
"""
from __future__ import annotations
import json
import re
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main() -> None:
    path = pathlib.Path("notebooks/GraphSAGE-baseline.ipynb")
    if not path.exists():
        print(f"File not found: {path}")
        return
        
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                text = output["text"]
                chunks.append("".join(text) if isinstance(text, list) else str(text))
    text = "\n".join(chunks)
    
    keywords = ["auc", "roc", "tpr", "fpr", "area"]
    for kw in keywords:
        matches = list(re.finditer(re.escape(kw), text, re.I))
        print(f"Từ khóa '{kw}': tìm thấy {len(matches)} lần xuất hiện.")
        for idx, m in enumerate(matches[:5]): # In tối đa 5 match đầu tiên để phân tích
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 100)
            clean_snippet = text[start:end].replace("\n", " ")
            print(f"   [{idx}] Vị trí {m.start()}: ... {clean_snippet} ...")

if __name__ == "__main__":
    main()
