"""
Fix: remove duplicate scheduler.step line from patchtst-baseline.ipynb

How to Run: python scratch/fix_duplicate_scheduler.py
Expected Output: Duplicate scheduler.step removed from notebook.
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "patchtst-baseline.ipynb"

nb = json.load(open(NOTEBOOK_PATH, encoding="utf-8"))

TARGET_LINE = "    scheduler.step(va['Loss'])  # Fix 1: gi\u1ea3m LR khi val_Loss kh\u00f4ng c\u1ea3i thi\u1ec7n\\n"

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell["source"]
    count = sum(1 for l in src if "scheduler.step(va['Loss'])" in l)
    if count > 1:
        # Keep only the first occurrence, remove subsequent duplicates
        seen = False
        new_src = []
        for line in src:
            if "scheduler.step(va['Loss'])" in line:
                if not seen:
                    new_src.append(line)
                    seen = True
                # else: skip duplicate
            else:
                new_src.append(line)
        cell["source"] = new_src
        print(f"[OK] Removed {count - 1} duplicate scheduler.step line(s)")

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Notebook saved.")
