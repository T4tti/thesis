"""
Patch script: Áp dụng Fix 1 + Fix 3 + Fix 4 vào patchtst-baseline.ipynb

Fix 1: ReduceLROnPlateau scheduler
Fix 3: Dropout 0.2 → 0.3, weight_decay 1e-4 → 1e-3
Fix 4: label_smoothing 0.0 → 0.1

How to Run: python scratch/patch_patchtst_notebook.py
Expected Output: Backup tạo tại patchtst-baseline.ipynb.bak, notebook được cập nhật.
"""

import json
import shutil
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "patchtst-baseline.ipynb"
BACKUP_PATH = NOTEBOOK_PATH.with_suffix(".ipynb.bak")

# ── helpers ──────────────────────────────────────────────────────────────────

def patch_source(lines: list[str]) -> tuple[list[str], list[str]]:
    """Apply all fixes to a cell's source lines. Returns (new_lines, changelog)."""
    changes: list[str] = []
    new: list[str] = []

    for line in lines:
        original = line

        # Fix 3a: Dropout 0.2 → 0.3
        if "drop=0.2," in line:
            line = line.replace("drop=0.2,", "drop=0.3,")
            changes.append("  [Fix 3] dropout: 0.2 -> 0.3")

        # Fix 3b: weight_decay 1e-4 → 1e-3
        if "weight_decay=1e-4" in line:
            line = line.replace("weight_decay=1e-4", "weight_decay=1e-3")
            changes.append("  [Fix 3] weight_decay: 1e-4 -> 1e-3")

        # Fix 4: label_smoothing 0.0 → 0.1
        if "'label_smoothing': 0.0," in line:
            line = line.replace("'label_smoothing': 0.0,", "'label_smoothing': 0.1,")
            changes.append("  [Fix 4] label_smoothing: 0.0 -> 0.1")

        new.append(line)

    # Fix 1: Inject scheduler definition after optimizer line
    # and scheduler.step() after va = run_epoch(...)
    result: list[str] = []
    i = 0
    while i < len(new):
        line = new[i]
        result.append(line)

        # Inject scheduler definition right after AdamW line
        if "optimizer = torch.optim.AdamW" in line and "scheduler" not in "".join(new[i:i+3]):
            scheduler_lines = [
                "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\\n",
                "    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True\\n",
                ")\\n",
                "print(f'LR Scheduler: ReduceLROnPlateau | factor=0.5 | patience=5 | min_lr=1e-6')\\n",
            ]
            result.extend(scheduler_lines)
            changes.append("  [Fix 1] Added ReduceLROnPlateau scheduler after optimizer")

        # Inject scheduler.step() right after va = run_epoch(...)
        if "va = run_epoch(model, val_loader, criterion, optimizer=None)" in line:
            result.append("    scheduler.step(va['Loss'])  # Fix 1: giảm LR khi val_Loss không cải thiện\\n")
            changes.append("  [Fix 1] Added scheduler.step(va['Loss']) in training loop")

        i += 1

    return result, changes


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    # Backup
    shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
    print(f"[OK] Backup created: {BACKUP_PATH}")

    with NOTEBOOK_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    total_changes: list[str] = []
    cells_patched = 0

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        src = cell["source"]
        src_text = "".join(src)

        # Only patch the cell with training loop (FocalOrdinalLoss + PatchTSTClassifier)
        if "PatchTSTClassifier(" not in src_text and "optimizer = torch.optim.AdamW" not in src_text:
            continue

        new_src, changes = patch_source(src)
        if changes:
            cell["source"] = new_src
            total_changes.extend(changes)
            cells_patched += 1

    if not total_changes:
        print("[WARN] No patches found -- check cell content.")
        return

    # Clear outputs of patched cells (will be regenerated on next run)
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src_text = "".join(cell["source"])
        if "PatchTSTClassifier(" in src_text or "optimizer = torch.optim.AdamW" in src_text:
            cell["outputs"] = []
            cell["execution_count"] = None

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n[OK] Patched {cells_patched} cell(s) with {len(total_changes)} changes:")
    for c in total_changes:
        print(c)
    print(f"\n[OK] Notebook saved: {NOTEBOOK_PATH}")
    print("\nSummary of changes:")
    print("  Fix 1: ReduceLROnPlateau(mode='min', factor=0.5, patience=5, min_lr=1e-6)")
    print("  Fix 3: dropout 0.2->0.3 | weight_decay 1e-4->1e-3")
    print("  Fix 4: label_smoothing 0.0->0.1")


if __name__ == "__main__":
    main()
