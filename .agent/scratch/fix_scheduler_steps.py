"""
Fix the scheduler step logic in Cell 26 of Transformer-BiLSTM.ipynb.
The previous partial apply created a nested if-block issue.
This script surgically fixes it.

How to Run:
  python e:/thesis/.agent/scratch/fix_scheduler_steps.py

Expected Output:
  Confirmation that per-batch and per-epoch scheduler steps are correct.
"""
import json
from pathlib import Path

NB_PATH = Path("e:/thesis/notebooks/Transformer-BiLSTM.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell26 = nb["cells"][26]
src = "".join(cell26["source"])

# --- Fix 1: Clean up the malformed per-batch block ---
# Current broken state (lines 161-167 area):
#   scaler_amp.update()
#   if SCHEDULER_IS_PER_BATCH:
#       try:
#           if not SCHEDULER_IS_PER_BATCH:    <-- WRONG nested guard
#               scheduler.step()  ...
#       except ValueError:
#           pass
#
# Desired state:
#   scaler_amp.update()
#   if SCHEDULER_IS_PER_BATCH:
#       try:
#           scheduler.step()
#       except ValueError:
#           pass

lines = src.split("\n")

# Find and rebuild the per-batch block
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Detect the broken per-batch block
    if line.strip() == "scaler_amp.update()" and i + 1 < len(lines) and "SCHEDULER_IS_PER_BATCH" in lines[i + 1]:
        # Output the scaler_amp.update() line
        new_lines.append(line)
        
        # Skip all lines of the broken block until we find the `pass` line
        j = i + 1
        while j < len(lines):
            if lines[j].strip() == "pass  # Guard against stepping beyond total_steps":
                break
            elif lines[j].strip() == "pass":
                break
            j += 1
        
        # Insert the clean per-batch block
        indent = "        "  # 8 spaces (inside for loop, inside if)
        new_lines.append(f"{indent}if SCHEDULER_IS_PER_BATCH:")
        new_lines.append(f"{indent}    try:")
        new_lines.append(f"{indent}        scheduler.step()")
        new_lines.append(f"{indent}    except ValueError:")
        new_lines.append(f"{indent}        pass  # Guard against stepping beyond total_steps")
        
        i = j + 1  # Skip past the broken block
        continue
    
    # Detect the per-epoch scheduler.step() and guard it
    if line.strip() == "scheduler.step()" and i > 0:
        # Check if it's the epoch-level one (indented with 4 spaces, not 8)
        leading = len(line) - len(line.lstrip())
        if leading <= 4:
            # This is the per-epoch scheduler.step()
            indent = " " * leading
            new_lines.append(f"{indent}if not SCHEDULER_IS_PER_BATCH:")
            new_lines.append(f"{indent}    scheduler.step()  # Only for per-epoch schedulers")
            i += 1
            continue
    
    new_lines.append(line)
    i += 1

src_fixed = "\n".join(new_lines)

# Write back
cell26["source"] = [line + "\n" for line in src_fixed.splitlines()]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# --- Verification ---
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb_v = json.load(f)

cell26_v = "".join(nb_v["cells"][26]["source"])

# Check that the scheduler logic is correct
checks = {
    "OneCycleLR present": "OneCycleLR" in cell26_v,
    "SCHEDULER_IS_PER_BATCH defined": "SCHEDULER_IS_PER_BATCH = True" in cell26_v,
    "Per-batch step: scheduler.step() inside try": "        if SCHEDULER_IS_PER_BATCH:\n            try:\n                scheduler.step()" in cell26_v,
    "Per-epoch guarded": "    if not SCHEDULER_IS_PER_BATCH:\n        scheduler.step()" in cell26_v,
    "No SequentialLR": "SequentialLR" not in cell26_v,
    "No nested SCHEDULER_IS_PER_BATCH": cell26_v.count("if SCHEDULER_IS_PER_BATCH:") == 1 and cell26_v.count("if not SCHEDULER_IS_PER_BATCH:") == 1,
}

print("\nVERIFICATION:")
all_ok = True
for name, ok in checks.items():
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        all_ok = False

if all_ok:
    print("\nSUCCESS: Scheduler fix verified!")
else:
    print("\nWARNING: Some checks failed. Dumping relevant lines...")
    lines = cell26_v.split("\n")
    for i, line in enumerate(lines):
        if "scheduler" in line.lower() or "SCHEDULER_IS" in line:
            print(f"  {i}: {line}")
