"""
Strategy A+B: Update Transformer-BiLSTM notebook
  A: Change INPUT_SIZE from 1 to 4
  B: Replace SequentialLR with OneCycleLR (per-batch stepping)

How to Run:
  python e:/thesis/.agent/scratch/apply_strategy_ab.py

Expected Output:
  Confirmation messages for each change applied, plus verification.
"""
import json
from pathlib import Path

NB_PATH = Path("e:/thesis/notebooks/Transformer-BiLSTM.ipynb")

# --- Load notebook ---
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes_applied = []

# ============================================================
# Strategy A: Change INPUT_SIZE from 1 to 4  (Cell 15)
# ============================================================
cell15 = nb["cells"][15]
src15_lines = cell15["source"]  # list of strings (each line)
src15_str = "".join(src15_lines)

# Verify it's the right cell
assert "INPUT_SIZE_DEFAULT" in src15_str, "Cell 15 does not contain INPUT_SIZE_DEFAULT"

# Replace INPUT_SIZE_DEFAULT = 1 with INPUT_SIZE_DEFAULT = 4
old_input_size = "INPUT_SIZE_DEFAULT = 1"
new_input_size = "INPUT_SIZE_DEFAULT = 4  # Strategy A: increased from 1 to activate temporal modeling"

if old_input_size in src15_str:
    src15_str = src15_str.replace(old_input_size, new_input_size)
    cell15["source"] = [line + "\n" for line in src15_str.splitlines()]
    # Fix last line - no trailing newline if original didn't have one
    if cell15["source"] and cell15["source"][-1].endswith("\n\n"):
        cell15["source"][-1] = cell15["source"][-1].rstrip("\n") + "\n"
    changes_applied.append("Strategy A: INPUT_SIZE_DEFAULT changed from 1 to 4")
else:
    # Maybe already changed or different format
    if "INPUT_SIZE_DEFAULT = 4" in src15_str:
        print("Strategy A: INPUT_SIZE_DEFAULT already set to 4, skipping.")
    else:
        raise ValueError(f"Cannot find '{old_input_size}' in Cell 15. Current content snippet: {src15_str[:200]}")

# ============================================================
# Strategy B: Replace SequentialLR with OneCycleLR (Cell 26)
# ============================================================
cell26 = nb["cells"][26]
src26_str = "".join(cell26["source"])

# Verify it's the right cell
assert "SequentialLR" in src26_str or "OneCycleLR" in src26_str, "Cell 26 does not contain scheduler definition"

# --- B1: Replace scheduler creation (3 lines -> OneCycleLR block) ---
old_scheduler_block = """WARMUP_EPOCHS = 5
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])"""

new_scheduler_block = """# Strategy B: OneCycleLR to match hyperparameter search config
# Previously used SequentialLR which ignored MAX_LR entirely.
# OneCycleLR uses both LR (base) and MAX_LR (peak), matching search assumptions.
WARMUP_EPOCHS = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=max(1, len(train_loader)),
    epochs=MAX_EPOCHS,
    pct_start=float(WARMUP_EPOCHS) / max(MAX_EPOCHS, 1),
    anneal_strategy='cos',
    div_factor=max(MAX_LR / max(LR, 1e-9), 1.0),
    final_div_factor=100.0,
)
SCHEDULER_IS_PER_BATCH = True  # OneCycleLR steps per batch, not per epoch"""

if old_scheduler_block in src26_str:
    src26_str = src26_str.replace(old_scheduler_block, new_scheduler_block)
    changes_applied.append("Strategy B1: Replaced SequentialLR with OneCycleLR")
elif "OneCycleLR" in src26_str:
    print("Strategy B1: OneCycleLR already present, skipping replacement.")
else:
    raise ValueError("Cannot find old SequentialLR block in Cell 26.")

# --- B2: Move scheduler.step() from per-epoch to per-batch ---
# OLD: scheduler.step() is called once per epoch (line 193)
# NEW: Move it inside the training batch loop (after scaler_amp.update())

# Step 1: Add per-batch step after scaler_amp.update() inside training loop
old_batch_update = "        scaler_amp.update()"
new_batch_update = """        scaler_amp.update()
        if SCHEDULER_IS_PER_BATCH:
            try:
                scheduler.step()
            except ValueError:
                pass  # Guard against stepping beyond total_steps"""

if old_batch_update in src26_str:
    # Only replace the FIRST occurrence (inside training loop)
    src26_str = src26_str.replace(old_batch_update, new_batch_update, 1)
    changes_applied.append("Strategy B2: Added per-batch scheduler.step() after scaler_amp.update()")
else:
    print("Strategy B2: scaler_amp.update() pattern not found, skipping.")

# Step 2: Guard the per-epoch scheduler.step() to only run for non-per-batch schedulers
old_epoch_step = "    scheduler.step()"
new_epoch_step = "    if not SCHEDULER_IS_PER_BATCH:\n        scheduler.step()  # Only for per-epoch schedulers"

if old_epoch_step in src26_str:
    src26_str = src26_str.replace(old_epoch_step, new_epoch_step, 1)
    changes_applied.append("Strategy B3: Guarded per-epoch scheduler.step() with SCHEDULER_IS_PER_BATCH flag")
else:
    print("Strategy B3: Per-epoch scheduler.step() not found, skipping.")

# Write back
cell26["source"] = [line + "\n" for line in src26_str.splitlines()]
if cell26["source"] and cell26["source"][-1].endswith("\n\n"):
    cell26["source"][-1] = cell26["source"][-1].rstrip("\n") + "\n"

# ============================================================
# Save notebook
# ============================================================
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n" + "=" * 60)
print("CHANGES APPLIED:")
for c in changes_applied:
    print(f"  [OK] {c}")
print("=" * 60)

# ============================================================
# Verification
# ============================================================
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb_verify = json.load(f)

cell15_v = "".join(nb_verify["cells"][15]["source"])
cell26_v = "".join(nb_verify["cells"][26]["source"])

checks = {
    "INPUT_SIZE_DEFAULT = 4": "INPUT_SIZE_DEFAULT = 4" in cell15_v,
    "OneCycleLR present": "OneCycleLR" in cell26_v,
    "SequentialLR removed": "SequentialLR" not in cell26_v,
    "SCHEDULER_IS_PER_BATCH defined": "SCHEDULER_IS_PER_BATCH = True" in cell26_v,
    "Per-batch step in train loop": "if SCHEDULER_IS_PER_BATCH:" in cell26_v,
    "Per-epoch step guarded": "if not SCHEDULER_IS_PER_BATCH:" in cell26_v,
}

print("\nVERIFICATION:")
all_ok = True
for name, ok in checks.items():
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        all_ok = False

if all_ok:
    print("\nSUCCESS: All changes verified successfully!")
else:
    print("\nFAILED: Some checks failed!")
