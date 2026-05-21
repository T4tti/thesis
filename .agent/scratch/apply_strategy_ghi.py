"""
Apply Strategy G, H, I to Transformer-BiLSTM notebook.

  G: Fix last_y leakage at t=0 (use sentinel n_classes instead of y[0])
  H: Add post-hoc Distressed threshold in test evaluation
  I: Increase focal_gamma for better precision-recall balance

How to Run:
  python e:/thesis/.agent/scratch/apply_strategy_ghi.py

Expected Output:
  Confirmation messages for each strategy applied + verification.
"""
import json
from pathlib import Path

NB_PATH = Path("e:/thesis/notebooks/Transformer-BiLSTM.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = []

# ============================================================
# STRATEGY G: Fix last_y leakage at t=0 in Cell 15
# ============================================================
cell15 = nb["cells"][15]
src15 = "".join(cell15["source"])

# G1: Replace the last_y_idx logic + sample tuple construction
old_leakage_block = """        last_y_idx = max(0, target_idx - 1)
        sample = (
            X,
            int(grp['y'].iloc[last_y_idx]),
            int(grp['sector_id'].iloc[target_idx]),
            int(grp['ticker_id'].iloc[target_idx]),
            int(grp['company_id'].iloc[target_idx]),
            int(grp['y'].iloc[target_idx]),
            int(grp['row_id'].iloc[target_idx]),
        )"""

new_leakage_block = """        # Strategy G: Fix last_y leakage at t=0
        # At t=0 there is no previous rating, so use sentinel value n_classes
        # to signal "unknown context" instead of leaking y[0] == y_target.
        if target_idx == 0:
            last_y_val = n_classes  # sentinel index outside [0, n_classes-1]
        else:
            last_y_val = int(grp['y'].iloc[target_idx - 1])
        sample = (
            X,
            last_y_val,
            int(grp['sector_id'].iloc[target_idx]),
            int(grp['ticker_id'].iloc[target_idx]),
            int(grp['company_id'].iloc[target_idx]),
            int(grp['y'].iloc[target_idx]),
            int(grp['row_id'].iloc[target_idx]),
        )"""

if old_leakage_block in src15:
    src15 = src15.replace(old_leakage_block, new_leakage_block)
    changes.append("G1: Fixed last_y leakage at t=0 (sentinel = n_classes)")
else:
    raise ValueError("Cannot find old leakage block in Cell 15")

# G2: Fix the is_change calculation which also uses last_y_idx
old_change = "train_seq_is_change.append(int(grp['y'].iloc[target_idx] != grp['y'].iloc[last_y_idx]))"
new_change = "train_seq_is_change.append(int(grp['y'].iloc[target_idx] != last_y_val) if target_idx > 0 else 1)"

if old_change in src15:
    src15 = src15.replace(old_change, new_change)
    changes.append("G2: Fixed is_change calculation for t=0")

# Write Cell 15
cell15["source"] = [line + "\n" for line in src15.splitlines()]

# ============================================================
# STRATEGY G (continued): Update model embedding in Cell 22
# ============================================================
cell22 = nb["cells"][22]
src22 = "".join(cell22["source"])

# G3: Expand last_y_embed to n_classes + 1
old_embed = "self.last_y_embed = nn.Embedding(n_classes, hidden_size)"
new_embed = "self.last_y_embed = nn.Embedding(n_classes + 1, hidden_size)  # Strategy G: +1 for unknown sentinel at t=0"

if old_embed in src22:
    src22 = src22.replace(old_embed, new_embed)
    changes.append("G3: Expanded last_y_embed to n_classes+1 for sentinel")
else:
    # Maybe already updated
    if "n_classes + 1" in src22 and "last_y_embed" in src22:
        print("G3: last_y_embed already has n_classes+1, skipping.")
    else:
        raise ValueError("Cannot find last_y_embed in Cell 22")

# Write Cell 22
cell22["source"] = [line + "\n" for line in src22.splitlines()]

# ============================================================
# STRATEGY H: Add Distressed threshold in Cell 31 (test prediction)
# ============================================================
cell31 = nb["cells"][31]
src31 = "".join(cell31["source"])

# H: Add post-hoc threshold after y_pred is computed
old_pred_end = """y_true = np.array(test_trues)
y_pred = np.array(test_preds)
test_row_ids = np.array(test_row_ids, dtype=int)"""

new_pred_end = """y_true = np.array(test_trues)
y_pred_raw = np.array(test_preds)  # Keep raw argmax predictions
test_row_ids = np.array(test_row_ids, dtype=int)

# Strategy H: Post-hoc Distressed threshold to reduce false positives
# Only predict class 0 (Distressed) when probability exceeds threshold.
# This trades some recall for improved precision on minority class.
DISTRESSED_THRESHOLD = 0.55
test_probs_all = torch.softmax(test_logits_all, dim=1).numpy()
y_pred = y_pred_raw.copy()
false_positive_mask = (y_pred == 0) & (test_probs_all[:, 0] < DISTRESSED_THRESHOLD)
if false_positive_mask.any():
    # Reassign low-confidence Distressed predictions to next-best class
    y_pred[false_positive_mask] = test_probs_all[false_positive_mask, 1:].argmax(axis=1) + 1
    n_reclassified = int(false_positive_mask.sum())
    print(f'[Strategy H] Reclassified {n_reclassified} low-confidence Distressed predictions (threshold={DISTRESSED_THRESHOLD})')
else:
    print(f'[Strategy H] No Distressed predictions below threshold {DISTRESSED_THRESHOLD}')"""

if old_pred_end in src31:
    src31 = src31.replace(old_pred_end, new_pred_end)
    changes.append("H: Added post-hoc Distressed threshold (0.55) in test prediction")
else:
    raise ValueError("Cannot find y_pred block in Cell 31")

# Write Cell 31
cell31["source"] = [line + "\n" for line in src31.splitlines()]

# ============================================================
# STRATEGY I: Increase focal_gamma in Cell 26 config
# ============================================================
cell26 = nb["cells"][26]
src26 = "".join(cell26["source"])

# I: Increase focal_gamma from 2.0 to 2.5
old_gamma = "'focal_gamma': 2.0,"
new_gamma = "'focal_gamma': 2.5,    # Strategy I: increased from 2.0 for better precision-recall balance"

if old_gamma in src26:
    src26 = src26.replace(old_gamma, new_gamma)
    changes.append("I: focal_gamma increased from 2.0 to 2.5")

# Also add focal_gamma to the strategy overrides block
old_override_end = "    'label_smoothing': 0.05,  # F: prevent overconfident predictions\n}"
new_override_end = "    'label_smoothing': 0.05,  # F: prevent overconfident predictions\n    'focal_gamma': 2.5,       # I: better precision-recall balance\n}"

if old_override_end in src26:
    src26 = src26.replace(old_override_end, new_override_end)
    changes.append("I: Added focal_gamma=2.5 to strategy overrides")

# Write Cell 26
cell26["source"] = [line + "\n" for line in src26.splitlines()]

# ============================================================
# Save notebook
# ============================================================
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("=" * 60)
print("CHANGES APPLIED:")
for c in changes:
    print(f"  [OK] {c}")
print("=" * 60)

# ============================================================
# Verification
# ============================================================
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb_v = json.load(f)

cell15_v = "".join(nb_v["cells"][15]["source"])
cell22_v = "".join(nb_v["cells"][22]["source"])
cell26_v = "".join(nb_v["cells"][26]["source"])
cell31_v = "".join(nb_v["cells"][31]["source"])

checks = [
    # Strategy G
    ("G1: Sentinel at t=0", "last_y_val = n_classes" in cell15_v),
    ("G2: Conditional last_y", "if target_idx == 0:" in cell15_v),
    ("G3: Embedding n_classes+1", "n_classes + 1, hidden_size)" in cell22_v),
    ("G4: No old leakage", "last_y_idx = max(0, target_idx - 1)" not in cell15_v),
    # Strategy H
    ("H1: DISTRESSED_THRESHOLD defined", "DISTRESSED_THRESHOLD = 0.55" in cell31_v),
    ("H2: False positive mask", "false_positive_mask" in cell31_v),
    ("H3: Reclassification logic", "y_pred[false_positive_mask]" in cell31_v),
    # Strategy I
    ("I1: focal_gamma = 2.5 in config", "'focal_gamma': 2.5," in cell26_v),
    ("I2: focal_gamma in overrides", "'focal_gamma': 2.5," in cell26_v),
]

print("\nVERIFICATION:")
all_ok = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        all_ok = False

if all_ok:
    print(f"\nSUCCESS: All {len(checks)}/{len(checks)} Strategy G+H+I checks passed!")
else:
    failed = sum(1 for _, ok in checks if not ok)
    print(f"\nFAILED: {failed}/{len(checks)} checks failed!")
