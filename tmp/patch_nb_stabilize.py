"""
Patch script: Stabilize Transformer-LSTM Training
===================================================
Applies 8 targeted changes to the notebook to eliminate
training oscillation and improve convergence stability.

How to Run:
    python tmp/patch_nb_stabilize.py

Expected Output:
    - Creates notebooks/transformer-lstm.ipynb (overwritten in-place)
    - Prints summary of all applied patches
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "transformer-lstm.ipynb"
BACKUP_PATH = NB_PATH.with_suffix(".ipynb.bak_stabilize")

def load_notebook(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_notebook(nb: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

def patch_cell_source(nb: dict, old: str, new: str, description: str) -> bool:
    """Find and replace exact substring in any cell's source.
    Returns True if patch was applied successfully."""
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if old in src:
            patched = src.replace(old, new, 1)
            # Rebuild source as list of lines (notebook format)
            lines = patched.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]]
            if lines[-1]:  # last line without trailing newline
                cell["source"].append(lines[-1])
            # Clear outputs to avoid stale results
            if "outputs" in cell:
                cell["outputs"] = []
            if "execution_count" in cell:
                cell["execution_count"] = None
            print(f"  ✓ {description}")
            return True
    print(f"  ✗ FAILED: {description} — target string not found")
    return False


def main():
    if not NB_PATH.exists():
        print(f"ERROR: Notebook not found at {NB_PATH}")
        sys.exit(1)

    # Backup
    nb = load_notebook(NB_PATH)
    save_notebook(nb, BACKUP_PATH)
    print(f"Backup saved to {BACKUP_PATH}\n")

    applied = 0
    total = 0

    # ================================================================
    # Change 1: Reduce noise and feature dropout (Cell 8 - DataLoader)
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old="TRAIN_WINDOW_NOISE_STD = 0.02\nTRAIN_FEATURE_DROPOUT = 0.05",
        new="TRAIN_WINDOW_NOISE_STD = 0.01  # Reduced from 0.02 for training stability\nTRAIN_FEATURE_DROPOUT = 0.03  # Reduced from 0.05 to close train/val loss gap",
        description="Change 1a: Reduce input noise 0.02→0.01 and feature dropout 0.05→0.03",
    ):
        applied += 1

    # ================================================================
    # Change 2: Increase batch size 32 → 64 (Cell 8 - DataLoader)
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old="BATCH_SIZE = 32\n",
        new="BATCH_SIZE = 64  # Increased from 32 for more stable gradients\n",
        description="Change 2: Batch size 32→64",
    ):
        applied += 1

    # ================================================================
    # Change 3: Training config — reduce dropout, LR, transition penalty
    # (Cell 11 - Training Loop)
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "default_train_config = {\n"
            "    'hidden_size': 96,\n"
            "    'dropout': 0.20,\n"
            "    'fuzzy_mfs': 5,\n"
            "    'd_model': 96,\n"
            "    'n_heads': 4,\n"
            "    'n_layers': 2,\n"
            "    'sector_emb_dim': 16,\n"
            "    'max_relative_position': 32,\n"
            "    'lr': 2e-4,\n"
            "    'max_lr': 8.0e-4,\n"
            "    'weight_decay': 2e-3,\n"
            "    'focal_gamma': 1.5,\n"
            "    'ordinal_alpha': 0.02,\n"
            "    'label_smoothing': 0.01,\n"
            "    'aux_transition_weight': 0.12,\n"
            "    'aux_transition_weight_min': 0.03,\n"
            "    'aux_transition_start_frac': 0.00,\n"
            "}"
        ),
        new=(
            "default_train_config = {\n"
            "    'hidden_size': 128,       # Increased from 96 for more model capacity\n"
            "    'dropout': 0.12,          # Reduced from 0.20 to close train/val loss gap\n"
            "    'fuzzy_mfs': 5,\n"
            "    'd_model': 128,           # Increased from 96 for better representation\n"
            "    'n_heads': 4,\n"
            "    'n_layers': 3,            # Increased from 2 for deeper representation\n"
            "    'sector_emb_dim': 16,\n"
            "    'max_relative_position': 32,\n"
            "    'lr': 3e-4,              # Increased slightly from 2e-4 (base LR for cosine)\n"
            "    'max_lr': 5.0e-4,        # Reduced from 8e-4 to prevent LR spikes\n"
            "    'weight_decay': 1e-3,    # Reduced from 2e-3\n"
            "    'focal_gamma': 1.5,\n"
            "    'ordinal_alpha': 0.02,\n"
            "    'label_smoothing': 0.01,\n"
            "    'aux_transition_weight': 0.08,    # Reduced from 0.12\n"
            "    'aux_transition_weight_min': 0.02, # Reduced from 0.03\n"
            "    'aux_transition_start_frac': 0.15, # Delayed from 0.00 to let base loss stabilize\n"
            "}"
        ),
        description="Change 3: Training config — model capacity + reduced LR/regularization",
    ):
        applied += 1

    # ================================================================
    # Change 4: Disable Mixup (interferes with ordinal loss)
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old="MIXUP_ALPHA = 0.20\nMIXUP_PROB = 0.03",
        new="MIXUP_ALPHA = 0.0   # Disabled: Mixup conflicts with ordinal loss & causes instability\nMIXUP_PROB = 0.0    # Disabled for training stability",
        description="Change 4: Disable Mixup (conflicts with ordinal loss)",
    ):
        applied += 1

    # ================================================================
    # Change 5: Reduce context regularization
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "LAST_Y_CONTEXT_DROPOUT_MAX = 0.20\n"
            "LAST_Y_CONTEXT_DROPOUT_MIN = 0.01\n"
            "LAST_Y_CONTEXT_PERMUTE_MAX = 0.08\n"
            "LAST_Y_CONTEXT_PERMUTE_MIN = 0.00"
        ),
        new=(
            "LAST_Y_CONTEXT_DROPOUT_MAX = 0.10  # Reduced from 0.20 to prevent reversed loss gap\n"
            "LAST_Y_CONTEXT_DROPOUT_MIN = 0.01\n"
            "LAST_Y_CONTEXT_PERMUTE_MAX = 0.03  # Reduced from 0.08\n"
            "LAST_Y_CONTEXT_PERMUTE_MIN = 0.00"
        ),
        description="Change 5: Reduce context dropout 0.20→0.10, permute 0.08→0.03",
    ):
        applied += 1

    # ================================================================
    # Change 6: Reduce transition penalty + delay warmup
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "TRANSITION_MARGIN = 0.06\n"
            "TRANSITION_PENALTY_WEIGHT_MAX = 0.22\n"
            "TRANSITION_WARMUP_FRACTION = 0.15"
        ),
        new=(
            "TRANSITION_MARGIN = 0.06\n"
            "TRANSITION_PENALTY_WEIGHT_MAX = 0.12  # Reduced from 0.22 to stabilize gradients\n"
            "TRANSITION_WARMUP_FRACTION = 0.30     # Delayed from 0.15 to let representations form first"
        ),
        description="Change 6: Transition penalty max 0.22→0.12, warmup 0.15→0.30",
    ):
        applied += 1

    # ================================================================
    # Change 7: Replace OneCycleLR with CosineAnnealing + linear warmup
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)\n"
            "scheduler = torch.optim.lr_scheduler.OneCycleLR(\n"
            "    optimizer,\n"
            "    max_lr=MAX_LR,\n"
            "    steps_per_epoch=len(train_loader),\n"
            "    epochs=MAX_EPOCHS,\n"
            "    pct_start=0.2,\n"
            "    anneal_strategy='cos',\n"
            "    div_factor=max(MAX_LR / LR, 1.0),\n"
            "    final_div_factor=100.0,\n"
            " )"
        ),
        new=(
            "optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)\n"
            "\n"
            "# Scheduler: Linear warmup (5 epochs) + CosineAnnealing for stable convergence\n"
            "WARMUP_EPOCHS = 5\n"
            "warmup_scheduler = torch.optim.lr_scheduler.LinearLR(\n"
            "    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS\n"
            ")\n"
            "cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n"
            "    optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5\n"
            ")\n"
            "scheduler = torch.optim.lr_scheduler.SequentialLR(\n"
            "    optimizer,\n"
            "    schedulers=[warmup_scheduler, cosine_scheduler],\n"
            "    milestones=[WARMUP_EPOCHS],\n"
            ")"
        ),
        description="Change 7: OneCycleLR → LinearWarmup + CosineAnnealing (epoch-level)",
    ):
        applied += 1

    # ================================================================
    # Change 8a: Replace per-step scheduler.step() with epoch-level
    # (inside training loop batch iteration)
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "        scaler_amp.scale(loss).backward()\n"
            "        scaler_amp.unscale_(optimizer)\n"
            "        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)\n"
            "        scaler_amp.step(optimizer)\n"
            "        scaler_amp.update()\n"
            "        scheduler.step()"
        ),
        new=(
            "        scaler_amp.scale(loss).backward()\n"
            "        scaler_amp.unscale_(optimizer)\n"
            "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)  # Relaxed from 0.8\n"
            "        scaler_amp.step(optimizer)\n"
            "        scaler_amp.update()\n"
            "        # NOTE: scheduler.step() moved to epoch level (after val loop)"
        ),
        description="Change 8a: Grad clip 0.8→1.5, move scheduler.step() to epoch level",
    ):
        applied += 1

    # ================================================================
    # Change 8b: Add scheduler.step() after validation loop
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "    val_loss = np.mean(val_losses)\n"
            "    vl_yt = torch.cat(vl_yt)\n"
            "    vl_logits = torch.cat(vl_logits)\n"
            "    vl_acc, vl_f1, vl_f1w, vl_auc, vl_mae = compute_cls_metrics(vl_yt, vl_logits, n_classes)"
        ),
        new=(
            "    val_loss = np.mean(val_losses)\n"
            "    vl_yt = torch.cat(vl_yt)\n"
            "    vl_logits = torch.cat(vl_logits)\n"
            "    vl_acc, vl_f1, vl_f1w, vl_auc, vl_mae = compute_cls_metrics(vl_yt, vl_logits, n_classes)\n"
            "\n"
            "    # Epoch-level scheduler step (moved from per-batch for CosineAnnealing)\n"
            "    scheduler.step()"
        ),
        description="Change 8b: Add epoch-level scheduler.step() after validation",
    ):
        applied += 1

    # ================================================================
    # Change 9: Remove persistence blend during TRAINING
    # (keep for validation only)
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "            raw_logits, transition_logits = model(X_in, last_y_in, sector_batch, return_aux=True)\n"
            "            logits = blend_with_transition_persistence(\n"
            "                raw_logits,\n"
            "                last_y_in,\n"
            "                transition_logits,\n"
            "                n_cls=n_classes,\n"
            "                persistence_bias=epoch_bias,\n"
            "                stay_blend_max=STAY_BLEND_MAX,\n"
            "                stay_conf_threshold=STAY_CONF_THRESHOLD,\n"
            "            )\n"
            "            base_loss = compute_mixup_loss(criterion, logits, y_a, y_b, lam)"
        ),
        new=(
            "            raw_logits, transition_logits = model(X_in, last_y_in, sector_batch, return_aux=True)\n"
            "            # Use raw logits during training for clean gradients (no persistence blend)\n"
            "            logits = raw_logits\n"
            "            base_loss = compute_mixup_loss(criterion, logits, y_a, y_b, lam)"
        ),
        description="Change 9: Remove persistence blend during training (clean gradients)",
    ):
        applied += 1

    # ================================================================
    # Change 10: Use raw metric for early stopping (remove EMA)
    # ================================================================
    total += 1
    if patch_cell_source(
        nb,
        old=(
            "    current_metric_raw = vl_acc if EARLY_STOP_METRIC == 'val_acc' else (vl_f1 if EARLY_STOP_METRIC == 'val_f1' else vl_f1w)\n"
            "    if metric_ema_value is None:\n"
            "        metric_ema_value = float(current_metric_raw)\n"
            "    else:\n"
            "        metric_ema_value = (1.0 - METRIC_EMA_ALPHA) * metric_ema_value + METRIC_EMA_ALPHA * float(current_metric_raw)\n"
            "    current_metric = float(metric_ema_value)"
        ),
        new=(
            "    current_metric_raw = vl_acc if EARLY_STOP_METRIC == 'val_acc' else (vl_f1 if EARLY_STOP_METRIC == 'val_f1' else vl_f1w)\n"
            "    # Use raw metric directly (no EMA smoothing) so best checkpoint = true best\n"
            "    current_metric = float(current_metric_raw)"
        ),
        description="Change 10: Use raw metric for early stopping (no EMA smoothing)",
    ):
        applied += 1

    # ================================================================
    # Save
    # ================================================================
    save_notebook(nb, NB_PATH)
    print(f"\n{'='*60}")
    print(f"Patches applied: {applied}/{total}")
    print(f"Notebook saved to: {NB_PATH}")
    if applied < total:
        print("WARNING: Some patches failed — check output above.")
        sys.exit(1)
    else:
        print("All patches applied successfully!")


if __name__ == "__main__":
    main()
