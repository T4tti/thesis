"""
Training Utilities V2 — Aggressive anti-persistence training strategy.

ROOT CAUSE FIX:
  The model learns the identity mapping y = last_y within 3 epochs because:
    1. last_y is directly accessible as input
    2. The most common output IS last_y (persistence ~49%)
    3. Transition penalty/aux loss start too late (epoch 12+)
    4. Context perturbation is too weak (4%)
    5. No class weights → majority+persistence dominates

SOLUTION — "Feature-First Curriculum":
  Phase 1 (0-30% epochs): HIGH last_y masking (50-70%) → forces model to
      learn from financial features. Transition penalty active from epoch 0.
  Phase 2 (30-70%): gradually reveal last_y, anneal masking to ~10%.
  Phase 3 (70-100%): fine-tune with low masking + SWA weight averaging.

How to Run:
  Copy functions into training loop cell of notebook.
  See TRAINING_LOOP_TEMPLATE at the bottom.

Expected Output:
  - ChgAcc should be > 0.15 (from ~0.02)
  - val F1w should beat persistence baseline 0.489
  - Uplift > 0 enables checkpoint saving
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  1. Aggressive Context Perturbation Schedule
# ═══════════════════════════════════════════════════════════════════════

class ContextScheduler:
    """Curriculum schedule for last_y context perturbation.

    The key insight: model must learn financial feature signals BEFORE
    it gets reliable access to last_y. Otherwise it instantly learns
    the trivial mapping y = last_y and never explores financial signals.

    Schedule:
      Phase 1 (0 → warmup_frac):    HIGH masking, force feature learning
      Phase 2 (warmup_frac → cool):  Linear ramp-down to moderate masking
      Phase 3 (cool → 1.0):          Low masking for fine-tuning
    """

    def __init__(
        self,
        # Phase 1: Force feature learning (high masking)
        dropout_max: float = 0.60,
        permute_max: float = 0.30,
        # Phase 3: Fine-tuning (low masking)
        dropout_min: float = 0.05,
        permute_min: float = 0.02,
        # Phase boundaries
        warmup_frac: float = 0.30,   # Phase 1 → Phase 2 transition
        cooldown_frac: float = 0.70,  # Phase 2 → Phase 3 transition
    ):
        self.dropout_max = dropout_max
        self.dropout_min = dropout_min
        self.permute_max = permute_max
        self.permute_min = permute_min
        self.warmup_frac = warmup_frac
        self.cooldown_frac = cooldown_frac

    def get(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        """Return (dropout_prob, permute_prob) for given epoch."""
        progress = epoch / max(1, max_epochs - 1)

        if progress < self.warmup_frac:
            # Phase 1: HIGH masking — force feature learning
            drop = self.dropout_max
            perm = self.permute_max
        elif progress < self.cooldown_frac:
            # Phase 2: Linear ramp-down
            phase_progress = (progress - self.warmup_frac) / (
                self.cooldown_frac - self.warmup_frac
            )
            drop = self.dropout_max + (self.dropout_min - self.dropout_max) * phase_progress
            perm = self.permute_max + (self.permute_min - self.permute_max) * phase_progress
        else:
            # Phase 3: Low masking for fine-tuning
            drop = self.dropout_min
            perm = self.permute_min

        return float(drop), float(perm)

    def __repr__(self) -> str:
        return (
            f"ContextScheduler("
            f"drop={self.dropout_max:.0%}→{self.dropout_min:.0%}, "
            f"perm={self.permute_max:.0%}→{self.permute_min:.0%}, "
            f"phases=[0, {self.warmup_frac:.0%}, {self.cooldown_frac:.0%}, 100%])"
        )


# ═══════════════════════════════════════════════════════════════════════
#  2. Immediate Transition Penalty (no warmup delay)
# ═══════════════════════════════════════════════════════════════════════

class TransitionScheduler:
    """Transition penalty schedule that starts IMMEDIATELY.

    Problem in original: transition penalty warmup_frac=0.10 means
    the first 12 epochs have ZERO transition penalty → model already
    converges to persistence before the penalty kicks in.

    Fix: Start with a meaningful penalty from epoch 0, then ramp up.
    """

    def __init__(
        self,
        weight_init: float = 0.10,    # Start with non-zero weight
        weight_max: float = 0.30,     # Ramp to this maximum
        margin: float = 0.15,         # Logit margin for transition penalty
        ramp_frac: float = 0.40,      # Ramp up over first 40% of training
        # Auxiliary transition head
        aux_weight_init: float = 0.15,
        aux_weight_max: float = 0.20,
        aux_weight_min: float = 0.05,
        aux_cooldown_frac: float = 0.60,
    ):
        self.weight_init = weight_init
        self.weight_max = weight_max
        self.margin = margin
        self.ramp_frac = ramp_frac
        self.aux_weight_init = aux_weight_init
        self.aux_weight_max = aux_weight_max
        self.aux_weight_min = aux_weight_min
        self.aux_cooldown_frac = aux_cooldown_frac

    def get_transition_weight(self, epoch: int, max_epochs: int) -> float:
        """Get transition penalty weight — starts non-zero from epoch 0."""
        progress = epoch / max(1, max_epochs - 1)
        if progress < self.ramp_frac:
            # Ramp from init to max
            ramp_progress = progress / self.ramp_frac
            return self.weight_init + (self.weight_max - self.weight_init) * ramp_progress
        return self.weight_max

    def get_aux_weight(self, epoch: int, max_epochs: int) -> float:
        """Get auxiliary transition loss weight — starts non-zero from epoch 0."""
        progress = epoch / max(1, max_epochs - 1)
        if progress < self.ramp_frac:
            # Ramp from init to max
            ramp_progress = progress / self.ramp_frac
            return self.aux_weight_init + (
                self.aux_weight_max - self.aux_weight_init
            ) * ramp_progress
        elif progress < self.aux_cooldown_frac:
            return self.aux_weight_max
        else:
            # Cooldown: reduce aux weight for final fine-tuning
            cool_progress = (progress - self.aux_cooldown_frac) / (
                1.0 - self.aux_cooldown_frac
            )
            return self.aux_weight_max + (
                self.aux_weight_min - self.aux_weight_max
            ) * cool_progress


# ═══════════════════════════════════════════════════════════════════════
#  3. Change-Aware Weighted Sampler
# ═══════════════════════════════════════════════════════════════════════

def build_change_weighted_sampler(
    train_seqs: list,
    change_boost: float = 3.0,
    minority_boost: float = 2.0,
    n_classes: int = 22,
) -> torch.utils.data.WeightedRandomSampler:
    """Build a weighted sampler that oversamples rating-CHANGE windows.

    Problem: ~49% of samples are "stay" (y_t+1 = y_t). Without oversampling
    change events, the model has no incentive to learn transition patterns.

    Strategy:
      - Change samples (y_t+1 != y_t): weight *= change_boost
      - Minority class samples (freq < median): weight *= minority_boost
      - Stay + majority: weight = 1.0

    Args:
        train_seqs: List of (X, last_y, sector_id, y_target) tuples.
        change_boost: Weight multiplier for rating-change samples.
        minority_boost: Additional multiplier for rare-class samples.
        n_classes: Number of classes.

    Returns:
        WeightedRandomSampler instance.
    """
    labels = np.array([s[3] for s in train_seqs], dtype=int)
    last_ys = np.array([s[1] for s in train_seqs], dtype=int)
    is_change = labels != last_ys

    # Compute class frequencies
    class_freq = np.bincount(labels, minlength=n_classes).astype(float)
    median_freq = float(np.median(class_freq[class_freq > 0]))

    # Build weights
    weights = np.ones(len(train_seqs), dtype=np.float64)

    # Boost change samples
    weights[is_change] *= change_boost

    # Boost minority classes
    for i, (label, is_chg) in enumerate(zip(labels, is_change)):
        if class_freq[label] < median_freq:
            weights[i] *= minority_boost

    weights = torch.tensor(weights, dtype=torch.float64)

    n_change = int(is_change.sum())
    n_stay = len(is_change) - n_change
    print(f"  Sampler: {n_change} change ({n_change/len(is_change):.1%}) / "
          f"{n_stay} stay ({n_stay/len(is_change):.1%})")
    print(f"  Boost: change={change_boost}x, minority={minority_boost}x")
    print(f"  Effective weight range: [{weights.min():.1f}, {weights.max():.1f}]")

    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  4. Enhanced Transition Penalty (stronger margin enforcement)
# ═══════════════════════════════════════════════════════════════════════

def compute_transition_penalty_v2(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    last_y_true: torch.Tensor,
    margin: float = 0.15,
    ordinal_weight: float = 0.5,
) -> torch.Tensor:
    """Enhanced transition penalty with ordinal-aware distance weighting.

    For change samples (y_t+1 != y_t):
      - Enforce logit(true_class) > logit(last_y) + margin
      - Weight penalty by ordinal distance |y_true - last_y| to penalize
        large rating jumps more (they are harder and more important)

    Args:
        logits: (B, n_classes) raw logits.
        y_true: (B,) true target labels.
        last_y_true: (B,) previous rating labels.
        margin: Minimum logit gap to enforce.
        ordinal_weight: Weight for distance-scaled penalty component.
    """
    change_mask = y_true != last_y_true
    if not bool(change_mask.any()):
        return logits.new_tensor(0.0)

    logits_change = logits[change_mask]
    y_change = y_true[change_mask]
    last_change = last_y_true[change_mask]

    true_logit = logits_change.gather(1, y_change.unsqueeze(1)).squeeze(1)
    last_logit = logits_change.gather(1, last_change.unsqueeze(1)).squeeze(1)

    # Base margin penalty
    base_penalty = F.softplus(last_logit - true_logit + margin)

    # Ordinal distance weighting: larger jumps get higher penalty
    ordinal_dist = torch.abs(y_change.float() - last_change.float())
    ordinal_dist = ordinal_dist / ordinal_dist.max().clamp(min=1.0)  # normalize to [0, 1]

    weighted_penalty = base_penalty * (1.0 + ordinal_weight * ordinal_dist)
    return weighted_penalty.mean()


# ═══════════════════════════════════════════════════════════════════════
#  5. Recommended Full Config
# ═══════════════════════════════════════════════════════════════════════

ANTI_PERSISTENCE_CONFIG = {
    # --- Model (use V2 or V3) ---
    "hidden_size": 128,
    "dropout": 0.10,
    "fuzzy_mfs": 5,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "sector_emb_dim": 16,

    # --- Optimizer ---
    "lr": 3e-4,
    "max_lr": 1e-3,
    "weight_decay": 8e-4,

    # --- Loss ---
    "focal_gamma": 2.0,          # Increased: more focus on hard examples
    "ordinal_alpha": 0.05,       # Increased: stronger ordinal regularization
    "label_smoothing": 0.02,
    "class_weight_strategy": "cb_loss",
    "cb_beta": 0.999,

    # --- Context perturbation (Feature-First Curriculum) ---
    "ctx_dropout_max": 0.60,       # Phase 1: mask 60% of last_y
    "ctx_dropout_min": 0.05,       # Phase 3: mask 5%
    "ctx_permute_max": 0.30,       # Phase 1: permute 30%
    "ctx_permute_min": 0.02,       # Phase 3: permute 2%
    "ctx_warmup_frac": 0.30,       # Phase 1 ends at 30% of training
    "ctx_cooldown_frac": 0.70,     # Phase 3 starts at 70%

    # --- Transition penalty (IMMEDIATE, no warmup) ---
    "trans_weight_init": 0.10,     # Start with 0.10 from epoch 0
    "trans_weight_max": 0.30,      # Ramp to 0.30
    "trans_margin": 0.15,          # Logit margin
    "trans_ramp_frac": 0.40,       # Ramp over first 40% of training

    # --- Auxiliary transition loss ---
    "aux_weight_init": 0.15,       # Start non-zero from epoch 0
    "aux_weight_max": 0.20,
    "aux_weight_min": 0.05,

    # --- Persistence bias: DISABLED ---
    "persistence_bias": 0.00,

    # --- Sampler ---
    "use_change_sampler": True,
    "change_boost": 3.0,           # 3x weight for change samples
    "minority_boost": 2.0,         # 2x additional for minority classes

    # --- MixUp: DISABLED initially (let model learn clean signal first) ---
    "mixup_alpha": 0.0,
    "mixup_prob": 0.0,

    # --- Training ---
    "max_epochs": 120,
    "patience": 30,                # Longer patience for curriculum learning
    "grad_clip": 1.0,
}


# ═══════════════════════════════════════════════════════════════════════
#  COMPLETE TRAINING LOOP TEMPLATE
# ═══════════════════════════════════════════════════════════════════════

TRAINING_LOOP_TEMPLATE = '''
# ============================================================
# TRAINING LOOP V2 — Anti-Persistence Strategy
# ============================================================
# Paste this into Cell 11 of the notebook, replacing the entire
# training loop cell.
#
# Prerequisites (from earlier cells):
#   - train_seqs, val_seqs, test_seqs are built
#   - class_freq_raw is computed
#   - model (TLSTMFuzzyClassifierV2 or V3) is created
#   - device is set
# ============================================================

import sys
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from models.tlstm_fuzzy_v2 import (
    TLSTMFuzzyClassifierV2, FocalOrdinalLossV2,
    compute_class_weights, build_training_criterion_v2,
)
from models.training_utils import (
    ContextScheduler, TransitionScheduler,
    build_change_weighted_sampler, compute_transition_penalty_v2,
    ANTI_PERSISTENCE_CONFIG,
)

# --- Config ---
cfg = ANTI_PERSISTENCE_CONFIG
MAX_EPOCHS = cfg['max_epochs']
PATIENCE = cfg['patience']
EARLY_STOP_METRIC = 'val_f1_weighted'
EARLY_STOP_MIN_DELTA = 3e-4

# --- Model (rebuild with V2) ---
model = TLSTMFuzzyClassifierV2(
    n_channels=n_channels, n_classes=n_classes, n_sectors=n_sectors,
    hidden_size=cfg['hidden_size'], dropout=cfg['dropout'],
    n_mfs=cfg['fuzzy_mfs'], d_model=cfg['d_model'],
    n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
    sector_emb_dim=cfg['sector_emb_dim'],
    max_relative_position=32,
).to(device)

# --- Loss with class weights ---
CLASS_WEIGHTS = compute_class_weights(
    class_freq_raw, strategy=cfg['class_weight_strategy'], cb_beta=cfg['cb_beta']
).to(device)
criterion = FocalOrdinalLossV2(
    n_classes=n_classes,
    gamma=cfg['focal_gamma'],
    ordinal_alpha=cfg['ordinal_alpha'],
    label_smoothing=cfg['label_smoothing'],
    class_weights=CLASS_WEIGHTS,
)
transition_criterion = nn.BCEWithLogitsLoss()

# --- Optimizer + Scheduler ---
optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
)

# --- Change-aware sampler (replaces shuffle=True) ---
if cfg['use_change_sampler']:
    change_sampler = build_change_weighted_sampler(
        train_seqs, change_boost=cfg['change_boost'],
        minority_boost=cfg['minority_boost'], n_classes=n_classes,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=change_sampler,
        drop_last=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg['max_lr'],
    steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS,
    pct_start=0.2, anneal_strategy='cos',
    div_factor=max(cfg['max_lr'] / cfg['lr'], 1.0),
    final_div_factor=100.0,
)

# --- Curriculum schedulers ---
ctx_scheduler = ContextScheduler(
    dropout_max=cfg['ctx_dropout_max'], dropout_min=cfg['ctx_dropout_min'],
    permute_max=cfg['ctx_permute_max'], permute_min=cfg['ctx_permute_min'],
    warmup_frac=cfg['ctx_warmup_frac'], cooldown_frac=cfg['ctx_cooldown_frac'],
)
trans_scheduler = TransitionScheduler(
    weight_init=cfg['trans_weight_init'], weight_max=cfg['trans_weight_max'],
    margin=cfg['trans_margin'], ramp_frac=cfg['trans_ramp_frac'],
    aux_weight_init=cfg['aux_weight_init'], aux_weight_max=cfg['aux_weight_max'],
    aux_weight_min=cfg['aux_weight_min'],
)

print(f"Context schedule: {ctx_scheduler}")
print(f"Persistence bias: DISABLED")
print(f"Transition penalty: immediate, init={cfg['trans_weight_init']}, "
      f"max={cfg['trans_weight_max']}, margin={cfg['trans_margin']}")
print(f"MixUp: DISABLED (clean signal learning)")

# --- AMP ---
AMP_ENABLED = torch.cuda.is_available()
AMP_DEVICE = 'cuda' if AMP_ENABLED else 'cpu'
scaler_amp = torch.amp.GradScaler(AMP_DEVICE, enabled=AMP_ENABLED)

# --- Training history ---
history = {
    'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
    'train_f1_weighted': [], 'val_f1_weighted': [],
    'val_change_acc': [], 'val_stay_acc': [],
    'train_auc': [], 'val_auc': [], 'lr': [],
    'ctx_drop': [], 'ctx_perm': [],
    'trans_weight': [], 'aux_weight': [],
}

val_persistence_acc = float(np.mean([int(s[1] == s[3]) for s in val_seqs]))
val_is_change = np.array([int(s[1] != s[3]) for s in val_seqs], dtype=bool)
best_metric = -np.inf
best_epoch = -1
patience_counter = 0
BEST_MODEL_PATH = ARTIFACT_DIR / 'transformer_best_model.pt'

print(f"Val persistence baseline: {val_persistence_acc:.4f}")
print(f"Training for max {MAX_EPOCHS} epochs (patience={PATIENCE})...")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print()

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = []
    all_yt, all_logits = [], []

    # Get curriculum values for this epoch
    ctx_drop, ctx_perm = ctx_scheduler.get(epoch, MAX_EPOCHS)
    trans_w = trans_scheduler.get_transition_weight(epoch, MAX_EPOCHS)
    aux_w = trans_scheduler.get_aux_weight(epoch, MAX_EPOCHS)

    for X_batch, last_y_batch, sector_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        last_y_batch = last_y_batch.to(device, non_blocking=True)
        sector_batch = sector_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Context perturbation (Feature-First Curriculum)
        last_y_in = perturb_last_y_context(
            last_y_batch, n_cls=n_classes,
            drop_prob=ctx_drop, permute_prob=ctx_perm,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
            logits, transition_logits = model(
                X_batch, last_y_in, sector_batch, return_aux=True
            )
            # NO persistence bias
            base_loss = criterion(logits, y_batch)

            # Transition penalty (IMMEDIATE from epoch 0)
            trans_penalty = compute_transition_penalty_v2(
                logits, y_true=y_batch, last_y_true=last_y_batch,
                margin=cfg['trans_margin'],
            )

            # Auxiliary transition head (IMMEDIATE from epoch 0)
            change_targets = (y_batch != last_y_batch).float()
            aux_loss = transition_criterion(transition_logits, change_targets)

            loss = base_loss + trans_w * trans_penalty + aux_w * aux_loss

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        epoch_loss.append(loss.item())
        all_yt.append(y_batch.detach())
        all_logits.append(logits.detach())

    # --- Train metrics ---
    train_loss = np.mean(epoch_loss)
    all_yt = torch.cat(all_yt)
    all_logits = torch.cat(all_logits)
    tr_acc, _, tr_f1w, tr_auc, _ = compute_cls_metrics(all_yt, all_logits, n_classes)

    # --- Validation ---
    model.eval()
    vl_yt, vl_logits = [], []
    val_losses = []
    with torch.no_grad():
        for X_batch, last_y_batch, sector_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            last_y_batch = last_y_batch.to(device, non_blocking=True)
            sector_batch = sector_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
                logits = model(X_batch, last_y_batch, sector_batch)
                loss = criterion(logits, y_batch)
            val_losses.append(loss.item())
            vl_yt.append(y_batch)
            vl_logits.append(logits)

    val_loss = np.mean(val_losses)
    vl_yt = torch.cat(vl_yt)
    vl_logits = torch.cat(vl_logits)
    vl_acc, _, vl_f1w, vl_auc, _ = compute_cls_metrics(vl_yt, vl_logits, n_classes)

    # Change/stay accuracy
    vl_pred = vl_logits.argmax(dim=1).cpu().numpy()
    vl_true = vl_yt.cpu().numpy()
    vl_last = np.array([s[1] for s in val_seqs[:len(vl_true)]], dtype=int)
    chg_mask = vl_true != vl_last
    stay_mask = ~chg_mask
    chg_acc = float(np.mean(vl_pred[chg_mask] == vl_true[chg_mask])) if chg_mask.any() else 0.0
    stay_acc = float(np.mean(vl_pred[stay_mask] == vl_true[stay_mask])) if stay_mask.any() else 0.0

    gap = tr_f1w - vl_f1w
    uplift = vl_f1w - val_persistence_acc
    current_lr = optimizer.param_groups[0]['lr']

    # --- Log ---
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)
    history['train_f1_weighted'].append(tr_f1w)
    history['val_f1_weighted'].append(vl_f1w)
    history['val_change_acc'].append(chg_acc)
    history['val_stay_acc'].append(stay_acc)
    history['train_auc'].append(tr_auc)
    history['val_auc'].append(vl_auc)
    history['lr'].append(current_lr)
    history['ctx_drop'].append(ctx_drop)
    history['ctx_perm'].append(ctx_perm)
    history['trans_weight'].append(trans_w)
    history['aux_weight'].append(aux_w)

    print(
        f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
        f"TrL: {train_loss:.4f} | VlL: {val_loss:.4f} | "
        f"TrF1w: {tr_f1w:.3f} | VlF1w: {vl_f1w:.3f} | "
        f"Gap: {gap:+.3f} | Uplift: {uplift:+.3f} | "
        f"ChgAcc: {chg_acc:.3f} | StayAcc: {stay_acc:.3f} | "
        f"CtxDr: {ctx_drop:.2f} | TPw: {trans_w:.2f} | AuxW: {aux_w:.2f} | "
        f"LR: {current_lr:.6f}"
    )

    # --- Checkpoint (save if best val_f1_weighted) ---
    if vl_f1w > best_metric + EARLY_STOP_MIN_DELTA:
        best_metric = vl_f1w
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  ★ New best! val_f1w={vl_f1w:.4f} (uplift={uplift:+.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\\nEarly stopping at epoch {epoch+1}")
            break

print(f"\\nTraining completed! Best val F1w: {best_metric:.4f} at epoch {best_epoch}")
print(f"Val persistence baseline: {val_persistence_acc:.4f}")
print(f"Uplift over persistence: {best_metric - val_persistence_acc:+.4f}")
'''

if __name__ == "__main__":
    print("Anti-Persistence Training Utilities")
    print("=" * 50)
    ctx = ContextScheduler()
    print(f"Context Scheduler: {ctx}")
    for frac in [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]:
        epoch = int(frac * 120)
        d, p = ctx.get(epoch, 120)
        print(f"  Epoch {epoch:3d} ({frac:.0%}): drop={d:.2f}, perm={p:.2f}")

    print()
    ts = TransitionScheduler()
    print("Transition Scheduler:")
    for frac in [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]:
        epoch = int(frac * 120)
        tw = ts.get_transition_weight(epoch, 120)
        aw = ts.get_aux_weight(epoch, 120)
        print(f"  Epoch {epoch:3d} ({frac:.0%}): trans_w={tw:.3f}, aux_w={aw:.3f}")
