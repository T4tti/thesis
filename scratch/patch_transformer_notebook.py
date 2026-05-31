"""
Patch Transformer-LSTM.ipynb to improve test accuracy → ≥ 0.930

Implements Steps 1–5 from the implementation plan:
  Step 1: Early-stop metric alignment to accuracy
  Step 2: INPUT_SIZE=8 + model capacity increase
  Step 3: BiLSTM + PriMO expansion
  Step 4: SWA + seed ensemble expansion
  Step 5: Noise cleanup (transition_head conditional, label_smoothing=0)

How to Run:
  python scratch/patch_transformer_notebook.py

Expected Output:
  Patched notebook saved with backup at notebooks/Transformer-LSTM.ipynb.bak
"""

import json
import copy
import re
import sys
from pathlib import Path
from typing import List, Tuple

NB_PATH = Path('notebooks/Transformer-LSTM.ipynb')
BACKUP_PATH = NB_PATH.with_suffix('.ipynb.bak')


def load_notebook(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(nb: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False)
    print(f'  Saved: {path}')


def get_cell_source(nb: dict, cell_idx: int) -> str:
    """Get cell source as a single string."""
    src = nb['cells'][cell_idx]['source']
    if isinstance(src, list):
        return ''.join(src)
    return src


def set_cell_source(nb: dict, cell_idx: int, new_src: str) -> None:
    """Set cell source, preserving the original format (list of lines)."""
    original = nb['cells'][cell_idx]['source']
    if isinstance(original, list):
        # Keep the list-of-lines format
        lines = new_src.split('\n')
        # Each line except the last should end with \n
        result = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                result.append(line + '\n')
            else:
                result.append(line)
        nb['cells'][cell_idx]['source'] = result
    else:
        nb['cells'][cell_idx]['source'] = new_src


def safe_replace(src: str, old: str, new: str, description: str, count: int = 1) -> str:
    """Replace with validation — raises if old not found."""
    occurrences = src.count(old)
    if occurrences == 0:
        raise ValueError(f'[FAIL] "{description}": pattern not found in source.\n  Pattern: {repr(old[:120])}')
    if occurrences > 1 and count == 1:
        print(f'  [WARN] "{description}": found {occurrences} occurrences, replacing first only.')
    result = src.replace(old, new, count)
    print(f'  [OK] {description}')
    return result


def apply_patches(nb: dict) -> dict:
    """Apply all patches. Returns modified notebook."""

    # ================================================================
    # STEP 1: Align early-stop metric to accuracy
    # ================================================================
    print('\n=== STEP 1: Align early-stop metric to accuracy ===')

    # --- Cell 27: Training loop ---
    src27 = get_cell_source(nb, 27)

    # 1a. Change EARLY_STOP_METRIC
    src27 = safe_replace(
        src27,
        "EARLY_STOP_METRIC = 'val_f1_weighted_gap_guarded'",
        "EARLY_STOP_METRIC = 'val_accuracy'",
        'Cell 27: EARLY_STOP_METRIC → val_accuracy',
    )

    # 1b. Change checkpoint selection value
    src27 = safe_replace(
        src27,
        '    current_metric = float(generalization_score)',
        '    current_metric = float(vl_acc)',
        'Cell 27: checkpoint selection → vl_acc',
    )

    # 1c. Disable label_smoothing floor
    src27 = safe_replace(
        src27,
        'MIN_LABEL_SMOOTHING_FOR_GENERALIZATION = 0.02',
        'MIN_LABEL_SMOOTHING_FOR_GENERALIZATION = 0.0',
        'Cell 27: disable label_smoothing floor',
    )

    # 1d. Increase weight_decay floor
    src27 = safe_replace(
        src27,
        "MIN_WEIGHT_DECAY_FOR_GENERALIZATION = 6e-3",
        "MIN_WEIGHT_DECAY_FOR_GENERALIZATION = 8e-3",
        'Cell 27: weight_decay floor → 8e-3',
    )

    set_cell_source(nb, 27, src27)

    # --- Cell 24: PriMO objective weights ---
    src24 = get_cell_source(nb, 24)

    src24 = safe_replace(
        src24,
        """PRIMO_OBJECTIVE_WEIGHTS = {
    'val_accuracy': 0.45,
    'val_weighted_f1': 0.25,
    'val_macro_f1': 0.20,
    'val_qwk': 0.10,
}""",
        """PRIMO_OBJECTIVE_WEIGHTS = {
    'val_accuracy': 0.65,
    'val_weighted_f1': 0.15,
    'val_macro_f1': 0.10,
    'val_qwk': 0.10,
}""",
        'Cell 24: PriMO objective → accuracy=0.65',
    )

    # --- Cell 30: Seed ensemble objective ---
    src30 = get_cell_source(nb, 30)

    src30 = safe_replace(
        src30,
        "return float(0.45 * metrics['accuracy'] + 0.25 * metrics['f1_weighted'] + 0.20 * metrics['f1_macro'] + 0.10 * metrics['qwk'])",
        "return float(0.70 * metrics['accuracy'] + 0.15 * metrics['f1_weighted'] + 0.10 * metrics['f1_macro'] + 0.05 * metrics['qwk'])",
        'Cell 30: seed_ensemble_objective → accuracy=0.70',
    )

    # ================================================================
    # STEP 2: Increase INPUT_SIZE + Model Capacity
    # ================================================================
    print('\n=== STEP 2: Increase INPUT_SIZE + Model Capacity ===')

    # --- Cell 16: INPUT_SIZE ---
    src16 = get_cell_source(nb, 16)

    src16 = safe_replace(
        src16,
        "INPUT_SIZE_DEFAULT = 4  # Strategy A: increased from 1 to activate temporal modeling",
        "INPUT_SIZE_DEFAULT = 8  # Increased: 8 steps covers ~2 years of quarterly financials",
        'Cell 16: INPUT_SIZE_DEFAULT → 8',
    )

    set_cell_source(nb, 16, src16)

    # --- Cell 23: Model defaults ---
    src23 = get_cell_source(nb, 23)

    src23 = safe_replace(src23, 'MODEL_D_MODEL = 64', 'MODEL_D_MODEL = 128', 'Cell 23: d_model → 128')
    src23 = safe_replace(src23, 'TRANSFORMER_HEADS = 4', 'TRANSFORMER_HEADS = 8', 'Cell 23: heads → 8')
    src23 = safe_replace(src23, 'TRANSFORMER_LAYERS = 1', 'TRANSFORMER_LAYERS = 2', 'Cell 23: layers → 2')
    src23 = safe_replace(src23, 'LSTM_HIDDEN = 64', 'LSTM_HIDDEN = 128', 'Cell 23: hidden → 128')
    src23 = safe_replace(src23, 'TICKER_EMB_DIM = 4', 'TICKER_EMB_DIM = 16', 'Cell 23: ticker_emb → 16')
    src23 = safe_replace(src23, 'COMPANY_EMB_DIM = 4', 'COMPANY_EMB_DIM = 16', 'Cell 23: company_emb → 16')
    src23 = safe_replace(src23, 'TLSTM_DROPOUT = 0.30', 'TLSTM_DROPOUT = 0.35', 'Cell 23: dropout → 0.35')

    set_cell_source(nb, 23, src23)

    # --- Cell 24: PriMO default config ---
    src24 = safe_replace(
        src24,
        """TRANSFORMER_PRIMO_DEFAULT_CONFIG = {
    'hidden_size': 64,
    'dropout': 0.30,
    'fuzzy_mfs': 1,
    'use_fuzzy': False,
    'bidirectional_lstm': False,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 1,
    'sector_emb_dim': 16,
    'ticker_emb_dim': 4,
    'company_emb_dim': 4,
    'max_relative_position': 8,
    'lr': 2.5e-4,
    'max_lr': 4.0e-4,
    'weight_decay': 6e-3,
    'focal_gamma': 1.5,
    'ordinal_alpha': 0.08,
    'label_smoothing': 0.02,
    'class_balance_beta': 0.995,
    'use_class_weights': False,
}""",
        """TRANSFORMER_PRIMO_DEFAULT_CONFIG = {
    'hidden_size': 128,
    'dropout': 0.35,
    'fuzzy_mfs': 1,
    'use_fuzzy': False,
    'bidirectional_lstm': False,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 2,
    'sector_emb_dim': 16,
    'ticker_emb_dim': 16,
    'company_emb_dim': 16,
    'max_relative_position': 8,
    'lr': 2.5e-4,
    'max_lr': 4.0e-4,
    'weight_decay': 8e-3,
    'focal_gamma': 1.5,
    'ordinal_alpha': 0.08,
    'label_smoothing': 0.0,
    'class_balance_beta': 0.995,
    'use_class_weights': False,
}""",
        'Cell 24: TRANSFORMER_PRIMO_DEFAULT_CONFIG → high capacity',
    )

    # --- Cell 24: PriMO search space ---
    src24 = safe_replace(
        src24,
        """TRANSFORMER_PRIMO_SPACE = {
    'hidden_size': [48, 64, 96],
    'dropout': (0.18, 0.40),
    'fuzzy_mfs': [1, 3, 5],
    'use_fuzzy': [False, True],
    'bidirectional_lstm': [False, True],
    'd_model': [48, 64, 96],
    'n_heads': [2, 4, 8],
    'n_layers': [0, 1, 2],
    'sector_emb_dim': [8, 16],
    'ticker_emb_dim': [2, 4, 8],
    'company_emb_dim': [2, 4, 8],
    'max_relative_position': [4, 8, 16],""",
        """TRANSFORMER_PRIMO_SPACE = {
    'hidden_size': [64, 96, 128, 192],
    'dropout': (0.20, 0.45),
    'fuzzy_mfs': [1, 3, 5],
    'use_fuzzy': [False, True],
    'bidirectional_lstm': [False, True],
    'd_model': [64, 96, 128, 192],
    'n_heads': [4, 8],
    'n_layers': [1, 2, 3],
    'sector_emb_dim': [8, 16],
    'ticker_emb_dim': [4, 8, 16],
    'company_emb_dim': [4, 8, 16],
    'max_relative_position': [4, 8, 16],""",
        'Cell 24: TRANSFORMER_PRIMO_SPACE → expanded',
    )

    # ================================================================
    # STEP 3: BiLSTM + PriMO Expansion
    # ================================================================
    print('\n=== STEP 3: BiLSTM + PriMO Expansion ===')

    # --- Add BiLSTM priors ---
    src24 = safe_replace(
        src24,
        "{**TRANSFORMER_PRIMO_DEFAULT_CONFIG, 'bidirectional_lstm': True, 'hidden_size': 48, 'dropout': 0.32, 'prior_name': 'bilstm_ablation_small'},\n]",
        """{**TRANSFORMER_PRIMO_DEFAULT_CONFIG, 'bidirectional_lstm': True, 'hidden_size': 96, 'dropout': 0.32, 'prior_name': 'bilstm_input8'},
    {**TRANSFORMER_PRIMO_DEFAULT_CONFIG, 'hidden_size': 128, 'd_model': 128, 'n_layers': 2, 'n_heads': 8,
     'bidirectional_lstm': True, 'ticker_emb_dim': 16, 'company_emb_dim': 16,
     'dropout': 0.35, 'weight_decay': 8e-3, 'prior_name': 'high_capacity_bilstm'},
]""",
        'Cell 24: Add BiLSTM priors',
    )

    # --- Increase BiLSTM mutation probability ---
    src24 = safe_replace(
        src24,
        "cfg['bidirectional_lstm'] = bool(rng.choice(TRANSFORMER_PRIMO_SPACE['bidirectional_lstm'])) if rng.random() < 0.20",
        "cfg['bidirectional_lstm'] = bool(rng.choice(TRANSFORMER_PRIMO_SPACE['bidirectional_lstm'])) if rng.random() < 0.40",
        'Cell 24: BiLSTM mutation prob → 0.40',
    )

    # --- Increase PriMO budget ---
    src24 = safe_replace(src24, 'PRIMO_TRIALS = 12', 'PRIMO_TRIALS = 24', 'Cell 24: PRIMO_TRIALS → 24')
    src24 = safe_replace(src24, 'PRIMO_FIDELITY_EPOCHS = 8', 'PRIMO_FIDELITY_EPOCHS = 15', 'Cell 24: PRIMO_FIDELITY_EPOCHS → 15')
    src24 = safe_replace(src24, 'PRIMO_PATIENCE = 3', 'PRIMO_PATIENCE = 5', 'Cell 24: PRIMO_PATIENCE → 5')
    src24 = safe_replace(src24, 'PRIMO_TRAIN_FRACTION = 0.50', 'PRIMO_TRAIN_FRACTION = 0.70', 'Cell 24: PRIMO_TRAIN_FRACTION → 0.70')

    set_cell_source(nb, 24, src24)

    # --- Cell 27: strategy overrides ---
    src27 = get_cell_source(nb, 27)

    src27 = safe_replace(
        src27,
        """_strategy_overrides = {
    'hidden_size': 64,
    'd_model': 64,
    'n_layers': 1,
    'dropout': 0.30,
    'use_fuzzy': False,
    'bidirectional_lstm': False,
    'fuzzy_mfs': 1,
    'ticker_emb_dim': 4,
    'company_emb_dim': 4,
    'ordinal_alpha': 0.08,
    'weight_decay': 6e-3,
    'label_smoothing': 0.02,
    'focal_gamma': 1.5,
    'use_class_weights': False,
}""",
        """_strategy_overrides = {
    'hidden_size': 128,
    'd_model': 128,
    'n_layers': 2,
    'dropout': 0.35,
    'use_fuzzy': False,
    'bidirectional_lstm': False,
    'fuzzy_mfs': 1,
    'ticker_emb_dim': 16,
    'company_emb_dim': 16,
    'ordinal_alpha': 0.08,
    'weight_decay': 8e-3,
    'label_smoothing': 0.0,
    'focal_gamma': 1.5,
    'use_class_weights': False,
}""",
        'Cell 27: _strategy_overrides → high capacity',
    )

    # Also update the default_train_config in Cell 27
    src27 = safe_replace(
        src27,
        """default_train_config = {
    'hidden_size': 64,
    'dropout': 0.30,
    'fuzzy_mfs': 1,
    'use_fuzzy': False,
    'bidirectional_lstm': False,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 1,
    'sector_emb_dim': 16,
    'ticker_emb_dim': 4,
    'company_emb_dim': 4,
    'max_relative_position': 8,
    'lr': 2.5e-4,
    'max_lr': 4.0e-4,
    'weight_decay': 6e-3,
    'focal_gamma': 1.5,
    'ordinal_alpha': 0.08,
    'label_smoothing': 0.02,
    'class_balance_beta': 0.995,
    'use_class_weights': False,
}""",
        """default_train_config = {
    'hidden_size': 128,
    'dropout': 0.35,
    'fuzzy_mfs': 1,
    'use_fuzzy': False,
    'bidirectional_lstm': False,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 2,
    'sector_emb_dim': 16,
    'ticker_emb_dim': 16,
    'company_emb_dim': 16,
    'max_relative_position': 8,
    'lr': 2.5e-4,
    'max_lr': 4.0e-4,
    'weight_decay': 8e-3,
    'focal_gamma': 1.5,
    'ordinal_alpha': 0.08,
    'label_smoothing': 0.0,
    'class_balance_beta': 0.995,
    'use_class_weights': False,
}""",
        'Cell 27: default_train_config → high capacity',
    )

    set_cell_source(nb, 27, src27)

    # ================================================================
    # STEP 4: SWA + Expand Seed Ensemble
    # ================================================================
    print('\n=== STEP 4: SWA + Expand Seed Ensemble ===')

    # --- Cell 30: Increase ensemble runs ---
    src30 = safe_replace(
        src30,
        "SEED_ENSEMBLE_TOTAL_RUNS = 3  # Set to 5 on Kaggle if runtime budget allows.",
        "SEED_ENSEMBLE_TOTAL_RUNS = 5  # Expanded for better ensemble averaging.",
        'Cell 30: SEED_ENSEMBLE_TOTAL_RUNS → 5',
    )
    src30 = safe_replace(
        src30,
        'SEED_ENSEMBLE_TOP_K = 2',
        'SEED_ENSEMBLE_TOP_K = 3',
        'Cell 30: SEED_ENSEMBLE_TOP_K → 3',
    )

    set_cell_source(nb, 30, src30)

    # --- Add SWA to training loop (Cell 27) ---
    # Insert SWA logic after the training loop completes but before the final save
    src27 = get_cell_source(nb, 27)

    swa_block = """
# ============================================================
# Stochastic Weight Averaging (SWA) — find flat minima
# ============================================================
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

SWA_ENABLED = True
SWA_START_EPOCH_FRAC = 0.75
SWA_LR_VALUE = 1e-4

if SWA_ENABLED and best_state is not None:
    print('\\n[SWA] Starting Stochastic Weight Averaging...')
    # Reload best model state as starting point for SWA
    swa_base_model = TLSTMFuzzyClassifier(
        n_channels=n_channels, n_classes=n_classes, n_sectors=n_sectors,
        n_tickers=n_tickers, n_companies=n_companies,
        hidden_size=LSTM_HIDDEN, dropout=TLSTM_DROPOUT, n_mfs=FUZZY_MFS,
        d_model=MODEL_D_MODEL, n_heads=TRANSFORMER_HEADS, n_layers=TRANSFORMER_LAYERS,
        sector_emb_dim=SECTOR_EMB_DIM, ticker_emb_dim=TICKER_EMB_DIM,
        company_emb_dim=COMPANY_EMB_DIM, max_relative_position=MAX_RELATIVE_POSITION,
        use_fuzzy=USE_FUZZY_LAYER, bidirectional_lstm=BIDIRECTIONAL_LSTM,
    ).to(device)
    swa_base_model.load_state_dict(best_state)

    swa_model = AveragedModel(swa_base_model)
    swa_optimizer = torch.optim.AdamW(swa_base_model.parameters(), lr=SWA_LR_VALUE, weight_decay=WEIGHT_DECAY)
    swa_scheduler = SWALR(swa_optimizer, swa_lr=SWA_LR_VALUE, anneal_epochs=5)

    SWA_EPOCHS = 15  # Short SWA fine-tuning
    for swa_epoch in range(SWA_EPOCHS):
        swa_base_model.train()
        for X_batch, last_y_batch, sector_batch, ticker_batch, company_batch, y_batch, row_id_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=PIN_MEMORY)
            last_y_batch = last_y_batch.to(device, non_blocking=PIN_MEMORY)
            sector_batch = sector_batch.to(device, non_blocking=PIN_MEMORY)
            ticker_batch = ticker_batch.to(device, non_blocking=PIN_MEMORY)
            company_batch = company_batch.to(device, non_blocking=PIN_MEMORY)
            y_batch = y_batch.to(device, non_blocking=PIN_MEMORY)
            swa_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=AMP_DEVICE, enabled=AMP_ENABLED):
                logits = swa_base_model(X_batch, last_y_batch, sector_batch, ticker_batch, company_batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, y_batch)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(swa_base_model.parameters(), 1.0)
                swa_optimizer.step()
        swa_model.update_parameters(swa_base_model)
        swa_scheduler.step()
        if (swa_epoch + 1) % 5 == 0 or swa_epoch == 0:
            print(f'[SWA] Epoch {swa_epoch + 1}/{SWA_EPOCHS} completed')

    # Update BatchNorm statistics
    print('[SWA] Updating BatchNorm statistics...')
    update_bn(train_loader, swa_model, device=device)

    # Compare SWA model vs best checkpoint on validation
    swa_model.eval()
    swa_vl_yt, swa_vl_logits = [], []
    with torch.no_grad():
        for X_batch, last_y_batch, sector_batch, ticker_batch, company_batch, y_batch, row_id_batch in val_loader:
            X_batch = X_batch.to(device)
            last_y_batch = last_y_batch.to(device)
            sector_batch = sector_batch.to(device)
            ticker_batch = ticker_batch.to(device)
            company_batch = company_batch.to(device)
            logits = swa_model(X_batch, last_y_batch, sector_batch, ticker_batch, company_batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            swa_vl_yt.append(y_batch)
            swa_vl_logits.append(logits.cpu())

    swa_vl_yt = torch.cat(swa_vl_yt)
    swa_vl_logits = torch.cat(swa_vl_logits)
    swa_acc, _, swa_f1w, swa_qwk, swa_auc = compute_cls_metrics(swa_vl_yt, swa_vl_logits, n_classes)
    print(f'[SWA] Val accuracy: {swa_acc:.4f} | F1w: {swa_f1w:.4f} | QWK: {swa_qwk:.4f}')
    print(f'[SWA] Best checkpoint val accuracy: {best_metric_value:.4f}')

    if swa_acc > best_metric_value:
        print('[SWA] SWA model is BETTER — using SWA weights.')
        # Extract the inner model weights from AveragedModel
        swa_state = {k.replace('module.', ''): v.cpu().clone()
                     for k, v in swa_model.state_dict().items() if k.startswith('module.')}
        if not swa_state:
            swa_state = {k: v.cpu().clone() for k, v in swa_model.state_dict().items()}
        best_state = swa_state
        best_metric_value = swa_acc
        model.load_state_dict(best_state)
        model.to(device)
        torch.save(best_state, BEST_MODEL_PATH)
    else:
        print('[SWA] Best checkpoint is still better — keeping original.')

    del swa_model, swa_base_model, swa_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""

    # Insert SWA block before the final metadata save
    anchor = "print(f'Training completed! Best {EARLY_STOP_METRIC}: {best_metric_value:.4f} at epoch {best_epoch}')"
    src27 = safe_replace(
        src27,
        anchor,
        anchor + swa_block,
        'Cell 27: Insert SWA block after training',
    )

    set_cell_source(nb, 27, src27)

    # ================================================================
    # STEP 5: Clean Up Noise
    # ================================================================
    print('\n=== STEP 5: Clean Up Noise ===')

    # --- Cell 23: Make transition_head conditional ---
    src23 = get_cell_source(nb, 23)

    # Add use_transition_head parameter to __init__
    src23 = safe_replace(
        src23,
        '        bidirectional_lstm=False,\n    ):\n        super().__init__()',
        '        bidirectional_lstm=False,\n        use_transition_head=False,\n    ):\n        super().__init__()',
        'Cell 23: Add use_transition_head param',
    )

    # Add instance variable and conditionally create transition_head
    src23 = safe_replace(
        src23,
        "        self.n_layers = int(n_layers)",
        "        self.n_layers = int(n_layers)\n        self.use_transition_head = bool(use_transition_head)",
        'Cell 23: Store use_transition_head',
    )

    src23 = safe_replace(
        src23,
        """        transition_in_dim = self.lstm_out_dim + sector_emb_dim + ticker_emb_dim + company_emb_dim
        self.transition_head = nn.Sequential(
            nn.Linear(transition_in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )""",
        """        transition_in_dim = self.lstm_out_dim + sector_emb_dim + ticker_emb_dim + company_emb_dim
        if self.use_transition_head:
            self.transition_head = nn.Sequential(
                nn.Linear(transition_in_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )""",
        'Cell 23: Make transition_head conditional',
    )

    # Update forward to conditionally use transition_head
    src23 = safe_replace(
        src23,
        """        aux_context = torch.cat([seq_ctx, sector_emb, ticker_emb, company_emb], dim=-1)
        transition_logits = self.transition_head(aux_context).squeeze(-1)

        out = torch.cat([seq_ctx, last_y_emb, sector_emb, ticker_emb, company_emb], dim=-1)
        logits = self.head(out)
        if return_aux:
            return logits, transition_logits
        return logits""",
        """        out = torch.cat([seq_ctx, last_y_emb, sector_emb, ticker_emb, company_emb], dim=-1)
        logits = self.head(out)
        if self.use_transition_head and return_aux:
            aux_context = torch.cat([seq_ctx, sector_emb, ticker_emb, company_emb], dim=-1)
            transition_logits = self.transition_head(aux_context).squeeze(-1)
            return logits, transition_logits
        return logits""",
        'Cell 23: Guard transition_head in forward()',
    )

    set_cell_source(nb, 23, src23)

    # --- Cell 21: Confirm label_smoothing=0.0 ---
    src21 = get_cell_source(nb, 21)

    # Already 0.0 in criterion_settings based on extraction, but let's be safe
    if "'label_smoothing': 0.00" in src21 or "'label_smoothing': 0.0" in src21:
        print('  [OK] Cell 21: label_smoothing already 0.0 in criterion_settings')
    else:
        src21 = safe_replace(
            src21,
            "'label_smoothing': 0.02",
            "'label_smoothing': 0.0",
            'Cell 21: label_smoothing → 0.0',
        )
        set_cell_source(nb, 21, src21)

    # ================================================================
    # STEP 5b: Threshold calibration objective → accuracy-first
    # ================================================================
    print('\n=== STEP 5b: Threshold calibration objective ===')
    src32 = get_cell_source(nb, 32)

    src32 = safe_replace(
        src32,
        """THRESHOLD_OBJECTIVE_WEIGHTS = {
    'accuracy': 1.00,
    'f1_weighted': 0.15,
    'f1_macro': 0.30,
    'qwk': 0.10,
}""",
        """THRESHOLD_OBJECTIVE_WEIGHTS = {
    'accuracy': 1.00,
    'f1_weighted': 0.10,
    'f1_macro': 0.20,
    'qwk': 0.05,
}""",
        'Cell 32: threshold objective → accuracy dominant',
    )

    set_cell_source(nb, 32, src32)

    # ================================================================
    # Add improvement_log cell at the end (before the summary)
    # ================================================================
    print('\n=== Adding improvement_log cell ===')

    improvement_log_cell = {
        "cell_type": "code",
        "source": [
            "# ============================================================\n",
            "# Improvement Log — track accuracy gains per step\n",
            "# ============================================================\n",
            "import pandas as pd\n",
            "\n",
            "improvement_log = pd.DataFrame([\n",
            "    {'step': 'baseline',  'change': 'original notebook (generalization_score early-stop)',\n",
            "     'val_acc': 0.925, 'test_acc': 0.9158, 'test_f1w': 0.9179},\n",
            "    {'step': 'Steps1-5', 'change': 'accuracy-stop + d128 + BiLSTM + SWA + 5-ensemble',\n",
            "     'val_acc': float(best_metric_value) if 'best_metric_value' in dir() else 0.0,\n",
            "     'test_acc': float(acc) if 'acc' in dir() else 0.0,\n",
            "     'test_f1w': float(f1_weighted) if 'f1_weighted' in dir() else 0.0},\n",
            "])\n",
            "print('='*60)\n",
            "print('IMPROVEMENT LOG')\n",
            "print('='*60)\n",
            "print(improvement_log.to_string(index=False))\n",
            "print()\n",
            "if 'acc' in dir() and float(acc) >= 0.930:\n",
            "    print('TARGET REACHED: test_accuracy >= 0.930')\n",
            "else:\n",
            "    print(f'TARGET NOT YET REACHED. Current test_acc = {float(acc) if \"acc\" in dir() else \"N/A\"}')\n",
        ],
        "metadata": {"tags": []},
        "outputs": [],
        "execution_count": None,
    }

    # Insert before the last cell (summary/export)
    nb['cells'].insert(len(nb['cells']) - 2, improvement_log_cell)
    print('  [OK] Added improvement_log cell')

    return nb


def main():
    print(f'Loading notebook: {NB_PATH}')
    nb = load_notebook(NB_PATH)
    print(f'Total cells: {len(nb["cells"])}')

    # Backup
    save_notebook(nb, BACKUP_PATH)
    print(f'Backup saved: {BACKUP_PATH}')

    # Apply patches
    nb_patched = apply_patches(copy.deepcopy(nb))

    # Save patched notebook
    save_notebook(nb_patched, NB_PATH)
    print(f'\n{"="*60}')
    print(f'All patches applied successfully!')
    print(f'Patched notebook: {NB_PATH}')
    print(f'Backup: {BACKUP_PATH}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
