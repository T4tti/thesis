import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reconfigure stdout for UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Execute the setup code in the global namespace
print("Running setup compiled...")
with open("e:/thesis/scratch/setup_compiled.py", "r", encoding="utf-8") as f:
    setup_code = f.read()
exec(setup_code, globals())

# Now we have all the setup variables:
# device, df, train_mask, val_mask, test_mask, x_all, y_all, last_y_all, sector_all,
# n_classes, n_sectors, MODEL_FEATURES, CreditGAT, build_loss, evaluate_logits, etc.

print("Setup completed successfully.")
print(f"Device: {device}")

# Define DropEdge
def apply_dropedge(edge_index, edge_weight, n_nodes, p=0.12):
    if p <= 0:
        return edge_index, edge_weight

    dst, src = edge_index
    is_self_loop = dst.eq(src)

    keep = is_self_loop | (
        torch.rand(edge_weight.size(0), device=edge_weight.device) > p
    )

    new_edge_index = edge_index[:, keep]
    new_edge_weight = edge_weight[keep]

    new_dst = new_edge_index[0]
    row_sum = torch.zeros(n_nodes, device=edge_weight.device)
    row_sum.scatter_add_(0, new_dst, new_edge_weight)

    new_edge_weight = new_edge_weight / row_sum[new_dst].clamp_min(1e-12)

    return new_edge_index, new_edge_weight

# Custom Checkpoint Score
def checkpoint_score(metrics, val_loss, train_loss):
    loss_gap = max(0.0, float(val_loss) - float(train_loss))
    score = (
        0.45 * metrics.get('Accuracy', metrics.get('accuracy'))
        + 0.35 * metrics.get('Macro_F1', metrics.get('macro_f1'))
        + 0.10 * metrics.get('balanced_accuracy', metrics.get('Accuracy', metrics.get('accuracy')))
        - 0.10 * loss_gap
    )
    return score

# Define configs for dry run (3 epochs each)
configs = {
    'Config A': {
        'lr': 1e-3,
        'weight_decay': 5e-5,
        'patience': 15,
        'SPARSE_GRAPH_HIDDEN': 96,
        'SPARSE_GRAPH_LAYERS': 3,
        'dropout': 0.28,
        'context_dropout': 0.16,
        'LABEL_SMOOTHING': 0.0,
        'PERSISTENCE_PRIOR_SCALE': 2.0,
        'CONTEXT_MASK_START': 0.20,
        'CONTEXT_MASK_END': 0.02,
        'CONTEXT_MASK_WARMUP_EPOCHS': 25,
        'EDGE_DROPOUT': 0.0,
        'LOSS_PROTOCOL': 'benchmark_ce',
        'ORDINAL_LAMBDA': 0.10,
        'plateau_factor': 0.60,
        'plateau_patience': 8,
        'min_lr': 1e-5,
        'mode': 'balanced_loss',
    },
    'Config B': {
        'lr': 6e-4,
        'weight_decay': 2e-4,
        'patience': 15,
        'SPARSE_GRAPH_HIDDEN': 72,
        'SPARSE_GRAPH_LAYERS': 2,
        'dropout': 0.32,
        'context_dropout': 0.20,
        'LABEL_SMOOTHING': 0.04,
        'PERSISTENCE_PRIOR_SCALE': 1.25,
        'CONTEXT_MASK_START': 0.25,
        'CONTEXT_MASK_END': 0.08,
        'CONTEXT_MASK_WARMUP_EPOCHS': 60,
        'EDGE_DROPOUT': 0.12,
        'LOSS_PROTOCOL': 'benchmark_ce',
        'ORDINAL_LAMBDA': 0.10,
        'plateau_factor': 0.5,
        'plateau_patience': 5,
        'min_lr': 3e-5,
        'mode': 'balanced_loss',
    },
    'Config C': {
        'lr': 6e-4,
        'weight_decay': 2e-4,
        'patience': 15,
        'SPARSE_GRAPH_HIDDEN': 72,
        'SPARSE_GRAPH_LAYERS': 2,
        'dropout': 0.32,
        'context_dropout': 0.20,
        'LABEL_SMOOTHING': 0.04,
        'PERSISTENCE_PRIOR_SCALE': 1.25,
        'CONTEXT_MASK_START': 0.25,
        'CONTEXT_MASK_END': 0.08,
        'CONTEXT_MASK_WARMUP_EPOCHS': 60,
        'EDGE_DROPOUT': 0.12,
        'LOSS_PROTOCOL': 'ordinal_ce_emd',
        'ORDINAL_LAMBDA': 0.03,
        'plateau_factor': 0.5,
        'plateau_patience': 5,
        'min_lr': 3e-5,
        'mode': 'balanced_loss',
    }
}

ablation_results = []
run_histories = {}

def scheduled_context_mask(epoch, warmup_epochs, start_val, end_val):
    progress = min(1.0, max(0.0, (float(epoch) - 1.0) / float(max(1, warmup_epochs - 1))))
    return start_val + (end_val - start_val) * progress

max_epochs = 3  # dry run
min_epochs = 2
min_delta = 1e-4

for config_name, config in configs.items():
    print(f"\n================ Running {config_name} ================")
    
    # Update global LABEL_SMOOTHING so loss function sees it
    global LABEL_SMOOTHING
    LABEL_SMOOTHING = config['LABEL_SMOOTHING']
    
    # Instantiate model
    model = CreditGAT(
        n_features=len(MODEL_FEATURES),
        n_classes=n_classes,
        n_sectors=n_sectors,
        n_nodes=x_all.size(0),
        edge_index=edge_index,
        edge_weight=edge_weight,
        hidden=config['SPARSE_GRAPH_HIDDEN'],
        num_layers=config['SPARSE_GRAPH_LAYERS'],
        dropout=config['dropout'],
        last_y_emb_dim=LAST_Y_EMB_DIM,
        sector_emb_dim=SECTOR_EMB_DIM,
        context_dropout=config['context_dropout'],
        persistence_prior_scale=config['PERSISTENCE_PRIOR_SCALE'],
    ).to(device)

    # Instantiate criterion
    criterion = build_loss(
        protocol=config['LOSS_PROTOCOL'],
        ordinal_lambda=config['ORDINAL_LAMBDA'],
    ).to(device)

    # Instantiate optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['plateau_factor'],
        patience=config['plateau_patience'],
        min_lr=config['min_lr'],
    )

    # Reset tracking
    best_score = -np.inf
    best_state = None
    best_epoch = 0
    best_payload = {}

    best_loss_value = np.inf
    best_loss_state = None
    best_loss_epoch = None
    best_loss_payload = {}

    best_balanced_loss = np.inf
    best_balanced_metric_score = -np.inf
    best_balanced_state = None
    best_balanced_epoch = None
    best_balanced_payload = {}

    no_improve = 0

    history = {
        'epoch': [],
        'train_Loss': [], 'val_Loss': [],
        'train_CE_Loss': [], 'val_CE_Loss': [],
        'train_NLL': [], 'val_NLL': [],
        'val_LossGap': [],
        'Learning_Rate': [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        
        # DropEdge
        train_edge_index, train_edge_weight = apply_dropedge(
            edge_index,
            edge_weight,
            x_all.size(0),
            p=config['EDGE_DROPOUT']
        )
        
        current_context_mask = scheduled_context_mask(
            epoch, 
            config['CONTEXT_MASK_WARMUP_EPOCHS'],
            config['CONTEXT_MASK_START'],
            config['CONTEXT_MASK_END']
        )
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            x_all, last_y_all, sector_all, 
            edge_index=train_edge_index, 
            edge_weight=train_edge_weight, 
            context_mask_prob=current_context_mask
        )
        loss = criterion(logits[train_mask], y_all[train_mask], epoch=epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_eval = model(x_all, last_y_all, sector_all)
            train_loss_parts = criterion.loss_parts(logits_eval[train_mask], y_all[train_mask], epoch=epoch)
            val_loss_parts = criterion.loss_parts(logits_eval[val_mask], y_all[val_mask], epoch=epoch)
            
            train_ce_loss = train_loss_parts['ce_loss'].item()
            val_ce_loss = val_loss_parts['ce_loss'].item()
            val_nll = val_loss_parts['nll'].item()
            val_objective = val_loss_parts['objective'].item()
            
            va, _, _, _ = evaluate_logits(logits_eval, val_mask)

        current_score = checkpoint_score(va, val_ce_loss, train_ce_loss)
        scheduler.step(current_score)
        
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_gap = max(0.0, float(val_ce_loss) - float(train_ce_loss))

        # Checkpoint: best_metric_state
        if current_score > best_score + min_delta:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_payload = {
                'epoch': epoch,
                'score': current_score,
                'train_loss': train_ce_loss,
                'val_loss': val_ce_loss,
                'metrics': va,
            }
            no_improve = 0
        else:
            no_improve += 1

        # Checkpoint: best_loss_state
        checkpoint_value = val_nll if config['LOSS_PROTOCOL'] == 'benchmark_ce' else val_objective
        if checkpoint_value < best_loss_value - min_delta:
            best_loss_value = float(checkpoint_value)
            best_loss_state = copy.deepcopy(model.state_dict())
            best_loss_epoch = int(epoch)
            best_loss_payload = {
                'epoch': epoch,
                'val_ce_loss': val_ce_loss,
                'train_ce_loss': train_ce_loss,
            }

        # Checkpoint: best_balanced_state
        best_metric_val_acc = best_payload.get('metrics', {}).get('Accuracy', 0.0) if best_payload else 0.0
        accuracy_floor = best_metric_val_acc - float(config.get('balanced_accuracy_floor_drop', 0.003))
        balanced_eligible = bool(va['Accuracy'] >= accuracy_floor)
        if balanced_eligible and (
            val_ce_loss < best_balanced_loss - min_delta
            or (abs(val_ce_loss - best_balanced_loss) <= min_delta and current_score > best_balanced_metric_score + min_delta)
        ):
            best_balanced_loss = float(val_ce_loss)
            best_balanced_metric_score = float(current_score)
            best_balanced_state = copy.deepcopy(model.state_dict())
            best_balanced_epoch = int(epoch)
            best_balanced_payload = {
                'epoch': epoch,
                'val_ce_loss': val_ce_loss,
                'train_ce_loss': train_ce_loss,
            }

        print(f"Epoch {epoch:02d} | ValCE: {val_ce_loss:.4f} | Gap: {val_loss_gap:.4f} | ValAcc: {va['Accuracy']:.4f} | Score: {current_score:.4f} | LR: {current_lr:.2e}")

        if epoch >= min_epochs and no_improve >= config['patience']:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    # Select final checkpoint
    if config['mode'] == 'balanced_loss' and best_balanced_state is not None:
        selected_checkpoint_tag = 'balanced_loss'
        selected_checkpoint_epoch = best_balanced_epoch
        selected_checkpoint_payload = best_balanced_payload
        model.load_state_dict(best_balanced_state)
    elif best_state is not None:
        selected_checkpoint_tag = 'metric'
        selected_checkpoint_epoch = best_epoch
        selected_checkpoint_payload = best_payload
        model.load_state_dict(best_state)
    elif best_loss_state is not None:
        selected_checkpoint_tag = 'val_NLL'
        selected_checkpoint_epoch = best_loss_epoch
        selected_checkpoint_payload = best_loss_payload
        model.load_state_dict(best_loss_state)
        
    print(f"Selected checkpoint: {selected_checkpoint_tag} at epoch {selected_checkpoint_epoch}")

print("\nAll tests completed successfully!")
