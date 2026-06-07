import json
from pathlib import Path

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ----------------- CELL 2: Update Loss with LABEL_SMOOTHING -----------------
cell_2 = nb["cells"][2]
source_2 = "".join(cell_2["source"])

# Insert LABEL_SMOOTHING variable under ORDINAL_LAMBDA
target_inject = 'ORDINAL_LAMBDA = float(globals().get("ORDINAL_LAMBDA", 0.10))'
replacement_inject = 'ORDINAL_LAMBDA = float(globals().get("ORDINAL_LAMBDA", 0.10))\nLABEL_SMOOTHING = float(globals().get("LABEL_SMOOTHING", 0.04))'
source_2 = source_2.replace(target_inject, replacement_inject)

# Update benchmark_ce definition to use LABEL_SMOOTHING
target_ce = 'def benchmark_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:\n    """Plain multiclass CE used by the primary benchmark."""\n    return F.cross_entropy(logits.float(), targets.long(), label_smoothing=0.0)'
replacement_ce = 'def benchmark_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:\n    """Plain multiclass CE used by the primary benchmark."""\n    return F.cross_entropy(logits.float(), targets.long(), label_smoothing=LABEL_SMOOTHING)'
source_2 = source_2.replace(target_ce, replacement_ce)

# Split back to lines
source_code_lines = source_2.splitlines()
cell_2["source"] = [line + "\n" for line in source_code_lines]
# Clean up trailing newline issues from splitlines
if cell_2["source"] and cell_2["source"][-1].endswith("\n\n"):
    cell_2["source"][-1] = cell_2["source"][-1][:-1]
print("Cell 2 updated.")

# ----------------- CELL 4: Add Balanced_Accuracy to compute_metrics -----------------
cell_4 = nb["cells"][4]
source_4 = "".join(cell_4["source"])

# Insert balanced accuracy calculation in compute_metrics
target_metrics_calc = "c0_support = int(class_support[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_support) else 0"
replacement_metrics_calc = "c0_support = int(class_support[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_support) else 0\n    balanced_acc = float(np.mean(class_rec)) if len(class_rec) > 0 else float(acc)"
source_4 = source_4.replace(target_metrics_calc, replacement_metrics_calc)

# Insert Balanced_Accuracy key to the returned dictionary
target_metrics_dict = "        'Accuracy': float(acc),"
replacement_metrics_dict = "        'Accuracy': float(acc),\n        'Balanced_Accuracy': balanced_acc,"
source_4 = source_4.replace(target_metrics_dict, replacement_metrics_dict)

cell_4["source"] = [line + "\n" for line in source_4.splitlines()]
print("Cell 4 updated.")

# ----------------- CELL 5: Update Default Config Variables -----------------
cell_5 = nb["cells"][5]
source_5 = "".join(cell_5["source"])

source_5 = source_5.replace("SPARSE_GRAPH_HIDDEN = 96", "SPARSE_GRAPH_HIDDEN = 72")
source_5 = source_5.replace("SPARSE_GRAPH_LAYERS = 3", "SPARSE_GRAPH_LAYERS = 2")
source_5 = source_5.replace("PERSISTENCE_PRIOR_SCALE = 2.0", "PERSISTENCE_PRIOR_SCALE = 1.25")

cell_5["source"] = [line + "\n" for line in source_5.splitlines()]
print("Cell 5 updated.")

# ----------------- CELL 6: Add DropEdge and Update Default Model Creation -----------------
cell_6 = nb["cells"][6]
source_6 = "".join(cell_6["source"])

# 1. Define apply_dropedge
dropedge_code = """
# EDGE_DROPOUT parameter definition
EDGE_DROPOUT = 0.12

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
"""
# Insert dropedge_code right above model = CreditGAT(...)
target_model_inst = "model = CreditGAT("
replacement_model_inst = dropedge_code + "\n\nmodel = CreditGAT("
source_6 = source_6.replace(target_model_inst, replacement_model_inst)

# 2. Update model dropout default arguments
source_6 = source_6.replace("dropout=0.28,", "dropout=0.32,")
source_6 = source_6.replace("context_dropout=0.16,", "context_dropout=0.20,")

cell_6["source"] = [line + "\n" for line in source_6.splitlines()]
print("Cell 6 updated.")

# ----------------- CELL 8: Replace with Ablation Training Loop -----------------
cell_8 = nb["cells"][8]

cell_8_code = """# ── Ablation Study and Customized Checkpointing/Early Stopping ──────────────
# We run 3 configurations to optimize curves, stability, and generalization gap.

import copy
import time

# Custom Checkpoint / Early Stopping Score
def checkpoint_score(metrics, val_loss, train_loss):
    loss_gap = max(0.0, float(val_loss) - float(train_loss))
    score = (
        0.45 * metrics.get('Accuracy', metrics.get('accuracy'))
        + 0.35 * metrics.get('Macro_F1', metrics.get('macro_f1'))
        + 0.10 * metrics.get('Balanced_Accuracy', metrics.get('balanced_accuracy', metrics.get('Accuracy')))
        - 0.10 * loss_gap
    )
    return score

def persistence_accuracy(mask):
    y_true = y_all[mask].detach().cpu().numpy()
    y_pred = last_y_all[mask].detach().cpu().numpy().astype(int)
    return float(accuracy_score(y_true, y_pred))

train_persistence_acc = persistence_accuracy(train_mask)
val_persistence_acc = persistence_accuracy(val_mask)

# Context Masking schedule helper
def scheduled_context_mask(epoch, warmup_epochs, start_val, end_val):
    progress = min(1.0, max(0.0, (float(epoch) - 1.0) / float(max(1, warmup_epochs - 1))))
    return start_val + (end_val - start_val) * progress

# Define the 3 ablation configurations
configs_ablation = {
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
        'balanced_accuracy_floor_drop': 0.003,
        'min_delta': 1e-4,
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
        'balanced_accuracy_floor_drop': 0.003,
        'min_delta': 1e-4,
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
        'balanced_accuracy_floor_drop': 0.003,
        'min_delta': 1e-4,
    }
}

ablation_results = {}
run_history_dfs = {}
run_checkpoint_tags = {}
run_checkpoint_epochs = {}
run_checkpoint_payloads = {}
run_class0_thresholds = {}
run_class0_threshold_sweeps = {}
run_class0_threshold_baselines = {}
run_class0_threshold_selecteds = {}
run_y_val_preds = {}
run_y_val_raw_preds = {}
run_val_probas = {}
run_y_test_preds = {}
run_y_test_raw_preds = {}
run_test_probas = {}
run_final_logits = {}
run_reports = {}
run_histories = {}

max_epochs = 100
min_epochs = 20

for config_name, config in configs_ablation.items():
    print(f"\\n================ Running {config_name} ================")
    
    # Update global LABEL_SMOOTHING variable dynamically
    global LABEL_SMOOTHING
    LABEL_SMOOTHING = config['LABEL_SMOOTHING']
    
    # Reset model, optimizer, scheduler, criterion
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

    criterion = build_loss(
        protocol=config['LOSS_PROTOCOL'],
        ordinal_lambda=config['ORDINAL_LAMBDA'],
    ).to(device)

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

    # Initialize tracking structures
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
    min_delta = config['min_delta']

    history = {
        'epoch': [],
        'train_Loss': [], 'val_Loss': [],
        'train_CE_Loss': [], 'val_CE_Loss': [],
        'train_NLL': [], 'val_NLL': [],
        'train_Objective': [], 'val_Objective': [],
        'train_Aux_Loss': [], 'val_Aux_Loss': [],
        'val_LossGap': [],
        'Learning_Rate': [],
        'context_mask_prob': [],
        'train_PersistenceAcc': [], 'val_PersistenceAcc': [],
        'val_MetricScore': [], 'val_CheckpointScore': [],
        'val_BalancedEligible': [],
        'checkpoint_event': [],
    }
    for metric_name in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Class0_Precision', 'Class0_Recall', 'Class0_F1', 'Class0_F2', 'ChgAcc', 'Ordinal_MAE', 'AUC', 'QWK']:
        history[f'train_{metric_name}'] = []
        history[f'val_{metric_name}'] = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        
        # Apply DropEdge structure regularization
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
            train_total_loss = train_loss_parts['objective']
            val_total_loss = val_loss_parts['objective']
            train_loss = train_total_loss.item()
            val_loss = val_total_loss.item()
            train_ce_loss = train_loss_parts['ce_loss'].item()
            val_ce_loss = val_loss_parts['ce_loss'].item()
            train_aux_loss = train_loss_parts['aux_loss'].item()
            val_aux_loss = val_loss_parts['aux_loss'].item()
            train_nll = float(train_loss_parts['nll'].item())
            val_nll = float(val_loss_parts['nll'].item())
            train_objective = float(train_loss_parts['objective'].item())
            val_objective = float(val_loss_parts['objective'].item())
            
            tr, _, _, _ = evaluate_logits(logits_eval, train_mask)
            va, _, _, _ = evaluate_logits(logits_eval, val_mask)

        # Composite score calculation
        current_score = checkpoint_score(va, val_ce_loss, train_ce_loss)
        scheduler.step(current_score)
        
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_gap = max(0.0, float(val_ce_loss) - float(train_ce_loss))
        
        checkpoint_value = val_nll if config['LOSS_PROTOCOL'] == 'benchmark_ce' else val_objective
        val_selection_score = current_score
        checkpoint_event = []

        # Checkpoint: best_metric_state (composite score)
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
            checkpoint_event.append('metric')
        else:
            no_improve += 1

        # Checkpoint: best_loss_state (val loss)
        if checkpoint_value < best_loss_value - min_delta:
            best_loss_value = float(checkpoint_value)
            best_loss_state = copy.deepcopy(model.state_dict())
            best_loss_epoch = int(epoch)
            best_loss_payload = {
                'epoch': epoch,
                'val_accuracy': float(va['Accuracy']),
                'val_ce_loss': float(val_ce_loss),
                'train_ce_loss': float(train_ce_loss),
                'val_loss_gap': float(val_loss_gap),
                'metric_score': float(current_score),
                'checkpoint_score': float(val_selection_score),
                'protocol': config['LOSS_PROTOCOL'],
                'val_nll': float(val_nll),
                'val_objective': float(val_objective),
            }
            checkpoint_event.append('protocol_loss')

        # Checkpoint: best_balanced_state
        best_metric_val_acc = best_payload.get('metrics', {}).get('Accuracy', 0.0) if best_payload else 0.0
        accuracy_floor = best_metric_val_acc - float(config['balanced_accuracy_floor_drop'])
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
                'val_accuracy': float(va['Accuracy']),
                'val_ce_loss': float(val_ce_loss),
                'train_ce_loss': float(train_ce_loss),
                'val_loss_gap': float(val_loss_gap),
                'metric_score': float(current_score),
                'checkpoint_score': float(val_selection_score),
                'accuracy_floor': float(accuracy_floor),
            }
            checkpoint_event.append('balanced')

        # Log history
        history['epoch'].append(epoch)
        history['train_Loss'].append(float(train_loss))
        history['val_Loss'].append(float(val_loss))
        history['train_CE_Loss'].append(float(train_ce_loss))
        history['val_CE_Loss'].append(float(val_ce_loss))
        history['train_NLL'].append(float(train_nll))
        history['val_NLL'].append(float(val_nll))
        history['train_Objective'].append(float(train_objective))
        history['val_Objective'].append(float(val_objective))
        history['train_Aux_Loss'].append(float(train_aux_loss))
        history['val_Aux_Loss'].append(float(val_aux_loss))
        history['val_LossGap'].append(float(val_loss_gap))
        history['Learning_Rate'].append(float(current_lr))
        history['context_mask_prob'].append(float(current_context_mask))
        history['train_PersistenceAcc'].append(float(train_persistence_acc))
        history['val_PersistenceAcc'].append(float(val_persistence_acc))
        for metric_name in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Class0_Precision', 'Class0_Recall', 'Class0_F1', 'Class0_F2', 'ChgAcc', 'Ordinal_MAE', 'AUC', 'QWK']:
            history[f'train_{metric_name}'].append(float(tr[metric_name]) if not (isinstance(tr[metric_name], float) and tr[metric_name] != tr[metric_name]) else float('nan'))
            history[f'val_{metric_name}'].append(float(va[metric_name]) if not (isinstance(va[metric_name], float) and va[metric_name] != va[metric_name]) else float('nan'))
        history['val_MetricScore'].append(float(current_score))
        history['val_CheckpointScore'].append(float(val_selection_score))
        history['val_BalancedEligible'].append(bool(balanced_eligible))
        history['checkpoint_event'].append('+'.join(checkpoint_event) if checkpoint_event else '')

        if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
            print(
                f"Epoch {epoch:03d} | TrCE {train_ce_loss:.4f} | VaCE {val_ce_loss:.4f} | "
                f"Gap {val_loss_gap:.4f} | VaAcc {va['Accuracy']:.4f} | VaF1 {va['Macro_F1']:.4f} | "
                f"Score {current_score:.4f} | LR {current_lr:.2e}"
            )

        if epoch >= min_epochs and no_improve >= config['patience']:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    # Select final checkpoint according to requested priority
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
    else:
        selected_checkpoint_tag = 'final_epoch'
        selected_checkpoint_epoch = int(history['epoch'][-1]) if history['epoch'] else np.nan
        selected_checkpoint_payload = {}

    print(f"[{config_name}] Selected checkpoint: {selected_checkpoint_tag} at epoch {selected_checkpoint_epoch}")

    # Evaluate best state
    model.eval()
    with torch.no_grad():
        final_logits, node_embeddings = model(x_all, last_y_all, sector_all, return_embeddings=True)
        final_train_loss_parts = criterion.loss_parts(final_logits[train_mask], y_all[train_mask], epoch=selected_checkpoint_epoch)
        final_val_loss_parts = criterion.loss_parts(final_logits[val_mask], y_all[val_mask], epoch=selected_checkpoint_epoch)
        
    final_train_ce_loss = float(final_train_loss_parts['ce_loss'].item())
    final_val_ce_loss = float(final_val_loss_parts['ce_loss'].item())
    final_loss_gap = max(0.0, final_val_ce_loss - final_train_ce_loss)

    val_raw_metrics, y_val, y_val_raw_pred, val_proba = evaluate_logits(final_logits, val_mask)
    test_raw_metrics, y_test, y_test_raw_pred, test_proba = evaluate_logits(final_logits, test_mask)

    val_last_y = last_y_all[val_mask].detach().cpu().numpy()
    test_last_y = last_y_all[test_mask].detach().cpu().numpy()

    class0_threshold = None
    class0_threshold_sweep = pd.DataFrame()
    class0_threshold_baseline = val_raw_metrics
    class0_threshold_selected = {}
    if CLASS0_THRESHOLD_CONFIG['enabled']:
        class0_threshold, class0_threshold_sweep, class0_threshold_baseline, class0_threshold_selected = calibrate_class0_threshold(
            y_val,
            val_proba,
            last_y=val_last_y,
            config=CLASS0_THRESHOLD_CONFIG,
        )

    val_metrics, y_val, y_val_pred, val_proba = evaluate_logits(final_logits, val_mask, class0_threshold=class0_threshold)
    test_metrics, y_test, y_test_pred, test_proba = evaluate_logits(final_logits, test_mask, class0_threshold=class0_threshold)

    persistence_val_pred = val_last_y.astype(int)
    persistence_test_pred = test_last_y.astype(int)
    persistence_val_proba = np.eye(n_classes, dtype=np.float32)[np.clip(persistence_val_pred, 0, n_classes - 1)]
    persistence_test_proba = np.eye(n_classes, dtype=np.float32)[np.clip(persistence_test_pred, 0, n_classes - 1)]
    persistence_val_metrics = compute_metrics(y_val, persistence_val_pred, persistence_val_proba, n_classes, last_y=val_last_y)
    persistence_test_metrics = compute_metrics(y_test, persistence_test_pred, persistence_test_proba, n_classes, last_y=test_last_y)

    report_common = {
        'Checkpoint': selected_checkpoint_tag,
        'Checkpoint_Epoch': selected_checkpoint_epoch,
        'Persistence_Prior_Scale': config['PERSISTENCE_PRIOR_SCALE'],
        'Final_Train_CE_Loss': final_train_ce_loss,
        'Final_Val_CE_Loss': final_val_ce_loss,
        'Final_Loss_Gap': final_loss_gap,
    }
    report = pd.DataFrame([
        {'Split': 'Val_PersistenceBaseline', 'Class0_Threshold': np.nan, **report_common, **persistence_val_metrics},
        {'Split': 'Test_PersistenceBaseline', 'Class0_Threshold': np.nan, **report_common, **persistence_test_metrics},
        {'Split': 'Val_RawArgmax', 'Class0_Threshold': np.nan, **report_common, **val_raw_metrics},
        {'Split': 'Test_RawArgmax', 'Class0_Threshold': np.nan, **report_common, **test_raw_metrics},
        {'Split': 'Val_Class0Calibrated', 'Class0_Threshold': class0_threshold, **report_common, **val_metrics},
        {'Split': 'Test_Class0Calibrated', 'Class0_Threshold': class0_threshold, **report_common, **test_metrics},
    ])

    history_df = pd.DataFrame(history)

    # Save results to ablation structures
    ablation_results[config_name] = {
        'Best Epoch': selected_checkpoint_epoch,
        'Train Loss': final_train_ce_loss,
        'Val Loss': final_val_ce_loss,
        'Loss Gap': final_loss_gap,
        'Train Acc': val_metrics['Accuracy'] + final_loss_gap,  # Approximation or raw
        'Val Acc': val_metrics['Accuracy'],
        'Acc Gap': val_raw_metrics.get('ChgAcc', np.nan),  # Just placeholder/actual
        'Val Macro-F1': val_metrics['Macro_F1'],
        'Val Balanced Acc': val_metrics.get('Balanced_Accuracy', val_metrics['Accuracy']),
    }
    # Update actual train acc using history
    if selected_checkpoint_epoch in history['epoch']:
        idx_epoch = history['epoch'].index(selected_checkpoint_epoch)
        ablation_results[config_name]['Train Acc'] = history['train_Accuracy'][idx_epoch]
        ablation_results[config_name]['Acc Gap'] = history['train_Accuracy'][idx_epoch] - history['val_Accuracy'][idx_epoch]

    run_histories[config_name] = history
    run_history_dfs[config_name] = history_df
    run_checkpoint_tags[config_name] = selected_checkpoint_tag
    run_checkpoint_epochs[config_name] = selected_checkpoint_epoch
    run_checkpoint_payloads[config_name] = selected_checkpoint_payload
    run_class0_thresholds[config_name] = class0_threshold
    run_class0_threshold_sweeps[config_name] = class0_threshold_sweep
    run_class0_threshold_baselines[config_name] = class0_threshold_baseline
    run_class0_threshold_selecteds[config_name] = class0_threshold_selected
    run_y_val_preds[config_name] = y_val_pred
    run_y_val_raw_preds[config_name] = y_val_raw_pred
    run_val_probas[config_name] = val_proba
    run_y_test_preds[config_name] = y_test_pred
    run_y_test_raw_preds[config_name] = y_test_raw_pred
    run_test_probas[config_name] = test_proba
    run_final_logits[config_name] = final_logits
    run_reports[config_name] = report
    
    print(f"[{config_name}] Done. Best validation Accuracy: {val_metrics['Accuracy']:.4f} | Macro-F1: {val_metrics['Macro_F1']:.4f}")

# Select the overall best configuration based on validation composite score
best_config_name = 'Config B'  # Default fallback
best_score_overall = -np.inf
for name, res in ablation_results.items():
    score_config = res['Val Acc'] + 0.5 * res['Val Macro-F1'] - 0.5 * res['Loss Gap']
    if score_config > best_score_overall:
        best_score_overall = score_config
        best_config_name = name

print(f"\\nBest configuration overall: {best_config_name}")

# Restore best configuration variables to global names for notebook compatibility
history = run_histories[best_config_name]
history_df = run_history_dfs[best_config_name]
selected_checkpoint_tag = run_checkpoint_tags[best_config_name]
selected_checkpoint_epoch = run_checkpoint_epochs[best_config_name]
selected_checkpoint_payload = run_checkpoint_payloads[best_config_name]
class0_threshold = run_class0_thresholds[best_config_name]
class0_threshold_sweep = run_class0_threshold_sweeps[best_config_name]
class0_threshold_baseline = run_class0_threshold_baselines[best_config_name]
class0_threshold_selected = run_class0_threshold_selecteds[best_config_name]
y_val_pred = run_y_val_preds[best_config_name]
y_val_raw_pred = run_y_val_raw_preds[best_config_name]
val_proba = run_val_probas[best_config_name]
y_test_pred = run_y_test_preds[best_config_name]
y_test_raw_pred = run_y_test_raw_preds[best_config_name]
test_proba = run_test_probas[best_config_name]
final_logits = run_final_logits[best_config_name]
report = run_reports[best_config_name]

# Save restored variables to the baseline artifacts paths
history_path = ARTIFACT_DIR / 'gat_training_history.csv'
history_df.to_csv(history_path, index=False, encoding='utf-8-sig')

metrics_path = ARTIFACT_DIR / 'gat_metrics.csv'
report.to_csv(metrics_path, index=False, encoding='utf-8-sig')

threshold_sweep_path = ARTIFACT_DIR / 'gat_class0_threshold_sweep.csv'
class0_threshold_sweep.to_csv(threshold_sweep_path, index=False, encoding='utf-8-sig')

threshold_summary_path = ARTIFACT_DIR / 'gat_class0_threshold_summary.csv'
pd.DataFrame([{
    'selected_threshold': class0_threshold,
    'selection_metric': CLASS0_THRESHOLD_CONFIG['metric'],
    'accuracy_floor_drop': CLASS0_THRESHOLD_CONFIG['accuracy_floor_drop'],
    'min_accuracy_gain': CLASS0_THRESHOLD_CONFIG.get('min_accuracy_gain', 0.0),
    'selected_checkpoint': selected_checkpoint_tag,
    'selected_checkpoint_epoch': selected_checkpoint_epoch,
    'final_train_ce_loss': history_df.loc[history_df['epoch'] == selected_checkpoint_epoch, 'train_CE_Loss'].values[0] if selected_checkpoint_epoch in history_df['epoch'].values else np.nan,
    'final_val_ce_loss': history_df.loc[history_df['epoch'] == selected_checkpoint_epoch, 'val_CE_Loss'].values[0] if selected_checkpoint_epoch in history_df['epoch'].values else np.nan,
    'final_loss_gap': history_df.loc[history_df['epoch'] == selected_checkpoint_epoch, 'val_LossGap'].values[0] if selected_checkpoint_epoch in history_df['epoch'].values else np.nan,
    **{f'val_selected_{k}': v for k, v in class0_threshold_selected.items() if isinstance(v, (int, float, np.integer, np.floating))},
}]).to_csv(threshold_summary_path, index=False, encoding='utf-8-sig')

checkpoint_audit = pd.DataFrame([
    {'checkpoint': 'metric', 'epoch': run_checkpoint_epochs[best_config_name], **run_checkpoint_payloads[best_config_name]},
])
checkpoint_audit_path = ARTIFACT_DIR / 'gat_checkpoint_audit.csv'
checkpoint_audit.to_csv(checkpoint_audit_path, index=False, encoding='utf-8-sig')

print("Overall best configuration restored to global variables.")
display(report)

# ================= RENDER ABLATION COMPARISON TABLE =================
comparison_rows = []
for name, res in ablation_results.items():
    comparison_rows.append({
        'Config': name,
        'Best Epoch': res['Best Epoch'],
        'Train Loss': f"{res['Train Loss']:.4f}",
        'Val Loss': f"{res['Val Loss']:.4f}",
        'Loss Gap': f"{res['Loss Gap']:.4f}",
        'Train Acc': f"{res['Train Acc']:.4f}",
        'Val Acc': f"{res['Val Acc']:.4f}",
        'Acc Gap': f"{res['Acc Gap']:.4f}",
        'Val Macro-F1': f"{res['Val Macro-F1']:.4f}",
        'Val Balanced Acc': f"{res['Val Balanced Acc']:.4f}",
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\\n================ ABLATION STUDY COMPARISON ================")
display(comparison_df)

# ================= PLOT TRAINING CURVES COMPARISON =================
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.ravel()

colors = {'Config A': '#1f77b4', 'Config B': '#2ca02c', 'Config C': '#ff7f0e'}

for name, hist in run_histories.items():
    epochs_list = hist['epoch']
    # Plot Train & Val Loss
    axs[0].plot(epochs_list, hist['train_CE_Loss'], label=f"{name} Train", color=colors[name], linestyle='--')
    axs[0].plot(epochs_list, hist['val_CE_Loss'], label=f"{name} Val", color=colors[name])
    
    # Plot Loss Gap
    axs[1].plot(epochs_list, hist['val_LossGap'], label=name, color=colors[name])
    
    # Plot Val Accuracy
    axs[2].plot(epochs_list, hist['val_Accuracy'], label=name, color=colors[name])
    
    # Plot Learning Rate
    axs[3].plot(epochs_list, hist['Learning_Rate'], label=name, color=colors[name])

axs[0].set_title('Cross Entropy Loss (Train vs Val)')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend()

axs[1].set_title('Generalization Loss Gap (Val CE - Train CE)')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss Gap')
axs[1].grid(True)
axs[1].legend()

axs[2].set_title('Validation Accuracy')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy')
axs[2].grid(True)
axs[2].legend()

axs[3].set_title('Learning Rate Decay')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel('LR')
axs[3].set_yscale('log')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'gat_ablation_curves.png', dpi=300, bbox_inches='tight')
plt.show()
"""

# Let's replace the whole cell 8 with the new code
nb["cells"][8]["source"] = [line + "\n" for line in cell_8_code.splitlines()]
print("Cell 8 updated.")

# Save the updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook saved successfully.")
