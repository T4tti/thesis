import json
from pathlib import Path

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ----------------- CELL 2: Update Loss defaults for Config B -----------------
cell_2 = nb["cells"][2]
source_2 = "".join(cell_2["source"])

# Update LOSS_PROTOCOL to benchmark_ce and ORDINAL_LAMBDA to 0.10 (matching Config B)
source_2 = source_2.replace(
    'LOSS_PROTOCOL = str(globals().get("LOSS_PROTOCOL", "benchmark_ce")).strip().lower()',
    'LOSS_PROTOCOL = "benchmark_ce"'
)
source_2 = source_2.replace(
    'ORDINAL_LAMBDA = float(globals().get("ORDINAL_LAMBDA", 0.10))',
    'ORDINAL_LAMBDA = 0.10'
)

source_code_lines = source_2.splitlines()
cell_2["source"] = [line + "\n" for line in source_code_lines]
print("Cell 2 updated.")

# ----------------- CELL 6: Update Model Instantiation & Optimizer -----------------
cell_6 = nb["cells"][6]
source_6 = "".join(cell_6["source"])

# Update optimizer and scheduler configuration in Cell 6 to match Config B
optimizer_target = """OPTIMIZER_CONFIG = {
    'lr': 1.0e-3,
    'weight_decay': 5e-5,
    'plateau_factor': 0.60,
    'plateau_patience': 8,
    'min_lr': 1e-5,
}"""
optimizer_replacement = """OPTIMIZER_CONFIG = {
    'lr': 6e-4,
    'weight_decay': 2e-4,
    'plateau_factor': 0.50,
    'plateau_patience': 5,
    'min_lr': 3e-5,
}"""
source_6 = source_6.replace(optimizer_target, optimizer_replacement)

# Update scheduler mode in Cell 6 to 'max' (since we track checkpoint_score)
scheduler_target = """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',"""
scheduler_replacement = """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',"""
source_6 = source_6.replace(scheduler_target, scheduler_replacement)

source_code_lines = source_6.splitlines()
cell_6["source"] = [line + "\n" for line in source_code_lines]
print("Cell 6 updated.")

# ----------------- CELL 8: Update to Single Run of Config B -----------------
cell_8 = nb["cells"][8]

cell_8_code = """history = {
    'epoch': [],
    'train_Loss': [], 'val_Loss': [],
    'train_CE_Loss': [], 'val_CE_Loss': [],
    'train_NLL': [], 'val_NLL': [],
    'train_Objective': [], 'val_Objective': [],
    'train_Aux_Loss': [], 'val_Aux_Loss': [],
    'val_LossGap': [],
    'train_Accuracy': [], 'val_Accuracy': [],
    'train_Macro_F1': [], 'val_Macro_F1': [],
    'train_Class0_Precision': [], 'val_Class0_Precision': [],
    'train_Class0_Recall': [], 'val_Class0_Recall': [],
    'train_Class0_F1': [], 'val_Class0_F1': [],
    'train_Class0_F2': [], 'val_Class0_F2': [],
    'train_ChgAcc': [], 'val_ChgAcc': [],
    'train_Ordinal_MAE': [], 'val_Ordinal_MAE': [],
    'train_AUC': [], 'val_AUC': [],
    'train_QWK': [], 'val_QWK': [],
    'train_PersistenceAcc': [], 'val_PersistenceAcc': [],
    'Learning_Rate': [],
    'context_mask_prob': [],
    'val_MetricScore': [], 'val_CheckpointScore': [],
    'val_BalancedEligible': [],
    'checkpoint_event': [],
}

# Config B parameters
CONTEXT_MASK_START = 0.25
CONTEXT_MASK_END = 0.08
CONTEXT_MASK_WARMUP_EPOCHS = 60

def scheduled_context_mask(epoch):
    progress = min(1.0, max(0.0, (float(epoch) - 1.0) / float(max(1, CONTEXT_MASK_WARMUP_EPOCHS - 1))))
    return CONTEXT_MASK_START + (CONTEXT_MASK_END - CONTEXT_MASK_START) * progress

CHECKPOINT_CONFIG = {
    'mode': 'balanced_loss',
    'balanced_accuracy_floor_drop': 0.003,
    'min_delta': 1e-4,
}

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

best_score = -np.inf
best_state = None
best_epoch = None
best_metric_val_acc = -np.inf
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

patience, no_improve = 15, 0
max_epochs = 100
min_epochs = 20
min_delta = float(CHECKPOINT_CONFIG.get('min_delta', 1e-4))

for epoch in range(1, max_epochs + 1):
    model.train()
    
    # Apply DropEdge graph regularization during training (Config B uses p=0.12)
    train_edge_index, train_edge_weight = apply_dropedge(
        edge_index,
        edge_weight,
        x_all.size(0),
        p=0.12
    )
    
    current_context_mask = scheduled_context_mask(epoch)
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

    # Step scheduler based on custom composite score
    current_score = checkpoint_score(va, val_ce_loss, train_ce_loss)
    scheduler.step(current_score)
    
    current_lr = optimizer.param_groups[0]['lr']
    val_loss_gap = max(0.0, float(val_ce_loss) - float(train_ce_loss))

    checkpoint_value = val_nll if LOSS_PROTOCOL == BENCHMARK_PROTOCOL else val_objective
    val_selection_score = current_score
    checkpoint_event = []

    # Checkpoint: best composite score
    if current_score > best_score + min_delta:
        best_score = current_score
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        best_metric_val_acc = float(va['Accuracy'])
        best_payload = {
            'val_accuracy': float(va['Accuracy']),
            'val_ce_loss': float(val_ce_loss),
            'train_ce_loss': float(train_ce_loss),
            'val_loss_gap': float(val_loss_gap),
            'metric_score': float(current_score),
            'checkpoint_score': float(val_selection_score),
        }
        checkpoint_event.append('metric')
        no_improve = 0
    else:
        no_improve += 1

    # Checkpoint: best val loss
    if checkpoint_value < best_loss_value - min_delta:
        best_loss_value = float(checkpoint_value)
        best_loss_state = copy.deepcopy(model.state_dict())
        best_loss_epoch = int(epoch)
        best_loss_payload = {
            'val_accuracy': float(va['Accuracy']),
            'val_ce_loss': float(val_ce_loss),
            'train_ce_loss': float(train_ce_loss),
            'val_loss_gap': float(val_loss_gap),
            'metric_score': float(current_score),
            'checkpoint_score': float(val_selection_score),
            'protocol': LOSS_PROTOCOL,
            'val_nll': float(val_nll),
            'val_objective': float(val_objective),
        }
        checkpoint_event.append('protocol_loss')

    # Checkpoint: best balanced loss
    accuracy_floor = best_metric_val_acc - float(CHECKPOINT_CONFIG['balanced_accuracy_floor_drop'])
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
            'val_accuracy': float(va['Accuracy']),
            'val_ce_loss': float(val_ce_loss),
            'train_ce_loss': float(train_ce_loss),
            'val_loss_gap': float(val_loss_gap),
            'metric_score': float(current_score),
            'checkpoint_score': float(val_selection_score),
            'accuracy_floor': float(accuracy_floor),
        }
        checkpoint_event.append('balanced')

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

    print(
        f"Epoch {epoch:03d} | TrLoss {train_loss:.4f} | VaLoss {val_loss:.4f} | "
        f"Gap {val_loss_gap:.4f} | TrCE {train_ce_loss:.4f} | VaCE {val_ce_loss:.4f} | "
        f"VaAcc {va['Accuracy']:.4f} | VaF1 {va['Macro_F1']:.4f} | "
        f"VaC0R {va['Class0_Recall']:.4f} | VaC0F2 {va['Class0_F2']:.4f} | "
        f"VaQWK {va['QWK']:.4f} | MetricScore {current_score:.4f} | "
        f"CkptScore {val_selection_score:.4f} | Balanced {balanced_eligible} | "
        f"CtxMask {current_context_mask:.2f} | LR {current_lr:.2e}"
    )

    if epoch >= min_epochs and no_improve >= patience:
        print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
        break

# Select and load checkpoint using Priority
balanced_floor = best_metric_val_acc - float(CHECKPOINT_CONFIG['balanced_accuracy_floor_drop'])
if CHECKPOINT_CONFIG['mode'] == 'balanced_loss' and best_balanced_state is not None:
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
    selected_checkpoint_tag = 'val_NLL' if LOSS_PROTOCOL == BENCHMARK_PROTOCOL else 'val_Objective'
    selected_checkpoint_epoch = best_loss_epoch
    selected_checkpoint_payload = best_loss_payload
    model.load_state_dict(best_loss_state)
else:
    selected_checkpoint_tag = 'final_epoch'
    selected_checkpoint_epoch = int(history['epoch'][-1]) if history['epoch'] else np.nan
    selected_checkpoint_payload = {}

checkpoint_audit = pd.DataFrame([
    {'checkpoint': 'metric', 'epoch': best_epoch, **best_payload},
    {'checkpoint': 'loss', 'epoch': best_loss_epoch, **best_loss_payload},
    {'checkpoint': 'balanced_loss', 'epoch': best_balanced_epoch, **best_balanced_payload},
    {'checkpoint': f'selected:{selected_checkpoint_tag}', 'epoch': selected_checkpoint_epoch, **selected_checkpoint_payload},
])
checkpoint_audit_path = ARTIFACT_DIR / 'gat_checkpoint_audit.csv'
checkpoint_audit.to_csv(checkpoint_audit_path, index=False, encoding='utf-8-sig')
print(f"Selected checkpoint: {selected_checkpoint_tag} at epoch {selected_checkpoint_epoch}")
print('Saved:', checkpoint_audit_path)

model.eval()
with torch.no_grad():
    final_logits, node_embeddings = model(x_all, last_y_all, sector_all, return_embeddings=True)
    final_train_loss_parts = criterion.loss_parts(final_logits[train_mask], y_all[train_mask], epoch=selected_checkpoint_epoch)
    final_val_loss_parts = criterion.loss_parts(final_logits[val_mask], y_all[val_mask], epoch=selected_checkpoint_epoch)
    final_train_total_loss = final_train_loss_parts['objective']
    final_val_total_loss = final_val_loss_parts['objective']

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

history_df = pd.DataFrame(history)
history_path = ARTIFACT_DIR / 'gat_training_history.csv'
history_df.to_csv(history_path, index=False, encoding='utf-8-sig')

persistence_val_pred = val_last_y.astype(int)
persistence_test_pred = test_last_y.astype(int)
persistence_val_proba = np.eye(n_classes, dtype=np.float32)[np.clip(persistence_val_pred, 0, n_classes - 1)]
persistence_test_proba = np.eye(n_classes, dtype=np.float32)[np.clip(persistence_test_pred, 0, n_classes - 1)]
persistence_val_metrics = compute_metrics(y_val, persistence_val_pred, persistence_val_proba, n_classes, last_y=val_last_y)
persistence_test_metrics = compute_metrics(y_test, persistence_test_pred, persistence_test_proba, n_classes, last_y=test_last_y)

report_common = {
    'Checkpoint': selected_checkpoint_tag,
    'Checkpoint_Epoch': selected_checkpoint_epoch,
    'Persistence_Prior_Scale': PERSISTENCE_PRIOR_SCALE,
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
display(report)

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
    'final_train_ce_loss': final_train_ce_loss,
    'final_val_ce_loss': final_val_ce_loss,
    'final_loss_gap': final_loss_gap,
    **{f'val_selected_{k}': v for k, v in class0_threshold_selected.items() if isinstance(v, (int, float, np.integer, np.floating))},
}]).to_csv(threshold_summary_path, index=False, encoding='utf-8-sig')

print('Selected class 0 threshold:', class0_threshold)
print('Final CE loss gap:', round(final_loss_gap, 6))
print('Saved:', metrics_path)
print('Saved:', history_path)
print('Saved:', threshold_sweep_path)
print('Saved:', threshold_summary_path)

# ================= PLOT TRAINING CURVES =================
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.ravel()

epochs_list = history['epoch']

# Plot Train & Val Loss
axs[0].plot(epochs_list, history['train_CE_Loss'], label="Train CE Loss", color='#1f77b4', linestyle='--')
axs[0].plot(epochs_list, history['val_CE_Loss'], label="Val CE Loss", color='#1f77b4')
axs[0].set_title('Cross Entropy Loss (Train vs Val)')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend()

# Plot Loss Gap
axs[1].plot(epochs_list, history['val_LossGap'], label="CE Loss Gap", color='#2ca02c')
axs[1].set_title('Generalization Loss Gap (Val CE - Train CE)')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss Gap')
axs[1].grid(True)
axs[1].legend()

# Plot Val Accuracy
axs[2].plot(epochs_list, history['val_Accuracy'], label="Val Accuracy", color='#ff7f0e')
axs[2].plot(epochs_list, history['val_Balanced_Accuracy'], label="Val Balanced Accuracy", color='#ff7f0e', linestyle=':')
axs[2].set_title('Validation Accuracy & Balanced Accuracy')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy')
axs[2].grid(True)
axs[2].legend()

# Plot Learning Rate
axs[3].plot(epochs_list, history['Learning_Rate'], label="Learning Rate", color='#d62728')
axs[3].set_title('Learning Rate Decay')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel('LR')
axs[3].set_yscale('log')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'gat_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
"""

# Replace cell 8
nb["cells"][8]["source"] = [line + "\n" for line in cell_8_code.splitlines()]
print("Cell 8 updated to Config B single run.")

# Save notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook saved successfully.")
