import json
import os

path = 'e:/thesis/notebooks/hhgnn-ccr.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Identify key cells by their content.
loss_cell_idx = -1
viz_cell_idx = -1
data_prep_idx = -1
train_cell_idx = -1
test_cell_idx = -1
metrics_cell_idx = -1
model_cell_idx = -1

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'HybridFocalOrdinalLoss' in src and 'FocalLoss' in src:
        loss_cell_idx = i
    if 'Visualize HHGNN Fuzzy Graph Structure' in src:
        viz_cell_idx = i
    if 'Data Preparation V2: Static Representation + Z-Scores' in src:
        data_prep_idx = i
    if 'Training Loop V2' in src:
        train_cell_idx = i
    if 'Test Set Evaluation V2' in src:
        test_cell_idx = i
    if 'model =' in src and 'HHGNNFuzzyV2CreditRating' in src:
        model_cell_idx = i

# --- Reordering Data Prep and Viz cells ---
if data_prep_idx > viz_cell_idx:
    data_prep_cell = nb['cells'].pop(data_prep_idx)
    # Re-find viz index after popping
    viz_cell_idx = -1
    for i, cell in enumerate(nb['cells']):
        if ''.join(cell.get('source', [])).find('Visualize HHGNN Fuzzy Graph Structure') != -1:
            viz_cell_idx = i
            break
    nb['cells'].insert(viz_cell_idx, data_prep_cell)
    # Refresh all indices
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        if 'HybridFocalOrdinalLoss' in src: loss_cell_idx = i
        if 'Visualize HHGNN Fuzzy Graph Structure' in src: viz_cell_idx = i
        if 'Data Preparation V2' in src: data_prep_idx = i
        if 'Training Loop V2' in src: train_cell_idx = i
        if 'Test Set Evaluation V2' in src: test_cell_idx = i
        if 'model =' in src and 'HHGNNFuzzyV2CreditRating' in src: model_cell_idx = i

# --- Fix Viz Cell ---
viz_src = nb['cells'][viz_cell_idx]['source']
new_viz_src = []
for line in viz_src:
    if 'sample_z_dummy   = np.zeros_like(sample_company)' in line or 'dummy z-scores' in line:
        new_viz_src.append("sample_z_all   = Z_all[0]   # using actual z-scores for visualization\n")
    elif 'build_fuzzy_feature_graph_v2(' in line and 'sample_z_dummy' in line:
        # It's split across lines or on one line
        new_viz_src.extend(line.replace('sample_z_dummy', 'sample_z_all'))
    elif 'FINANCIAL_FEATURES\n' in line and not line.startswith('#'):
        # Fix the call args if it is split
        new_viz_src.append(line.replace('sample_z_dummy', 'sample_z_all'))
    elif 'sample_company, sample_z_dummy, FINANCIAL_FEATURES' in line:
        new_viz_src.append(line.replace('sample_z_dummy', 'sample_z_all'))
    else:
        new_viz_src.append(line)
nb['cells'][viz_cell_idx]['source'] = new_viz_src

# --- Rewrite Loss Cell ---
new_loss_cell = """# ============================================================
# Focal Loss + Class-0 Focus Enhancements
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    \"\"\"Multiclass focal loss with optional class balancing and label smoothing.\"\"\"

    def __init__(
        self,
        gamma: float = 2.0,
        class_weight=None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.current_tau = 0.0 # for compatibility with train log
        self.current_recall_weight = 0.0

    def set_training_progress(self, epoch: int, max_epochs: int):
        pass # Compatibility

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss

        if self.class_weight is not None:
            focal = focal * self.class_weight[targets]

        if self.reduction == 'sum':
            return focal.sum()
        return focal.mean()

def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1).long()

def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1).clamp(min=1e-8)
    return probs / probs.sum(dim=1, keepdim=True)

def predict_with_class0_threshold(
    logits: torch.Tensor, class0_threshold: float = 0.40
) -> torch.Tensor:
    probs = logits_to_probs(logits)
    if probs.shape[1] <= 1:
        return torch.zeros((probs.shape[0],), dtype=torch.long, device=logits.device)
    non0_pred = torch.argmax(probs[:, 1:], dim=1) + 1
    pred = torch.where(probs[:, 0] >= class0_threshold, torch.zeros_like(non0_pred), non0_pred)
    return pred.long()

def corn_predict(logits: torch.Tensor, class0_threshold: float = None) -> torch.Tensor:
    if class0_threshold is None:
        return logits_to_pred(logits)
    return predict_with_class0_threshold(logits, class0_threshold=class0_threshold)

def corn_to_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits_to_probs(logits)

def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred.astype(int) - y_true.astype(int)) <= 1))

def find_best_class0_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    beta: float = 1.5,
    t_min: float = 0.20,
    t_max: float = 0.70,
    n_steps: int = 21,
    default_t: float = 0.40,
 ) -> tuple[float, float]:
    from sklearn.metrics import fbeta_score
    y_true_c0 = (y_true.astype(int) == 0).astype(int)
    if probs.shape[1] <= 1:
        return default_t, 0.0
    best_t, best_score = default_t, -1.0
    for t in np.linspace(t_min, t_max, n_steps):
        pred_non0 = probs[:, 1:].argmax(axis=1) + 1
        y_pred = np.where(probs[:, 0] >= float(t), 0, pred_non0)
        y_pred_c0 = (y_pred == 0).astype(int)
        score = fbeta_score(y_true_c0, y_pred_c0, beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_t = float(score), float(t)
    return best_t, best_score

def build_effective_num_weights(class_counts: np.ndarray, beta: float = 0.995) -> np.ndarray:
    class_counts = np.asarray(class_counts, dtype=np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    effective_num = 1.0 - np.power(beta, class_counts)
    w = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    return w / w.mean()

all_train_labels = y_all[train_idx].astype(int)
class_counts = np.bincount(all_train_labels, minlength=n_classes).astype(float)
class_weights = build_effective_num_weights(class_counts, beta=0.995)

CLASS0_WEIGHT_BOOST = 2.0
class_weights[0] *= CLASS0_WEIGHT_BOOST
class_weights = np.clip(class_weights, *np.percentile(class_weights, [5, 95]))
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

FOCAL_GAMMA = 2.2
FOCAL_LABEL_SMOOTH = 0.02
C0_THRESHOLD_INIT = 0.40
CALIB_BETA = 1.5
CALIB_START_EPOCH = 12
CALIB_EVERY = 2
THRESHOLD_EMA = 0.30

criterion = FocalLoss(
    gamma=FOCAL_GAMMA,
    class_weight=class_weights_tensor,
    label_smoothing=FOCAL_LABEL_SMOOTH,
)

print(f"Loss Function: FocalLoss")
print(f"Class counts: {class_counts.astype(int).tolist()}")
print(f"Class-0 weight boost: x{CLASS0_WEIGHT_BOOST}")
print(f"Label smoothing: {FOCAL_LABEL_SMOOTH:.3f}")
"""
# Make sure to keep formatting as a list of lines with newlines
nb['cells'][loss_cell_idx]['source'] = [line + '\n' for line in new_loss_cell.strip().split('\n')]

# --- Fix Model Cell ---
# Remove unneeded outputs and architecture parts for dual-head if needed, but the prompt says 
# "loại bỏ corn sử dụng loss function focal loss" which doesn't explicitly restrict architecture 
# changes other than making it use Focal Loss. It's safer to just let the model output what it wants 
# but only use `logits_main` for the loss calculation. Wait, `lce` and `lc` are both outputted. 

# Let's fix loop and metric calculation to pass just single logits instead of ordinal.
# --- Rewrite Train Cell ---
train_src = nb['cells'][train_cell_idx]['source']
for j, line in enumerate(train_src):
    if 'lc, lce = model(' in line:
        train_src[j] = line.replace('lc, lce = model(', 'logits_main, logits_aux = model(')
    if 'loss = criterion(lc, lce' in line:
        train_src[j] = line.replace('loss = criterion(lc, lce, batch.y, fuzzy_weights=fw_batch)', 'loss = criterion(logits_main, batch.y)')
    if 'loss = criterion(lc, lce, batch.y, fuzzy_weights=None)' in line:
        train_src[j] = line.replace('loss = criterion(lc, lce, batch.y, fuzzy_weights=None)', 'loss = criterion(logits_main, batch.y)')
    # Fix metric arguments inside loop
    if 'compute_cls_metrics_v2(' in line: pass # keep arguments to find it below
    if 'all_lcorn.append(lc.detach())' in line:
        train_src[j] = line.replace('all_lcorn.append(lc.detach())', 'all_lcorn.append(logits_main.detach())')
    if 'vl_lcorn.append(lc)' in line:
        train_src[j] = line.replace('vl_lcorn.append(lc)', 'vl_lcorn.append(logits_main)')
    if 'def compute_cls_metrics_v2(' in line: pass

train_src_str = "".join(train_src)
train_src_str = train_src_str.replace('all_lc, all_lce2', 'all_lc, all_lc')
train_src_str = train_src_str.replace('vl_lc, vl_lce2', 'vl_lc, vl_lc')

# Find the definition of compute_cls_metrics_v2 and fix it to print 4 requested metrics explicitly.
# Although the notebook already returns acc, f1_macro, f1_w, auc, mae, qwk
# We simply ensure it calculates what the user requested. We'll leave the function logic as is since it
# exactly matches the request. "đánh giá tập test đủ 4 metric accuracy, F1 macro, qwk, auc"

nb['cells'][train_cell_idx]['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in train_src_str.split('\n')[:-1]]

# --- Fix test evaluation cell ---
test_src = nb['cells'][test_cell_idx]['source']
test_src_str = "".join(test_src)
test_src_str = test_src_str.replace('lc, lce = model(', 'logits_main, logits_aux = model(')
test_src_str = test_src_str.replace('preds = corn_predict(lc', 'preds = corn_predict(logits_main')
test_src_str = test_src_str.replace('test_lc_all.append(lc.cpu())', 'test_lc_all.append(logits_main.cpu())')
test_src_str = test_src_str.replace('test_lceil.append(lce.cpu())', '# removed')

# The prompt demands: "đánh giá tập test đủ 4 metric accuracy, F1 macro, qwk, auc"
# The print currently has those, we just ensure the text looks correct.
if 'AUC-ROC' not in test_src_str:
    test_src_str = test_src_str.replace("print(f'  F1-weighted:", "print(f'  AUC-ROC (OvR):      {auc_ovr:.4f}')\nprint(f'  F1-weighted:")

nb['cells'][test_cell_idx]['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in test_src_str.split('\n')[:-1]]

# The smoke test cell also needs fixing
smoke_test_idx = -1
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Smoke Test' in src:
        smoke_test_idx = i
if smoke_test_idx != -1:
    smoke_src = "".join(nb['cells'][smoke_test_idx]['source'])
    smoke_src = smoke_src.replace('criterion(logits_main, logits_aux, batch.y, fuzzy_weights=None)', 'criterion(logits_main, batch.y)')
    nb['cells'][smoke_test_idx]['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in smoke_src.split('\n')[:-1]]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
