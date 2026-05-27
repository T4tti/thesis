class FocalOrdinalLoss(nn.Module):
    """Focal + ordinal regularization for ordered rating classes."""
    def __init__(self, n_classes, gamma=1.5, ordinal_alpha=0.04, label_smoothing=0.0):
        super().__init__()
        self.n_classes = int(n_classes)
        self.gamma = float(gamma)
        self.ordinal_alpha = float(ordinal_alpha)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        logits = logits.float()
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0.0 and self.n_classes > 1:
            smooth = self.label_smoothing
            smooth_targets = torch.full_like(
                log_probs,
                fill_value=smooth / max(1, self.n_classes - 1),
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smooth)
            pt = (probs * smooth_targets).sum(dim=1).clamp(min=1e-8, max=1.0 - 1e-8)
        else:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0 - 1e-8)

        focal_term = (-((1.0 - pt) ** self.gamma) * torch.log(pt)).mean()

        if self.ordinal_alpha <= 0.0:
            return focal_term

        class_positions = torch.arange(self.n_classes, device=logits.device, dtype=probs.dtype)
        expected_rank = (probs * class_positions.unsqueeze(0)).sum(dim=1)
        target_rank = targets.to(dtype=probs.dtype)
        denom = max(1.0, float(self.n_classes - 1))
        ordinal_term = ((expected_rank - target_rank) / denom).pow(2).mean()

        return focal_term + self.ordinal_alpha * ordinal_term


model = PatchTSTClassifier(
    n_features=len(MODEL_FEATURES),
    n_classes=n_classes,
    n_sectors=n_sectors,
    input_size=INPUT_SIZE,
    patch_len=4,
    stride=2,
    d_model=128,
    n_heads=4,
    n_layers=3,
    drop=0.3,
).to(device)

criterion_settings = {
    'focal_gamma': 1.5,
    'ordinal_alpha': 0.04,
    'label_smoothing': 0.1,
}
criterion = FocalOrdinalLoss(
    n_classes=n_classes,
    gamma=criterion_settings['focal_gamma'],
    ordinal_alpha=criterion_settings['ordinal_alpha'],
    label_smoothing=criterion_settings['label_smoothing'],
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
)
print(f'LR Scheduler: ReduceLROnPlateau | factor=0.5 | patience=5 | min_lr=1e-6')
print(
    f"Loss: focal_ordinal | gamma={criterion_settings['focal_gamma']} | "
    f"ordinal_alpha={criterion_settings['ordinal_alpha']} | "
    f"label_smoothing={criterion_settings['label_smoothing']}"
 )

history = {
    'epoch': [],
    'train_Loss': [], 'val_Loss': [],
    'train_Accuracy': [], 'val_Accuracy': [],
    'train_Macro_F1': [], 'val_Macro_F1': [],
    'train_AUC': [], 'val_AUC': [],
    'train_QWK': [], 'val_QWK': [],
}

best_val_f1 = -1.0
best_state = None
patience, no_improve = 50, 0
max_epochs = 50

for epoch in range(1, max_epochs + 1):
    tr = run_epoch(model, train_loader, criterion, optimizer)
    va = run_epoch(model, val_loader, criterion, optimizer=None)
    scheduler.step(va['Loss'])  # Fix 1: giảm LR khi val_Loss không cải thiện\n
    history['epoch'].append(epoch)
    history['train_Loss'].append(float(tr['Loss']))
    history['val_Loss'].append(float(va['Loss']))
    for metric_name in ['Accuracy', 'Macro_F1', 'AUC', 'QWK']:
        history[f'train_{metric_name}'].append(float(tr[metric_name]))
        history[f'val_{metric_name}'].append(float(va[metric_name]))

    print(
        f"Epoch {epoch:02d} | TrLoss {tr['Loss']:.4f} | VaLoss {va['Loss']:.4f} | "
        f"VaAcc {va['Accuracy']:.4f} | VaF1 {va['Macro_F1']:.4f} | VaAUC {va['AUC']:.4f} | VaQWK {va['QWK']:.4f}"
    )

    if va['Macro_F1'] > best_val_f1 + 1e-4:
        best_val_f1 = va['Macro_F1']
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print('Early stopping.')
            break

if best_state is not None:
    model.load_state_dict(best_state)

history_df = pd.DataFrame(history)
history_path = ARTIFACT_DIR / 'patchtst_training_history.csv'
history_df.to_csv(history_path, index=False)

val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)
test_metrics = run_epoch(model, test_loader, criterion, optimizer=None)

report = pd.DataFrame([
    {'Split': 'Val', **{k: v for k, v in val_metrics.items() if k in ['Accuracy', 'Macro_F1', 'AUC', 'QWK']}},
    {'Split': 'Test', **{k: v for k, v in test_metrics.items() if k in ['Accuracy', 'Macro_F1', 'AUC', 'QWK']}},
])
display(report)

out_path = ARTIFACT_DIR / 'patchtst_metrics.csv'
report.to_csv(out_path, index=False)
print('Saved:', out_path)
print('Saved:', history_path)