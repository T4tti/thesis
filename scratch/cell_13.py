if 'history_df' not in globals():
    raise RuntimeError('Khong tim thay history_df. Hay chay lai cell huan luyen truoc.')

history_df = history_df.copy()
history_df['val_MetricScore'] = history_df.apply(
    lambda row: selection_score({
        'Accuracy': row['val_Accuracy'],
        'QWK': row['val_QWK'],
        'Macro_F1': row['val_Macro_F1'],
        'Class0_F2': row['val_Class0_F2'],
        'ChgAcc': row['val_ChgAcc'],
        'Ordinal_MAE': row['val_Ordinal_MAE'],
    }),
    axis=1,
)
history_df['val_CheckpointScore'] = history_df.apply(
    lambda row: checkpoint_score(
        {
            'Accuracy': row['val_Accuracy'],
            'QWK': row['val_QWK'],
            'Macro_F1': row['val_Macro_F1'],
            'Class0_F2': row['val_Class0_F2'],
            'ChgAcc': row['val_ChgAcc'],
            'Ordinal_MAE': row['val_Ordinal_MAE'],
        },
        row['val_CE_Loss'] if 'val_CE_Loss' in row else row['val_Loss'],
        row['train_CE_Loss'] if 'train_CE_Loss' in row else row['train_Loss'],
    ),
    axis=1,
)
loss_for_min_col = 'val_CE_Loss' if 'val_CE_Loss' in history_df.columns else 'val_Loss'
train_loss_for_gap_col = 'train_CE_Loss' if 'train_CE_Loss' in history_df.columns else 'train_Loss'
history_df['val_LossGap'] = history_df.get('val_LossGap', history_df[loss_for_min_col] - history_df[train_loss_for_gap_col])

selected_epoch = selected_checkpoint_epoch if 'selected_checkpoint_epoch' in globals() else None
if selected_epoch is not None and not pd.isna(selected_epoch) and (history_df['epoch'] == int(selected_epoch)).any():
    best_epoch_idx = history_df.index[history_df['epoch'] == int(selected_epoch)][0]
else:
    best_epoch_idx = history_df['val_CheckpointScore'].idxmax()

min_val_loss_idx = history_df[loss_for_min_col].idxmin()
best_val_acc_idx = history_df['val_Accuracy'].idxmax()
best_gap_idx = history_df['val_LossGap'].idxmin()
best_epoch = int(history_df.loc[best_epoch_idx, 'epoch']) if 'epoch' in history_df.columns else int(best_epoch_idx) + 1
min_val_loss_epoch = int(history_df.loc[min_val_loss_idx, 'epoch']) if 'epoch' in history_df.columns else int(min_val_loss_idx) + 1
best_val_acc_epoch = int(history_df.loc[best_val_acc_idx, 'epoch']) if 'epoch' in history_df.columns else int(best_val_acc_idx) + 1
best_gap_epoch = int(history_df.loc[best_gap_idx, 'epoch']) if 'epoch' in history_df.columns else int(best_gap_idx) + 1

row = history_df.loc[best_epoch_idx]
best_train_loss = float(row['train_Loss'])
best_val_loss = float(row['val_Loss'])
best_train_ce_loss = float(row['train_CE_Loss']) if 'train_CE_Loss' in row else best_train_loss
best_val_ce_loss = float(row['val_CE_Loss']) if 'val_CE_Loss' in row else best_val_loss
best_val_aux_loss = float(row['val_Aux_Loss']) if 'val_Aux_Loss' in row else 0.0
best_loss_gap = float(row['val_LossGap'])
best_train_acc = float(row['train_Accuracy'])
best_val_acc = float(row['val_Accuracy'])
best_val_class0_recall = float(row['val_Class0_Recall'])
best_val_class0_f2 = float(row['val_Class0_F2'])
checkpoint_tag = selected_checkpoint_tag if 'selected_checkpoint_tag' in globals() else CHECKPOINT_CONFIG.get('mode', 'unknown')

print(f'Best metrics (selected checkpoint: {checkpoint_tag}):')
print(f'Train Loss:       {best_train_loss:.6f} @ epoch {best_epoch}')
print(f'Val Loss:         {best_val_loss:.6f} @ epoch {best_epoch}')
print(f'Train CE Loss:    {best_train_ce_loss:.6f} @ epoch {best_epoch}')
print(f'Val CE Loss:      {best_val_ce_loss:.6f} @ epoch {best_epoch}')
print(f'CE Loss Gap:      {best_loss_gap:.6f} @ epoch {best_epoch}')
print(f'Val Aux Loss:     {best_val_aux_loss:.6f} @ epoch {best_epoch}')
print(f'Train Acc:        {best_train_acc:.6f} @ epoch {best_epoch}')
print(f'Val Acc:          {best_val_acc:.6f} @ epoch {best_epoch}')
print(f'Val Class0 Recall:{best_val_class0_recall:.6f} @ epoch {best_epoch}')
print(f'Val Class0 F2:    {best_val_class0_f2:.6f} @ epoch {best_epoch}')
print(f'Min Val Loss:     {float(history_df.loc[min_val_loss_idx, loss_for_min_col]):.6f} @ epoch {min_val_loss_epoch}')
print(f'Best Val Acc:     {float(history_df.loc[best_val_acc_idx, "val_Accuracy"]):.6f} @ epoch {best_val_acc_epoch}')
print(f'Min CE Loss Gap:  {float(history_df.loc[best_gap_idx, "val_LossGap"]):.6f} @ epoch {best_gap_epoch}')

summary_df = pd.DataFrame([
    {
        'checkpoint': checkpoint_tag,
        'epoch': best_epoch,
        'train_loss': best_train_loss,
        'val_loss': best_val_loss,
        'train_ce_loss': best_train_ce_loss,
        'val_ce_loss': best_val_ce_loss,
        'ce_loss_gap': best_loss_gap,
        'val_aux_loss': best_val_aux_loss,
        'train_acc': best_train_acc,
        'val_acc': best_val_acc,
        'val_class0_recall': best_val_class0_recall,
        'val_class0_f2': best_val_class0_f2,
        'val_metric_score': float(row['val_MetricScore']),
        'val_checkpoint_score': float(row['val_CheckpointScore']),
        'checkpoint_mode': CHECKPOINT_CONFIG.get('mode', 'unknown') if 'CHECKPOINT_CONFIG' in globals() else 'unknown',
        'balanced_accuracy_floor_drop': CHECKPOINT_CONFIG.get('balanced_accuracy_floor_drop', np.nan) if 'CHECKPOINT_CONFIG' in globals() else np.nan,
        'min_val_loss_epoch': min_val_loss_epoch,
        'min_val_loss': float(history_df.loc[min_val_loss_idx, loss_for_min_col]),
        'min_val_loss_source': loss_for_min_col,
        'best_val_acc_epoch': best_val_acc_epoch,
        'best_val_acc': float(history_df.loc[best_val_acc_idx, 'val_Accuracy']),
        'best_gap_epoch': best_gap_epoch,
        'best_gap': float(history_df.loc[best_gap_idx, 'val_LossGap']),
        'learning_rate': float(row['Learning_Rate']) if 'Learning_Rate' in row else np.nan,
        'selected_class0_threshold': class0_threshold if 'class0_threshold' in globals() else np.nan,
        'persistence_prior_scale': PERSISTENCE_PRIOR_SCALE if 'PERSISTENCE_PRIOR_SCALE' in globals() else np.nan,
    }
])

training_summary_path = ARTIFACT_DIR / 'gat_training_summary.csv'
summary_df.to_csv(training_summary_path, index=False, encoding='utf-8-sig')
display(summary_df)
print('Saved:', training_summary_path)
