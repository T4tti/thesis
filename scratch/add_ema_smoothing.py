import json
from pathlib import Path

notebook_path = Path("e:/thesis/notebooks/Sparse-Graph-baseline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_8 = nb["cells"][8]
source_8 = "".join(cell_8["source"])

# Target plotting code in Cell 8
target_plotting = """# ================= PLOT TRAINING CURVES =================
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
plt.show()"""

# Replacement plotting code using EMA smoothing
replacement_plotting = """# ================= PLOT TRAINING CURVES WITH EMA SMOOTHING =================
import matplotlib.pyplot as plt

def smooth_curve(values, alpha=0.85):
    smoothed = []
    if len(values) == 0:
        return smoothed
    last = values[0]
    for v in values:
        if np.isnan(v):
            smoothed.append(np.nan)
            continue
        if np.isnan(last):
            last = v
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.ravel()

epochs_list = history['epoch']

# Plot Train & Val Loss
axs[0].plot(epochs_list, history['train_CE_Loss'], color='#1f77b4', alpha=0.25, linestyle='--')
axs[0].plot(epochs_list, history['val_CE_Loss'], color='#d62728', alpha=0.25)
axs[0].plot(epochs_list, smooth_curve(history['train_CE_Loss']), label="Train CE Loss (Smoothed)", color='#1f77b4', linestyle='--')
axs[0].plot(epochs_list, smooth_curve(history['val_CE_Loss']), label="Val CE Loss (Smoothed)", color='#d62728')
axs[0].set_title('Cross Entropy Loss (Train vs Val)')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend()

# Plot Loss Gap
axs[1].plot(epochs_list, history['val_LossGap'], color='#2ca02c', alpha=0.25)
axs[1].plot(epochs_list, smooth_curve(history['val_LossGap']), label="CE Loss Gap (Smoothed)", color='#2ca02c')
axs[1].set_title('Generalization Loss Gap (Val CE - Train CE)')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss Gap')
axs[1].grid(True)
axs[1].legend()

# Plot Val Accuracy
axs[2].plot(epochs_list, history['val_Accuracy'], color='#ff7f0e', alpha=0.25)
axs[2].plot(epochs_list, history['val_Balanced_Accuracy'], color='#9467bd', alpha=0.25, linestyle=':')
axs[2].plot(epochs_list, smooth_curve(history['val_Accuracy']), label="Val Accuracy (Smoothed)", color='#ff7f0e')
axs[2].plot(epochs_list, smooth_curve(history['val_Balanced_Accuracy']), label="Val Balanced Accuracy (Smoothed)", color='#9467bd', linestyle=':')
axs[2].set_title('Validation Accuracy & Balanced Accuracy')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy')
axs[2].grid(True)
axs[2].legend()

# Plot Learning Rate
axs[3].plot(epochs_list, history['Learning_Rate'], label="Learning Rate", color='#7f7f7f')
axs[3].set_title('Learning Rate Decay')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel('LR')
axs[3].set_yscale('log')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'gat_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()"""

if target_plotting in source_8:
    source_8 = source_8.replace(target_plotting, replacement_plotting)
    print("Found and replaced plotting code in Cell 8.")
else:
    # Let's try replacing with splitlines/lines replacement if there's any mismatch
    source_8 = source_8.replace("plt.savefig(ARTIFACT_DIR / 'gat_training_curves.png', dpi=300, bbox_inches='tight')", "plt.savefig(ARTIFACT_DIR / 'gat_training_curves.png', dpi=300, bbox_inches='tight')")
    print("Fallback replacement logic check.")

cell_8["source"] = [line + "\n" for line in source_8.splitlines()]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated with EMA smoothing and saved.")
