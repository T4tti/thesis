# How to Run: python scratch/patch_transformer_curves.py
# Expected Output: Modifies cell 36 of notebooks/transformer-lstm.ipynb to map standard metrics (Loss, Accuracy, Macro_F1, ChgAcc, AUC, QWK) to the lowercase/train_eval keys of the Transformer-LSTM notebook, preventing KeyErrors / RuntimeErrors.

import json
from pathlib import Path

def main():
    nb_path = Path('notebooks/transformer-lstm.ipynb')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_source = [
        "if 'history_df' not in globals():\n",
        "    raise RuntimeError('Khong tim thay history_df. Hay chay lai cell huan luyen truoc.')\n",
        "\n",
        "from matplotlib.ticker import MultipleLocator\n",
        "\n",
        "sns.set_theme(style='whitegrid', context='paper')\n",
        "\n",
        "# Map display metric names to the actual column names in history_df of this notebook\n",
        "metric_map = {\n",
        "    'Loss': {\n",
        "        'train': 'train_eval_loss' if 'train_eval_loss' in history_df.columns else 'train_loss',\n",
        "        'val': 'val_loss'\n",
        "    },\n",
        "    'Accuracy': {\n",
        "        'train': 'train_eval_acc' if 'train_eval_acc' in history_df.columns else 'train_acc',\n",
        "        'val': 'val_acc'\n",
        "    },\n",
        "    'Macro_F1': {\n",
        "        'train': 'train_eval_f1_weighted' if 'train_eval_f1_weighted' in history_df.columns else 'train_f1_weighted',\n",
        "        'val': 'val_f1_weighted'\n",
        "    },\n",
        "    'ChgAcc': {\n",
        "        'train': 'train_eval_chgacc' if 'train_eval_chgacc' in history_df.columns else 'train_chgacc',\n",
        "        'val': 'val_chgacc'\n",
        "    },\n",
        "    'AUC': {\n",
        "        'train': 'train_eval_auc' if 'train_eval_auc' in history_df.columns else 'train_auc',\n",
        "        'val': 'val_auc'\n",
        "    },\n",
        "    'QWK': {\n",
        "        'train': 'train_eval_qwk' if 'train_eval_qwk' in history_df.columns else 'train_qwk',\n",
        "        'val': 'val_qwk'\n",
        "    }\n",
        "}\n",
        "\n",
        "# Verify required columns exist in history_df\n",
        "missing = []\n",
        "for metric, cols in metric_map.items():\n",
        "    if cols['train'] not in history_df.columns:\n",
        "        missing.append(cols['train'])\n",
        "    if cols['val'] not in history_df.columns:\n",
        "        missing.append(cols['val'])\n",
        "\n",
        "if missing:\n",
        "    raise RuntimeError(f'Thieu cot trong history_df: {missing}. Hay chay lai cell huan luyen.')\n",
        "\n",
        "fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=160, constrained_layout=True)\n",
        "axes = axes.ravel()\n",
        "max_epoch = int(history_df['epoch'].max())\n",
        "\n",
        "for ax, metric in zip(axes, metric_map.keys()):\n",
        "    t_col = metric_map[metric]['train']\n",
        "    v_col = metric_map[metric]['val']\n",
        "    \n",
        "    ax.plot(history_df['epoch'], history_df[t_col], label='Train', linewidth=1.8, color='#1f77b4')\n",
        "    ax.plot(history_df['epoch'], history_df[v_col], label='Validation', linewidth=1.8, color='#d62728')\n",
        "    \n",
        "    if metric == 'Loss':\n",
        "        val_loss_col = metric_map['Loss']['val']\n",
        "        best_epoch = int(history_df.loc[history_df[val_loss_col].idxmin(), 'epoch'])\n",
        "        ax.axvline(best_epoch, color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.8, label='Best val loss')\n",
        "        \n",
        "    ax.set_title(metric, fontsize=11, fontweight='semibold')\n",
        "    ax.set_xlabel('Epoch')\n",
        "    ax.set_xlim(0, max_epoch)\n",
        "    ax.xaxis.set_major_locator(MultipleLocator(10))\n",
        "    ax.set_ylabel(metric)\n",
        "    ax.grid(True, linestyle='--', alpha=0.35)\n",
        "    ax.legend(frameon=True, fontsize=9)\n",
        "\n",
        "fig.suptitle('Transformer-LSTM Training Curves', fontsize=13, fontweight='bold')\n",
        "curve_path = ARTIFACT_DIR / 'transformer_lstm_training_curves.png' if 'ARTIFACT_DIR' in globals() else Path('transformer_lstm_training_curves.png')\n",
        "fig.savefig(curve_path, dpi=300, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print('Saved:', curve_path)\n"
    ]

    nb['cells'][36]['source'] = new_source
    nb['cells'][36]['outputs'] = []
    nb['cells'][36]['execution_count'] = None

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("Successfully patched cell 36 with metric mappings!")

if __name__ == '__main__':
    main()
