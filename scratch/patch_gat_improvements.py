"""
Patch script for gat-baseline.ipynb
Implements improvements 1, 2, 3, 5, 8 from the analysis:
  1. Enable class weights (use_class_weights = True)
  2. Increase focal loss strength (gamma=2.0, weight=0.5, ordinal=0.05, warmup=20)
  3. Add CosineAnnealingLR LR scheduler (T_max=100) suitable for 100 epochs
  5. Increase Class0_F2 weight in selection score (0.10 -> 0.25)
  8. Class-balanced KNN graph construction

How to Run:
  python scratch/patch_gat_improvements.py

Expected Output:
  Report of which cells were modified, or warning if already patched.
"""

import json
from pathlib import Path

notebook_path = Path(r'e:\thesis\notebooks\gat-baseline.ipynb')

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

changes: dict[str, list[str]] = {}

for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue

    source = ''.join(cell['source'])
    original = source
    cell_id = cell.get('id', 'unknown')
    cell_changes: list[str] = []

    # ================================================================
    # Changes 1 & 2: LOSS_CONFIG — class weights + focal/ordinal tuning
    # ================================================================
    old_loss_config = (
        "LOSS_CONFIG = {\n"
        "    'ce_weight': 1.0,\n"
        "    'focal_gamma': 0.25,\n"
        "    'focal_weight': 0.03,\n"
        "    'ordinal_weight': 0.005,\n"
        "    'warmup_epochs': 100,\n"
        "    'use_class_weights': False,\n"
        "}"
    )
    new_loss_config = (
        "LOSS_CONFIG = {\n"
        "    'ce_weight': 1.0,\n"
        "    'focal_gamma': 2.0,\n"
        "    'focal_weight': 0.5,\n"
        "    'ordinal_weight': 0.05,\n"
        "    'warmup_epochs': 20,\n"
        "    'use_class_weights': True,\n"
        "}"
    )
    if old_loss_config in source:
        source = source.replace(old_loss_config, new_loss_config)
        cell_changes.append("[1] use_class_weights = True")
        cell_changes.append("[2] focal_gamma=2.0, focal_weight=0.5, ordinal_weight=0.05, warmup=20")

    # ================================================================
    # Change 3a: Add LR scheduler after optimizer
    # ================================================================
    old_optimizer_line = (
        "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)\n"
        "print('Loss config:', LOSS_CONFIG)"
    )
    new_optimizer_lines = (
        "optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)\n"
        "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n"
        "    optimizer, T_max=100, eta_min=1e-5,\n"
        ")\n"
        "print('Loss config:', LOSS_CONFIG)"
    )
    if old_optimizer_line in source and 'scheduler' not in source:
        source = source.replace(old_optimizer_line, new_optimizer_lines)
        cell_changes.append("[3a] CosineAnnealingLR scheduler added")

    # ================================================================
    # Change 3b: Add scheduler.step() in training loop
    # ================================================================
    old_step = "    optimizer.step()\n\n    model.eval()"
    new_step = "    optimizer.step()\n    scheduler.step()\n\n    model.eval()"
    if old_step in source and 'scheduler.step()' not in source:
        source = source.replace(old_step, new_step)
        cell_changes.append("[3b] scheduler.step() added to training loop")

    # ================================================================
    # Change 5: Selection score — increase Class0_F2 weight
    # ================================================================
    old_selection = (
        "        0.60 * metrics['Accuracy']\n"
        "        + 0.15 * metrics['QWK']\n"
        "        + 0.10 * metrics['Macro_F1']\n"
        "        + 0.10 * metrics['Class0_F2']\n"
        "        + 0.05 * chg_acc\n"
    )
    new_selection = (
        "        0.40 * metrics['Accuracy']\n"
        "        + 0.15 * metrics['QWK']\n"
        "        + 0.10 * metrics['Macro_F1']\n"
        "        + 0.25 * metrics['Class0_F2']\n"
        "        + 0.10 * chg_acc\n"
    )
    if old_selection in source:
        source = source.replace(old_selection, new_selection)
        cell_changes.append("[5] selection_score: Accuracy 0.60->0.40, Class0_F2 0.10->0.25, chg_acc 0.05->0.10")

    # ================================================================
    # Change 8: Class-balanced KNN graph construction
    # ================================================================
    old_knn_body = (
        "def build_edge_index(frame, feature_matrix, train_mask_np, k_neighbors=16):\n"
        "    edges = []\n"
        "    n_nodes = len(frame)\n"
        "    train_indices = np.flatnonzero(train_mask_np)\n"
        "    k = min(int(k_neighbors), len(train_indices))\n"
        "    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')\n"
        "    nn.fit(feature_matrix[train_indices])\n"
        "    neigh = nn.kneighbors(feature_matrix, return_distance=False)\n"
        "    for dst in range(n_nodes):\n"
        "        for local_src in neigh[dst]:\n"
        "            src = int(train_indices[local_src])\n"
        "            edges.append((src, dst))\n"
    )
    new_knn_body = (
        "def build_edge_index(frame, feature_matrix, train_mask_np, k_neighbors=16):\n"
        "    \"\"\"Class-balanced KNN: ensures each node has neighbors from every class.\"\"\"\n"
        "    edges = []\n"
        "    n_nodes = len(frame)\n"
        "    train_indices = np.flatnonzero(train_mask_np)\n"
        "    labels = frame[TARGET_COL].values\n"
        "\n"
        "    # Per-class KNN to balance neighbor representation\n"
        "    for cls in range(n_classes):\n"
        "        cls_mask = labels[train_indices] == cls\n"
        "        cls_indices = train_indices[cls_mask]\n"
        "        k = min(max(k_neighbors // n_classes, 2), len(cls_indices))\n"
        "        nn_cls = NearestNeighbors(n_neighbors=k, metric='euclidean')\n"
        "        nn_cls.fit(feature_matrix[cls_indices])\n"
        "        neigh = nn_cls.kneighbors(feature_matrix, return_distance=False)\n"
        "        for dst in range(n_nodes):\n"
        "            for local_src in neigh[dst]:\n"
        "                src = int(cls_indices[local_src])\n"
        "                edges.append((src, dst))\n"
    )
    if old_knn_body in source:
        source = source.replace(old_knn_body, new_knn_body)
        cell_changes.append("[8] Class-balanced KNN implemented")

    # Apply changes to cell
    if source != original:
        cell['source'] = source.splitlines(True)
        # Ensure last line ends with newline
        if cell['source'] and not cell['source'][-1].endswith('\n'):
            cell['source'][-1] += '\n'
        # Clear stale outputs
        cell['outputs'] = []
        cell['execution_count'] = None
        changes[cell_id] = cell_changes

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

# Report
if changes:
    print("=" * 60)
    print("  GAT Baseline Notebook — Patch Applied Successfully")
    print("=" * 60)
    print(f"\n  Modified {len(changes)} cell(s):\n")
    for cell_id, cell_changes in changes.items():
        print(f"  Cell '{cell_id}':")
        for change in cell_changes:
            print(f"    ✅ {change}")
    print(f"\n  Please re-run the notebook to see updated results.")
    print("=" * 60)
else:
    print("⚠️  No changes were made. The notebook may already be patched.")
