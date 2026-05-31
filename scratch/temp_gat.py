import matplotlib
matplotlib.use('Agg')  # Headless plotting
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Avoid character encoding issues
def display(*args, **kwargs):
    for arg in args:
        print(arg)

import os

import copy

import random

from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize

from sklearn.metrics import (

    accuracy_score, f1_score, precision_score, recall_score,

    roc_auc_score, roc_curve, auc, cohen_kappa_score,

    confusion_matrix, classification_report,

    precision_recall_fscore_support,

)



import torch

import torch.nn as nn

import torch.nn.functional as F



SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def detect_kaggle_runtime() -> bool:

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '').strip():

        return True

    return Path('/kaggle/input').exists() and Path('/kaggle/working').exists()





IN_KAGGLE = detect_kaggle_runtime()





def find_project_root(start: Path) -> Path:

    for p in [start, *start.parents]:

        if (p / 'data').exists() and (p / 'src').exists():

            return p

    return start





PROJECT_ROOT = Path('/kaggle/working') if IN_KAGGLE else find_project_root(Path.cwd().resolve())

ARTIFACT_DIR = PROJECT_ROOT / 'credit_rating_artifacts'

DMF_ARTIFACT_DIR = ARTIFACT_DIR / 'dmf_gat_lstm'

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DMF_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print('Device:', device)

print('Project root:', PROJECT_ROOT)

print('DMF artifact dir:', DMF_ARTIFACT_DIR)


# ----------------------------------------

FINANCIAL_FEATURES = [

    'current_ratio', 'debt_equity_ratio', 'gross_profit_margin', 'operating_profit_margin',

    'ebit_margin', 'pretax_profit_margin', 'net_profit_margin', 'asset_turnover',

    'roe', 'roa', 'operating_cashflow_ps', 'free_cashflow_ps'

]

TARGET_COL = 'rating_detail'

TARGET_ORDERED_LABELS = ['Distressed', 'HY', 'IG']





def resolve_split_path(default_path, local_fallbacks=None):

    candidates = [Path(default_path)]

    for p in (local_fallbacks or []):

        p_obj = Path(p)

        candidates.append(PROJECT_ROOT / p_obj if not p_obj.is_absolute() else p_obj)

    if IN_KAGGLE:

        kaggle_root = Path('/kaggle/input')

        expanded = []

        for p in candidates:

            expanded.append(p)

            if not p.exists() and kaggle_root.exists():

                expanded.extend(kaggle_root.rglob(p.name))

        candidates = expanded

    seen = set()

    deduped = []

    for p in candidates:

        p = Path(p)

        key = str(p)

        if key not in seen:

            seen.add(key)

            deduped.append(p)

    for p in deduped:

        if p.exists():

            return p

    raise FileNotFoundError(f'Khong tim thay file split: {deduped}')





TRAIN_PATH = resolve_split_path(

    '/kaggle/input/datasets/tailength/corporate-credit-rating/test/train.csv',

    ['data/processed/test/train.csv'],

)

VAL_PATH = resolve_split_path(

    '/kaggle/input/datasets/tailength/corporate-credit-rating/test/val.csv',

    ['data/processed/test/val.csv'],

)

TEST_PATH = resolve_split_path(

    '/kaggle/input/datasets/tailength/corporate-credit-rating/test/test.csv',

    ['data/processed/test/test.csv'],

)





def load_split(path, split_name):

    frame = pd.read_csv(path)

    frame = frame.copy()

    frame['__split__'] = split_name

    frame['__split_row_index__'] = np.arange(len(frame), dtype=int)

    if 'row_id' not in frame.columns:

        frame['row_id'] = [f'{split_name}_{i:06d}' for i in range(len(frame))]

    else:

        frame['row_id'] = frame['row_id'].astype(str)

    return frame





train_df = load_split(TRAIN_PATH, 'train')

val_df = load_split(VAL_PATH, 'val')

test_df = load_split(TEST_PATH, 'test')

df = pd.concat([train_df, val_df, test_df], ignore_index=True)



split_contract = {

    'train_path': str(TRAIN_PATH),

    'val_path': str(VAL_PATH),

    'test_path': str(TEST_PATH),

    'train_rows': int(len(train_df)),

    'val_rows': int(len(val_df)),

    'test_rows': int(len(test_df)),

    'row_id_rule': 'existing row_id if present, otherwise <split>_<zero_padded_original_split_index>',

}

print('DMF split contract:', split_contract)



df = df.dropna(subset=[TARGET_COL]).copy()

target_as_num = pd.to_numeric(df[TARGET_COL], errors='coerce')

if target_as_num.notna().all():

    df[TARGET_COL] = target_as_num.astype(int)

    observed = sorted(df[TARGET_COL].unique().tolist())

    raw_to_id = {int(v): i for i, v in enumerate(observed)}

    id_to_raw = {i: int(v) for v, i in raw_to_id.items()}

    df[TARGET_COL] = df[TARGET_COL].map(raw_to_id).astype(int)

else:

    tgt = df[TARGET_COL].astype(str).str.strip()

    observed = sorted(tgt.unique().tolist())

    ordered = [x for x in TARGET_ORDERED_LABELS if x in observed] if set(observed).issubset(set(TARGET_ORDERED_LABELS)) else observed

    raw_to_id = {v: i for i, v in enumerate(ordered)}

    id_to_raw = {i: v for i, v in raw_to_id.items()}

    df[TARGET_COL] = tgt.map(raw_to_id).astype(int)



n_classes = int(df[TARGET_COL].nunique())

label_contract = pd.DataFrame({

    'label_id': list(range(n_classes)),

    'label_name': [str(id_to_raw.get(i, i)) for i in range(n_classes)],

})

label_contract.to_csv(DMF_ARTIFACT_DIR / 'label_mapping.csv', index=False, encoding='utf-8-sig')



df['rating_date'] = pd.to_datetime(df['rating_date'], errors='coerce', format='mixed')

if 'sector' not in df.columns:

    df['sector'] = 'UNKNOWN'

df['sector'] = df['sector'].fillna('UNKNOWN').astype(str)

if 'ticker' not in df.columns:

    df['ticker'] = 'UNKNOWN'

df['ticker'] = df['ticker'].fillna('UNKNOWN').astype(str)

if 'company_name' not in df.columns:

    df['company_name'] = df['ticker']

df['company_name'] = df['company_name'].fillna(df['ticker']).astype(str)



sector_encoder = LabelEncoder()

df['sector_id'] = sector_encoder.fit_transform(df['sector'])

n_sectors = int(df['sector_id'].nunique())



train_mask_raw = df['__split__'].eq('train')

stats_ref = df.loc[train_mask_raw].copy()

for c in FINANCIAL_FEATURES:

    med = stats_ref[c].median() if stats_ref[c].notna().any() else 0.0

    df[c] = df[c].fillna(float(0.0 if pd.isna(med) else med))

for c in FINANCIAL_FEATURES:

    lo = stats_ref[c].quantile(0.01)

    hi = stats_ref[c].quantile(0.99)

    if pd.notna(lo) and pd.notna(hi):

        df[c] = df[c].clip(float(lo), float(hi))



df = df.sort_values(['ticker', 'rating_date', '__split__', '__split_row_index__']).reset_index(drop=True)

for c in FINANCIAL_FEATURES:

    df[f'{c}_delta'] = df.groupby('ticker')[c].diff().fillna(0.0)

MODEL_FEATURES = FINANCIAL_FEATURES + [f'{c}_delta' for c in FINANCIAL_FEATURES]



scaler = RobustScaler()

scaler.fit(df.loc[df['__split__'].eq('train'), MODEL_FEATURES].values)

df[MODEL_FEATURES] = scaler.transform(df[MODEL_FEATURES].values)



df['last_y'] = df.groupby('ticker')[TARGET_COL].shift(1)

df['last_y'] = df['last_y'].fillna(df[TARGET_COL]).astype(int)



x_all = torch.tensor(df[MODEL_FEATURES].values.astype(np.float32), dtype=torch.float32, device=device)

y_all = torch.tensor(df[TARGET_COL].values.astype(int), dtype=torch.long, device=device)

last_y_all = torch.tensor(df['last_y'].values.astype(int), dtype=torch.long, device=device)

sector_all = torch.tensor(df['sector_id'].values.astype(int), dtype=torch.long, device=device)



train_mask = torch.tensor(df['__split__'].eq('train').values, dtype=torch.bool, device=device)

val_mask = torch.tensor(df['__split__'].eq('val').values, dtype=torch.bool, device=device)

test_mask = torch.tensor(df['__split__'].eq('test').values, dtype=torch.bool, device=device)



train_class_counts = torch.bincount(y_all[train_mask], minlength=n_classes).float()

train_class_weights = train_class_counts.sum() / train_class_counts.clamp_min(1.0)

train_class_weights = train_class_weights / train_class_weights.mean().clamp_min(1e-12)

print('Train class counts:', train_class_counts.detach().cpu().numpy().astype(int).tolist())

print('Train class weights:', np.round(train_class_weights.detach().cpu().numpy(), 4).tolist())



print('Rows train/val/test:', int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))

print('n_classes:', n_classes, '| n_sectors:', n_sectors, '| n_features:', len(MODEL_FEATURES))


# ----------------------------------------

CLASS0_LABEL_ID = 0



CLASS0_THRESHOLD_CONFIG = {

    'enabled': True,

    'metric': 'Class0_F2',

    'accuracy_floor_drop': 0.01,

    'threshold_grid': np.round(np.arange(0.05, 0.501, 0.01), 2).tolist(),

}





def predict_with_class0_threshold(proba, class0_threshold=None):

    pred = np.asarray(proba).argmax(axis=1).astype(int)

    if class0_threshold is None:

        return pred

    promote_mask = np.asarray(proba)[:, CLASS0_LABEL_ID] >= float(class0_threshold)

    pred[promote_mask] = CLASS0_LABEL_ID

    return pred





def compute_metrics(y_true, y_pred, proba, n_cls, last_y=None):

    y_true_arr = np.asarray(y_true)

    y_pred_arr = np.asarray(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)

    f1m = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)

    f1w = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)

    prec = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)

    rec = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)

    class_prec, class_rec, class_f1, class_support = precision_recall_fscore_support(

        y_true_arr,

        y_pred_arr,

        labels=list(range(n_cls)),

        zero_division=0,

    )

    c0_precision = float(class_prec[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_prec) else float('nan')

    c0_recall = float(class_rec[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_rec) else float('nan')

    c0_f1 = float(class_f1[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_f1) else float('nan')

    c0_support = int(class_support[CLASS0_LABEL_ID]) if CLASS0_LABEL_ID < len(class_support) else 0

    if c0_precision + c0_recall > 0:

        c0_f2 = float(5.0 * c0_precision * c0_recall / (4.0 * c0_precision + c0_recall))

    else:

        c0_f2 = 0.0

    qwk = cohen_kappa_score(y_true_arr, y_pred_arr, weights='quadratic')

    try:

        y_bin = label_binarize(y_true_arr, classes=list(range(n_cls)))

        auc_score = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')

    except Exception:

        auc_score = float('nan')

    ordinal_mae = np.mean(np.abs(y_true_arr - y_pred_arr))

    # ChgAcc: accuracy on samples where label changed vs last known rating.

    if last_y is not None:

        last_y_arr = np.asarray(last_y)

        change_mask = last_y_arr != y_true_arr

        if change_mask.sum() > 0:

            chg_acc = float(accuracy_score(y_true_arr[change_mask], y_pred_arr[change_mask]))

        else:

            chg_acc = float('nan')

    else:

        chg_acc = float('nan')

    return {

        'Accuracy': float(acc),

        'Precision_Weighted': float(prec),

        'Recall_Weighted': float(rec),

        'Macro_F1': float(f1m),

        'Weighted_F1': float(f1w),

        'Class0_Precision': c0_precision,

        'Class0_Recall': c0_recall,

        'Class0_F1': c0_f1,

        'Class0_F2': c0_f2,

        'Class0_Support': c0_support,

        'AUC': float(auc_score),

        'QWK': float(qwk),

        'ChgAcc': chg_acc,

        'Ordinal_MAE': float(ordinal_mae),

    }





def evaluate_logits(logits, mask, class0_threshold=None):

    probs = torch.softmax(logits[mask], dim=1).detach().cpu().numpy()

    y_true = y_all[mask].detach().cpu().numpy()

    y_pred = predict_with_class0_threshold(probs, class0_threshold=class0_threshold)

    last_y_np = last_y_all[mask].detach().cpu().numpy()

    return compute_metrics(y_true, y_pred, probs, n_classes, last_y=last_y_np), y_true, y_pred, probs





def selection_score(metrics):

    chg_acc = 0.0 if np.isnan(metrics['ChgAcc']) else metrics['ChgAcc']

    return (

        0.60 * metrics['Accuracy']

        + 0.15 * metrics['QWK']

        + 0.10 * metrics['Macro_F1']

        + 0.10 * metrics['Class0_F2']

        + 0.05 * chg_acc

        - 0.05 * metrics['Ordinal_MAE']

    )





def calibrate_class0_threshold(y_true, proba, last_y=None, config=None):

    config = config or CLASS0_THRESHOLD_CONFIG

    baseline_pred = predict_with_class0_threshold(proba, class0_threshold=None)

    baseline_metrics = compute_metrics(y_true, baseline_pred, proba, n_classes, last_y=last_y)

    rows = []

    for threshold in config['threshold_grid']:

        pred = predict_with_class0_threshold(proba, class0_threshold=threshold)

        metrics = compute_metrics(y_true, pred, proba, n_classes, last_y=last_y)

        rows.append({'class0_threshold': float(threshold), **metrics})

    sweep_df = pd.DataFrame(rows)

    accuracy_floor = baseline_metrics['Accuracy'] - float(config.get('accuracy_floor_drop', 0.01))

    candidates = sweep_df[sweep_df['Accuracy'] >= accuracy_floor].copy()

    if candidates.empty:

        candidates = sweep_df.copy()

    sort_cols = [config.get('metric', 'Class0_F2'), 'Accuracy', 'Macro_F1', 'QWK']

    best_row = candidates.sort_values(sort_cols, ascending=False).iloc[0]

    best_threshold = float(best_row['class0_threshold'])

    return best_threshold, sweep_df, baseline_metrics, best_row.to_dict()


# ----------------------------------------

KNN_K = 16





def build_edge_index(frame, feature_matrix, train_mask_np, k_neighbors=16):

    edges = []

    n_nodes = len(frame)

    train_indices = np.flatnonzero(train_mask_np)

    k = min(int(k_neighbors), len(train_indices))

    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')

    nn.fit(feature_matrix[train_indices])

    neigh = nn.kneighbors(feature_matrix, return_distance=False)

    for dst in range(n_nodes):

        for local_src in neigh[dst]:

            src = int(train_indices[local_src])

            edges.append((src, dst))



    # Temporal edges: only previous observation -> current observation within the same ticker.

    for _, g in frame.groupby('ticker', sort=False):

        idx = g.sort_values(['rating_date', '__split__', '__split_row_index__']).index.to_numpy()

        for pos in range(1, len(idx)):

            edges.append((int(idx[pos - 1]), int(idx[pos])))



    # Self loops preserve each node's own financial state.

    for i in range(n_nodes):

        edges.append((i, i))



    edge_df = pd.DataFrame(edges, columns=['src', 'dst']).drop_duplicates()

    edge_index_np = edge_df[['src', 'dst']].to_numpy(dtype=np.int64).T

    return torch.tensor(edge_index_np, dtype=torch.long, device=device), edge_df





edge_index, edge_df = build_edge_index(

    df,

    df[MODEL_FEATURES].values.astype(np.float32),

    df['__split__'].eq('train').values,

    k_neighbors=KNN_K,

)

print('Graph nodes:', len(df), '| edges:', edge_index.shape[1], '| KNN_K:', KNN_K)

display(edge_df.head())


# ----------------------------------------

# Visualization: graph structure

# Yeu cau: cell xay dung graph (cell 4) va cell du lieu (cell 2) phai chay truoc.

if 'edge_df' not in globals() or 'df' not in globals():

    raise RuntimeError('Khong tim thay edge_df hoac df. Hay chay cell xay dung graph truoc.')



import networkx as nx

from matplotlib.gridspec import GridSpec

from matplotlib.ticker import MaxNLocator

from matplotlib.lines import Line2D

from matplotlib.patches import Patch, FancyBboxPatch

import matplotlib.patheffects as pe



sns.set_theme(style='whitegrid', context='paper')

plt.rcParams.update({

    'font.family': 'DejaVu Sans',

    'axes.spines.top': False,

    'axes.spines.right': False,

})



# ── Palette (white background) ──────────────────────────────────────────────

BG        = 'white'

PANEL_BG  = '#f8f9fa'

GRID_CLR  = '#e9ecef'

TEXT_MAIN = '#212529'

TEXT_SUB  = '#6c757d'



LABEL_COLORS = {

    'Distressed': '#c0392b',

    'HY':         '#e67e22',

    'IG':         '#16a085',

}

LABEL_COLORS_LIGHT = {

    'Distressed': '#fadbd8',

    'HY':         '#fde8d8',

    'IG':         '#d0ece7',

}

KNN_COLOR  = '#3498db'

TEMP_COLOR = '#8e44ad'

SELF_COLOR = '#95a5a6'



SPLIT_COLORS = {'train': '#2980b9', 'val': '#e67e22', 'test': '#27ae60'}



# ── 1. Graph statistics ──────────────────────────────────────────────────────

n_nodes     = len(df)

n_edges     = len(edge_df)

split_counts = df['__split__'].value_counts().to_dict()



self_edges = edge_df[edge_df['src'] == edge_df['dst']]

temporal_edges = edge_df[

    (edge_df['src'] != edge_df['dst']) &

    edge_df.apply(

        lambda r: df.at[int(r['src']), 'ticker'] == df.at[int(r['dst']), 'ticker'], axis=1

    )

]

knn_edges = edge_df[

    (edge_df['src'] != edge_df['dst']) &

    ~edge_df.index.isin(temporal_edges.index)

]

avg_degree = (len(knn_edges) + len(temporal_edges)) * 2 / max(n_nodes, 1)



print(f'Nodes={n_nodes:,} | KNN={len(knn_edges):,} | Temporal={len(temporal_edges):,} | Self={len(self_edges):,} | AvgDeg={avg_degree:.2f}')



# ── 2. Build NetworkX subgraph ───────────────────────────────────────────────

MAX_VIZ_NODES = 150

rng_viz = np.random.default_rng(42)

sample_idx = np.arange(n_nodes) if n_nodes <= MAX_VIZ_NODES else rng_viz.choice(n_nodes, size=MAX_VIZ_NODES, replace=False)

sample_set  = set(sample_idx.tolist())

sorted_sample = sorted(sample_set)



sub_edges = edge_df[

    edge_df['src'].isin(sample_set) & edge_df['dst'].isin(sample_set) &

    (edge_df['src'] != edge_df['dst'])

]

remap = {old: new for new, old in enumerate(sorted_sample)}

G = nx.DiGraph()

G.add_nodes_from(range(len(sample_idx)))

for _, row in sub_edges.iterrows():

    G.add_edge(remap[int(row['src'])], remap[int(row['dst'])])



# Node metadata

id_to_raw_local  = {v: k for k, v in raw_to_id.items()} if 'raw_to_id' in globals() else {i: str(i) for i in range(n_classes)}

node_label_ids   = [int(df.at[old, TARGET_COL]) for old in sorted_sample]

node_label_names = [str(id_to_raw_local.get(lid, lid)) for lid in node_label_ids]

node_splits_list = [df.at[old, '__split__'] for old in sorted_sample]

node_colors      = [LABEL_COLORS.get(name, '#adb5bd') for name in node_label_names]

node_sizes       = [120 if sp == 'train' else (80 if sp == 'val' else 60) for sp in node_splits_list]



# Layout

temporal_pairs = set(zip(temporal_edges['src'].tolist(), temporal_edges['dst'].tolist()))

knn_ex, temp_ex = [], []

for s, d in G.edges():

    orig_s, orig_d = sorted_sample[s], sorted_sample[d]

    if (orig_s, orig_d) in temporal_pairs or (orig_d, orig_s) in temporal_pairs:

        temp_ex.append((s, d))

    else:

        knn_ex.append((s, d))



try:

    pos = nx.spring_layout(G, seed=42, k=0.65, iterations=80)

except Exception:

    pos = nx.random_layout(G, seed=42)



# ── 3. Figure & GridSpec ─────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 13), dpi=150, facecolor=BG)

fig.patch.set_facecolor(BG)



gs = GridSpec(

    3, 3,

    figure=fig,

    left=0.04, right=0.97, top=0.92, bottom=0.05,

    hspace=0.42, wspace=0.30,

    width_ratios=[1.6, 1, 1],

    height_ratios=[1, 1, 0.75],

)



# ── Panel A: Graph network (spans 3 rows, col 0) ─────────────────────────────

ax_graph = fig.add_subplot(gs[:, 0])

ax_graph.set_facecolor(PANEL_BG)

for spine in ax_graph.spines.values():

    spine.set_linewidth(0.8)

    spine.set_edgecolor('#dee2e6')



# KNN edges

nx.draw_networkx_edges(

    G, pos, edgelist=knn_ex, ax=ax_graph,

    edge_color=KNN_COLOR, alpha=0.18, arrows=False, width=0.7,

    style='solid',

)

# Temporal edges

nx.draw_networkx_edges(

    G, pos, edgelist=temp_ex, ax=ax_graph,

    edge_color=TEMP_COLOR, alpha=0.65, arrows=True,

    arrowstyle='-|>', arrowsize=10, width=1.2,

    connectionstyle='arc3,rad=0.08',

)

# Nodes — outer ring (white border effect)

nx.draw_networkx_nodes(

    G, pos, ax=ax_graph,

    node_color='white',

    node_size=[s + 30 for s in node_sizes],

    linewidths=0,

)

nx.draw_networkx_nodes(

    G, pos, ax=ax_graph,

    node_color=node_colors,

    node_size=node_sizes,

    linewidths=0.8,

    edgecolors='white',

    alpha=0.90,

)



n_shown = G.number_of_edges()

ax_graph.set_title(

    f'Credit Graph — Subgraph Sample\n'

    f'{len(sample_idx)} nodes · {n_shown} edges shown',

    fontsize=11, fontweight='bold', color=TEXT_MAIN, pad=10, loc='left',

)

ax_graph.axis('off')



# Legend

legend_handles = [

    Patch(facecolor=c, edgecolor='white', linewidth=0.8, label=lb)

    for lb, c in LABEL_COLORS.items() if lb in set(node_label_names)

]

legend_handles += [

    Line2D([0], [0], color=KNN_COLOR,  linewidth=1.8, alpha=0.8,  label='KNN similarity'),

    Line2D([0], [0], color=TEMP_COLOR, linewidth=1.8, alpha=0.9,  label='Temporal (same ticker)',

           marker='>', markersize=6, markerfacecolor=TEMP_COLOR),

]

legend_handles += [

    Line2D([0], [0], color=SPLIT_COLORS['train'], linewidth=0, marker='o',

           markersize=8, markerfacecolor=SPLIT_COLORS['train'], alpha=0.6, label='Train node'),

    Line2D([0], [0], color=SPLIT_COLORS['val'],   linewidth=0, marker='o',

           markersize=7, markerfacecolor=SPLIT_COLORS['val'],   alpha=0.6, label='Val node'),

    Line2D([0], [0], color=SPLIT_COLORS['test'],  linewidth=0, marker='o',

           markersize=6, markerfacecolor=SPLIT_COLORS['test'],  alpha=0.6, label='Test node'),

]

leg = ax_graph.legend(

    handles=legend_handles, loc='lower left',

    framealpha=0.95, facecolor='white', edgecolor='#dee2e6',

    fontsize=8, labelcolor=TEXT_MAIN, ncol=2,

    borderpad=0.8, handlelength=1.6, columnspacing=1.0,

)

leg.get_frame().set_linewidth(0.8)



# ── Panel B: Degree distribution (row 0, col 1-2) ────────────────────────────

ax_deg = fig.add_subplot(gs[0, 1])

ax_deg.set_facecolor(PANEL_BG)

degrees = [d for _, d in G.degree()]

ax_deg.hist(degrees, bins=25, color=KNN_COLOR, edgecolor='white', linewidth=0.5, alpha=0.82)

ax_deg.set_xlabel('Node Degree', color=TEXT_SUB, fontsize=8.5)

ax_deg.set_ylabel('Count', color=TEXT_SUB, fontsize=8.5)

ax_deg.set_title('Degree Distribution', fontsize=10, fontweight='semibold', color=TEXT_MAIN, loc='left')

ax_deg.tick_params(colors=TEXT_SUB, labelsize=7.5)

ax_deg.yaxis.set_major_locator(MaxNLocator(integer=True))

ax_deg.grid(True, axis='y', linestyle='--', alpha=0.5, color=GRID_CLR, linewidth=0.8)

ax_deg.spines['left'].set_color('#dee2e6')

ax_deg.spines['bottom'].set_color('#dee2e6')

mean_deg = np.mean(degrees)

ax_deg.axvline(mean_deg, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.8,

               label=f'Mean={mean_deg:.1f}')

ax_deg.legend(fontsize=7.5, framealpha=0.9, facecolor='white', edgecolor='#dee2e6')



# ── Panel C: Edge type breakdown (row 0, col 2) ───────────────────────────────

ax_pie = fig.add_subplot(gs[0, 2])

ax_pie.set_facecolor(PANEL_BG)

wedge_vals   = [len(knn_edges), len(temporal_edges), len(self_edges)]

wedge_lbls   = ['KNN', 'Temporal', 'Self-loop']

wedge_clrs   = [KNN_COLOR, TEMP_COLOR, SELF_COLOR]

wedges, texts, autotexts = ax_pie.pie(

    wedge_vals,

    labels=wedge_lbls,

    colors=wedge_clrs,

    autopct='%1.1f%%',

    startangle=90,

    pctdistance=0.75,

    textprops={'fontsize': 8, 'color': TEXT_MAIN},

    wedgeprops={'linewidth': 1.2, 'edgecolor': 'white'},

)

for at in autotexts:

    at.set_fontsize(7.5)

    at.set_color('white')

    at.set_fontweight('semibold')

ax_pie.set_title('Edge Type Breakdown', fontsize=10, fontweight='semibold', color=TEXT_MAIN, loc='left')



# ── Panel D: Class distribution per split (row 1, col 1-2) ───────────────────

ax_bar = fig.add_subplot(gs[1, 1:])

ax_bar.set_facecolor(PANEL_BG)



splits_order = ['train', 'val', 'test']

label_order  = [str(id_to_raw_local.get(i, i)) for i in range(n_classes)]

bar_w = 0.22

x_pos = np.arange(len(splits_order))



for li, lname in enumerate(label_order):

    counts = []

    for sp in splits_order:

        lid = raw_to_id.get(lname, li) if isinstance(lname, str) else lname

        counts.append(int((df.loc[df['__split__'] == sp, TARGET_COL] == lid).sum()))

    offset = (li - len(label_order) / 2 + 0.5) * bar_w

    bars = ax_bar.bar(

        x_pos + offset, counts, bar_w,

        label=str(lname),

        color=LABEL_COLORS.get(str(lname), '#adb5bd'),

        edgecolor='white', linewidth=0.8, alpha=0.88,

    )

    for bar_obj, cnt in zip(bars, counts):

        if cnt > 0:

            ax_bar.text(

                bar_obj.get_x() + bar_obj.get_width() / 2,

                bar_obj.get_height() + 0.3,

                str(cnt), ha='center', va='bottom',

                fontsize=7.5, color=TEXT_MAIN, fontweight='semibold',

            )



ax_bar.set_xticks(x_pos)

ax_bar.set_xticklabels([s.capitalize() for s in splits_order], fontsize=9, color=TEXT_MAIN)

ax_bar.tick_params(axis='y', labelsize=8, colors=TEXT_SUB)

ax_bar.set_ylabel('Count', fontsize=8.5, color=TEXT_SUB)

ax_bar.set_title('Class Distribution per Split', fontsize=10, fontweight='semibold', color=TEXT_MAIN, loc='left')

ax_bar.legend(

    framealpha=0.9, facecolor='white', edgecolor='#dee2e6',

    fontsize=8, labelcolor=TEXT_MAIN, loc='upper right',

)

ax_bar.grid(True, axis='y', linestyle='--', alpha=0.5, color=GRID_CLR, linewidth=0.8)

ax_bar.spines['left'].set_color('#dee2e6')

ax_bar.spines['bottom'].set_color('#dee2e6')

ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.15)



# ── Panel E: Summary stats table (row 2, col 1-2) ────────────────────────────

ax_stats = fig.add_subplot(gs[2, 1:])

ax_stats.set_facecolor(PANEL_BG)

ax_stats.axis('off')



stats_rows = [

    ['Metric',           'Value'],

    ['Total nodes',      f'{n_nodes:,}'],

    ['Total edges',      f'{n_edges:,}'],

    ['  KNN edges',      f'{len(knn_edges):,}'],

    ['  Temporal edges', f'{len(temporal_edges):,}'],

    ['  Self-loops',     f'{len(self_edges):,}'],

    ['Avg degree',       f'{avg_degree:.2f}'],

    ['KNN K',            f'{KNN_K}'],

    ['Classes',          f'{n_classes}'],

    ['Sectors',          f'{n_sectors}'],

]



col_widths = [0.55, 0.45]

row_h = 0.088

x_starts = [0.04, 0.59]



for ri, row_data in enumerate(stats_rows):

    y = 0.96 - ri * row_h

    is_header = ri == 0

    bg_color = '#e9ecef' if is_header else ('white' if ri % 2 == 0 else '#f8f9fa')

    rect = FancyBboxPatch(

        (0.01, y - row_h * 0.85), 0.98, row_h * 0.88,

        boxstyle='round,pad=0.005',

        facecolor=bg_color, edgecolor='#dee2e6', linewidth=0.5,

        transform=ax_stats.transAxes, clip_on=False,

    )

    ax_stats.add_patch(rect)

    for ci, (txt, xs) in enumerate(zip(row_data, x_starts)):

        ax_stats.text(

            xs, y - row_h * 0.35, txt,

            transform=ax_stats.transAxes,

            fontsize=8.5 if not is_header else 9,

            fontweight='bold' if is_header else ('semibold' if ci == 0 else 'normal'),

            color=TEXT_MAIN if not is_header else '#495057',

            va='center',

        )



ax_stats.set_title('Graph Summary', fontsize=10, fontweight='semibold', color=TEXT_MAIN, loc='left', pad=4)



# ── Main title ───────────────────────────────────────────────────────────────

fig.suptitle(

    'GAT Credit Graph — Structure Visualization',

    fontsize=14, fontweight='bold', color=TEXT_MAIN, y=0.965,

)



graph_viz_path = ARTIFACT_DIR / 'gat_graph_visualization.png'

fig.savefig(graph_viz_path, dpi=200, bbox_inches='tight', facecolor=BG)

plt.show()

print('Saved:', graph_viz_path)


# ----------------------------------------

def edge_softmax(scores, dst, num_nodes):

    # scores: [E, H], dst: [E]

    if hasattr(torch.Tensor, 'scatter_reduce_'):

        expanded = dst.view(-1, 1).expand(-1, scores.size(1))

        max_per_dst = torch.full((num_nodes, scores.size(1)), -1e9, device=scores.device, dtype=scores.dtype)

        max_per_dst.scatter_reduce_(0, expanded, scores, reduce='amax', include_self=True)

        exp_scores = torch.exp(scores - max_per_dst[dst])

        denom = torch.zeros((num_nodes, scores.size(1)), device=scores.device, dtype=scores.dtype)

        denom.index_add_(0, dst, exp_scores)

        return exp_scores / (denom[dst] + 1e-12)



    # Fallback for older PyTorch versions.

    attn = torch.zeros_like(scores)

    for node in torch.unique(dst):

        mask = dst == node

        attn[mask] = torch.softmax(scores[mask], dim=0)

    return attn





class SparseGATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, concat=True, negative_slope=0.2):

        super().__init__()

        self.heads = int(heads)

        self.out_dim = int(out_dim)

        self.concat = bool(concat)

        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)

        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))

        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))

        self.bias = nn.Parameter(torch.zeros(out_dim * heads if concat else out_dim))

        self.dropout = nn.Dropout(dropout)

        self.negative_slope = negative_slope

        self.reset_parameters()



    def reset_parameters(self):

        nn.init.xavier_uniform_(self.lin.weight)

        nn.init.xavier_uniform_(self.attn_src)

        nn.init.xavier_uniform_(self.attn_dst)

        nn.init.zeros_(self.bias)



    def forward(self, x, edge_index):

        src, dst = edge_index

        n_nodes = x.size(0)

        h = self.lin(x).view(n_nodes, self.heads, self.out_dim)

        h_src = h[src]

        h_dst = h[dst]

        scores = (h_src * self.attn_src.unsqueeze(0)).sum(-1) + (h_dst * self.attn_dst.unsqueeze(0)).sum(-1)

        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)

        alpha = self.dropout(edge_softmax(scores, dst, n_nodes))

        messages = h_src * alpha.unsqueeze(-1)

        out = torch.zeros((n_nodes, self.heads, self.out_dim), device=x.device, dtype=x.dtype)

        out.index_add_(0, dst, messages)

        if self.concat:

            out = out.reshape(n_nodes, self.heads * self.out_dim)

        else:

            out = out.mean(dim=1)

        return out + self.bias





class CreditGAT(nn.Module):

    def __init__(self, n_features, n_classes, n_sectors, hidden=64, heads=4, dropout=0.25):

        super().__init__()

        self.last_y_emb = nn.Embedding(n_classes, 16)

        self.sector_emb = nn.Embedding(n_sectors, 8)

        in_dim = n_features + 16 + 8

        self.input_norm = nn.LayerNorm(in_dim)

        self.gat1 = SparseGATLayer(in_dim, hidden, heads=heads, dropout=dropout, concat=True)

        self.gat2 = SparseGATLayer(hidden * heads, hidden, heads=2, dropout=dropout, concat=False)

        self.head = nn.Sequential(

            nn.LayerNorm(hidden),

            nn.Dropout(dropout),

            nn.Linear(hidden, n_classes),

        )



    def forward(self, x, last_y, sector_id, edge_index, return_embeddings=False):

        base = torch.cat([x, self.last_y_emb(last_y), self.sector_emb(sector_id)], dim=1)

        base = self.input_norm(base)

        h = F.elu(self.gat1(base, edge_index))

        h = F.elu(self.gat2(h, edge_index))

        logits = self.head(h)

        if return_embeddings:

            return logits, h

        return logits





class RampedFocalOrdinalLoss(nn.Module):

    def __init__(

        self,

        n_classes,

        class_weights=None,

        ce_weight=1.0,

        focal_gamma=0.25,

        focal_weight=0.03,

        ordinal_weight=0.005,

        warmup_epochs=100,

    ):

        super().__init__()

        self.n_classes = int(n_classes)

        self.ce_weight = float(ce_weight)

        self.focal_gamma = float(focal_gamma)

        self.focal_weight = float(focal_weight)

        self.ordinal_weight = float(ordinal_weight)

        self.warmup_epochs = max(1, int(warmup_epochs))

        if class_weights is None:

            self.register_buffer('class_weights', None)

        else:

            self.register_buffer('class_weights', class_weights.detach().float().clone())

        self.register_buffer('class_positions', torch.arange(self.n_classes, dtype=torch.float32))



    def ramp(self, epoch=None):

        if epoch is None:

            return 1.0

        return min(1.0, max(0.0, float(epoch) / float(self.warmup_epochs)))



    def forward(self, logits, targets, epoch=None):

        targets = targets.long()

        log_probs = F.log_softmax(logits, dim=1)

        probs = log_probs.exp()



        ce_per_sample = F.nll_loss(log_probs, targets, weight=self.class_weights, reduction='none')

        ce_loss = ce_per_sample.mean()



        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-8)

        focal_loss = ((1.0 - pt) ** self.focal_gamma * ce_per_sample).mean()



        distances = torch.abs(self.class_positions.to(logits.device).view(1, -1) - targets.float().view(-1, 1))

        ordinal_loss = (probs * distances).sum(dim=1)

        if self.n_classes > 1:

            ordinal_loss = ordinal_loss / float(self.n_classes - 1)

        ordinal_loss = ordinal_loss.mean()



        ramp = self.ramp(epoch)

        return self.ce_weight * ce_loss + ramp * (self.focal_weight * focal_loss + self.ordinal_weight * ordinal_loss)





model = CreditGAT(

    n_features=len(MODEL_FEATURES),

    n_classes=n_classes,

    n_sectors=n_sectors,

    hidden=64,

    heads=4,

    dropout=0.25,

).to(device)



LOSS_CONFIG = {

    'ce_weight': 1.0,

    'focal_gamma': 0.25,

    'focal_weight': 0.03,

    'ordinal_weight': 0.005,

    'warmup_epochs': 100,

    'use_class_weights': False,

}

criterion = RampedFocalOrdinalLoss(

    n_classes=n_classes,

    class_weights=train_class_weights if LOSS_CONFIG['use_class_weights'] else None,

    ce_weight=LOSS_CONFIG['ce_weight'],

    focal_gamma=LOSS_CONFIG['focal_gamma'],

    focal_weight=LOSS_CONFIG['focal_weight'],

    ordinal_weight=LOSS_CONFIG['ordinal_weight'],

    warmup_epochs=LOSS_CONFIG['warmup_epochs'],

).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

print('Loss config:', LOSS_CONFIG)

print(model)


# ----------------------------------------

history = {

    'epoch': [],

    'train_Loss': [], 'val_Loss': [],

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

}



best_val_score = -1.0

best_state = None

patience, no_improve = 100, 0

max_epochs = 100



for epoch in range(1, max_epochs + 1):

    model.train()

    optimizer.zero_grad(set_to_none=True)

    logits = model(x_all, last_y_all, sector_all, edge_index)

    loss = criterion(logits[train_mask], y_all[train_mask], epoch=epoch)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()



    model.eval()

    with torch.no_grad():

        logits_eval = model(x_all, last_y_all, sector_all, edge_index)

        train_loss = criterion(logits_eval[train_mask], y_all[train_mask], epoch=epoch).item()

        val_loss = criterion(logits_eval[val_mask], y_all[val_mask], epoch=epoch).item()

        tr, _, _, _ = evaluate_logits(logits_eval, train_mask)

        va, _, _, _ = evaluate_logits(logits_eval, val_mask)



    history['epoch'].append(epoch)

    history['train_Loss'].append(float(train_loss))

    history['val_Loss'].append(float(val_loss))

    for metric_name in ['Accuracy', 'Macro_F1', 'Class0_Precision', 'Class0_Recall', 'Class0_F1', 'Class0_F2', 'ChgAcc', 'Ordinal_MAE', 'AUC', 'QWK']:

        history[f'train_{metric_name}'].append(float(tr[metric_name]) if not (isinstance(tr[metric_name], float) and tr[metric_name] != tr[metric_name]) else float('nan'))

        history[f'val_{metric_name}'].append(float(va[metric_name]) if not (isinstance(va[metric_name], float) and va[metric_name] != va[metric_name]) else float('nan'))



    val_selection_score = selection_score(va)

    print(

        f"Epoch {epoch:03d} | TrLoss {train_loss:.4f} | VaLoss {val_loss:.4f} | "

        f"VaAcc {va['Accuracy']:.4f} | VaF1 {va['Macro_F1']:.4f} | "

        f"VaC0R {va['Class0_Recall']:.4f} | VaC0F2 {va['Class0_F2']:.4f} | "

        f"VaQWK {va['QWK']:.4f} | VaScore {val_selection_score:.4f}"

    )



    if val_selection_score > best_val_score + 1e-4:

        best_val_score = val_selection_score

        best_state = copy.deepcopy(model.state_dict())

        no_improve = 0

    else:

        no_improve += 1

        if no_improve >= patience:

            print('Early stopping.')

            break



if best_state is not None:

    model.load_state_dict(best_state)



model.eval()

with torch.no_grad():

    final_logits, node_embeddings = model(x_all, last_y_all, sector_all, edge_index, return_embeddings=True)



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



report = pd.DataFrame([

    {'Split': 'Val_RawArgmax', 'Class0_Threshold': np.nan, **val_raw_metrics},

    {'Split': 'Test_RawArgmax', 'Class0_Threshold': np.nan, **test_raw_metrics},

    {'Split': 'Val_Class0Calibrated', 'Class0_Threshold': class0_threshold, **val_metrics},

    {'Split': 'Test_Class0Calibrated', 'Class0_Threshold': class0_threshold, **test_metrics},

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

    **{f'val_selected_{k}': v for k, v in class0_threshold_selected.items() if isinstance(v, (int, float, np.integer, np.floating))},

}]).to_csv(threshold_summary_path, index=False, encoding='utf-8-sig')



print('Selected class 0 threshold:', class0_threshold)

print('Saved:', metrics_path)

print('Saved:', history_path)

print('Saved:', threshold_sweep_path)

print('Saved:', threshold_summary_path)


# ----------------------------------------

id_to_raw_local = {v: k for k, v in raw_to_id.items()} if 'raw_to_id' in globals() else {i: i for i in range(n_classes)}

class_labels = [str(id_to_raw_local.get(i, i)) for i in range(n_classes)]

label_ids = list(range(n_classes))



cm = confusion_matrix(y_test, y_test_pred, labels=label_ids)

cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)



plt.figure(figsize=(6, 5), dpi=160)

sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)

threshold_title = f' | class0 threshold={class0_threshold:.2f}' if class0_threshold is not None else ''

plt.title(f'GAT Confusion Matrix (Test){threshold_title}')

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.tight_layout()

cm_plot_path = ARTIFACT_DIR / 'gat_test_confusion_matrix.png'

plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')

plt.show()



display(cm_df)

print('Classification report (test set, class 0 calibrated):')

print(classification_report(

    y_test,

    y_test_pred,

    labels=label_ids,

    target_names=class_labels,

    digits=4,

    zero_division=0,

))



if 'test_raw_metrics' in globals():

    print('Raw argmax class 0 metrics:')

    print({

        'Class0_Precision': round(test_raw_metrics['Class0_Precision'], 4),

        'Class0_Recall': round(test_raw_metrics['Class0_Recall'], 4),

        'Class0_F1': round(test_raw_metrics['Class0_F1'], 4),

        'Accuracy': round(test_raw_metrics['Accuracy'], 4),

    })

    print('Calibrated class 0 metrics:')

    print({

        'Class0_Precision': round(test_metrics['Class0_Precision'], 4),

        'Class0_Recall': round(test_metrics['Class0_Recall'], 4),

        'Class0_F1': round(test_metrics['Class0_F1'], 4),

        'Accuracy': round(test_metrics['Accuracy'], 4),

    })



cls_report_df = pd.DataFrame(

    classification_report(

        y_test,

        y_test_pred,

        labels=label_ids,

        target_names=class_labels,

        output_dict=True,

        zero_division=0,

    )

).transpose()



cm_csv_path = ARTIFACT_DIR / 'gat_test_confusion_matrix.csv'

cls_csv_path = ARTIFACT_DIR / 'gat_test_classification_report.csv'

cm_df.to_csv(cm_csv_path, encoding='utf-8-sig')

cls_report_df.to_csv(cls_csv_path, encoding='utf-8-sig')



plt.figure(figsize=(10, 8))

y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

for i in range(n_classes):

    fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_proba[:, i])

    roc_auc_val = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {class_labels[i]} (area = {roc_auc_val:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('GAT - Receiver Operating Characteristic (ROC) - Multiclass')

plt.legend(loc='lower right')

plt.grid(alpha=0.3)

roc_plot_path = ARTIFACT_DIR / 'gat_test_roc_curves.png'

plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')

plt.show()



print('Saved:', cm_plot_path)

print('Saved:', cm_csv_path)

print('Saved:', cls_csv_path)

print('Saved:', roc_plot_path)


# ----------------------------------------

# Export prediction CSVs with the same contract as LSTM for DMF/DCS.



def prediction_frame(split_name, split_mask, y_true, proba, pred=None, class0_threshold=None):

    pred = predict_with_class0_threshold(proba, class0_threshold=class0_threshold) if pred is None else np.asarray(pred).astype(int)

    raw_pred = np.argmax(proba, axis=1).astype(int)

    conf = proba[np.arange(len(pred)), pred]

    rows = df.loc[split_mask.detach().cpu().numpy(), ['row_id', 'ticker', 'company_name', 'rating_date']].copy().reset_index(drop=True)

    rows.insert(0, 'split', split_name)

    rows['rating_date'] = pd.to_datetime(rows['rating_date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

    rows['true_label'] = y_true.astype(int)

    rows['true_label_name'] = [str(id_to_raw_local.get(int(y), y)) for y in y_true]

    rows['pred_label'] = pred

    rows['pred_label_name'] = [str(id_to_raw_local.get(int(y), y)) for y in pred]

    rows['raw_pred_label'] = raw_pred

    rows['raw_pred_label_name'] = [str(id_to_raw_local.get(int(y), y)) for y in raw_pred]

    rows['class0_threshold'] = np.nan if class0_threshold is None else float(class0_threshold)

    rows['confidence'] = conf.astype(float)

    rows['raw_confidence'] = np.max(proba, axis=1).astype(float)

    for cls_idx in range(proba.shape[1]):

        rows[f'prob_{cls_idx}'] = proba[:, cls_idx].astype(float)

    return rows





gat_val_predictions = prediction_frame('val', val_mask, y_val, val_proba, pred=y_val_pred, class0_threshold=class0_threshold)

gat_test_predictions = prediction_frame('test', test_mask, y_test, test_proba, pred=y_test_pred, class0_threshold=class0_threshold)



val_csv = DMF_ARTIFACT_DIR / 'gat_val_predictions.csv'

test_csv = DMF_ARTIFACT_DIR / 'gat_test_predictions.csv'

gat_val_predictions.to_csv(val_csv, index=False, encoding='utf-8-sig')

gat_test_predictions.to_csv(test_csv, index=False, encoding='utf-8-sig')



np.save(DMF_ARTIFACT_DIR / 'gat_val_embeddings.npy', node_embeddings[val_mask].detach().cpu().numpy())

np.save(DMF_ARTIFACT_DIR / 'gat_test_embeddings.npy', node_embeddings[test_mask].detach().cpu().numpy())

np.save(ARTIFACT_DIR / 'gat_val_proba.npy', val_proba.astype(np.float32))

np.save(ARTIFACT_DIR / 'gat_test_proba.npy', test_proba.astype(np.float32))

np.save(ARTIFACT_DIR / 'gat_y_val.npy', y_val.astype(int))

np.save(ARTIFACT_DIR / 'gat_y_test.npy', y_test.astype(int))



label_contract.to_csv(DMF_ARTIFACT_DIR / 'label_mapping.csv', index=False, encoding='utf-8-sig')

print(f'[OK] Saved DMF val CSV  -> {val_csv}')

print(f'[OK] Saved DMF test CSV -> {test_csv}')

print(gat_test_predictions.head())


# ----------------------------------------

# Diagnostics: class 0 threshold trade-off and false negatives.

if 'class0_threshold_sweep' not in globals() or class0_threshold_sweep.empty:

    raise RuntimeError('Chua co class0_threshold_sweep. Hay chay lai cell huan luyen truoc.')



display_cols = [

    'class0_threshold', 'Accuracy', 'Macro_F1',

    'Class0_Precision', 'Class0_Recall', 'Class0_F1', 'Class0_F2',

]

threshold_view = class0_threshold_sweep[display_cols].sort_values(

    ['Class0_F2', 'Accuracy', 'Macro_F1'],

    ascending=False,

).head(12)

display(threshold_view)



class0_tradeoff_path = ARTIFACT_DIR / 'gat_class0_threshold_top_candidates.csv'

threshold_view.to_csv(class0_tradeoff_path, index=False, encoding='utf-8-sig')



test_rows = df.loc[test_mask.detach().cpu().numpy(), [

    'row_id', 'ticker', 'company_name', 'rating_date', 'sector',

    TARGET_COL, 'last_y',

]].copy().reset_index(drop=True)

test_rows['pred_label'] = y_test_pred.astype(int)

test_rows['raw_pred_label'] = y_test_raw_pred.astype(int)

for cls_idx in range(test_proba.shape[1]):

    test_rows[f'prob_{cls_idx}'] = test_proba[:, cls_idx]



class0_error_mask = (test_rows[TARGET_COL].astype(int) == CLASS0_LABEL_ID) & (test_rows['pred_label'] != CLASS0_LABEL_ID)

class0_false_negatives = test_rows.loc[class0_error_mask].sort_values('prob_0', ascending=False)

class0_false_negative_path = ARTIFACT_DIR / 'gat_class0_false_negatives.csv'

class0_false_negatives.to_csv(class0_false_negative_path, index=False, encoding='utf-8-sig')



print('Class 0 confusion by last_y:')

display(pd.crosstab(

    test_rows.loc[test_rows[TARGET_COL].astype(int).eq(CLASS0_LABEL_ID), 'last_y'],

    test_rows.loc[test_rows[TARGET_COL].astype(int).eq(CLASS0_LABEL_ID), 'pred_label'],

    rownames=['last_y'],

    colnames=['pred_label'],

))



print('Top class 0 false negatives by prob_0:')

display(class0_false_negatives.head(20))



print('Saved:', class0_tradeoff_path)

print('Saved:', class0_false_negative_path)


# ----------------------------------------

# Visualization: training curves

if 'history_df' not in globals():

    raise RuntimeError('Khong tim thay history_df. Hay chay lai cell huan luyen truoc.')



from matplotlib.ticker import MultipleLocator



sns.set_theme(style='whitegrid', context='paper')

metrics = ['Loss', 'Accuracy', 'Macro_F1', 'Class0_Recall', 'Class0_F2', 'QWK']

required_cols = [f'train_{m}' for m in metrics] + [f'val_{m}' for m in metrics]

missing = [c for c in required_cols if c not in history_df.columns]

if missing:

    raise RuntimeError(f'Thieu cot trong history_df: {missing}. Hay chay lai cell huan luyen.')



fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=160, constrained_layout=True)

axes = axes.ravel()

max_epoch = int(history_df['epoch'].max())



for ax, metric in zip(axes, metrics):

    ax.plot(history_df['epoch'], history_df[f'train_{metric}'], label='Train', linewidth=1.8, color='#1f77b4')

    ax.plot(history_df['epoch'], history_df[f'val_{metric}'], label='Validation', linewidth=1.8, color='#d62728')

    if metric == 'Loss':

        best_epoch = int(history_df.loc[history_df['val_Loss'].idxmin(), 'epoch'])

        ax.axvline(best_epoch, color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.8, label='Best val loss')

    ax.set_title(metric, fontsize=11, fontweight='semibold')

    ax.set_xlabel('Epoch')

    ax.set_xlim(0, max_epoch)

    ax.xaxis.set_major_locator(MultipleLocator(10))

    ax.set_ylabel(metric)

    ax.grid(True, linestyle='--', alpha=0.35)

    ax.legend(frameon=True, fontsize=9)



fig.suptitle('GAT Training Curves with Class 0 Monitoring', fontsize=13, fontweight='bold')

curve_path = ARTIFACT_DIR / 'gat_training_curves.png'

fig.savefig(curve_path, dpi=300, bbox_inches='tight')

plt.show()



print('Saved:', curve_path)


# ----------------------------------------

if 'history_df' not in globals():

    raise RuntimeError('Khong tim thay history_df. Hay chay lai cell huan luyen truoc.')



history_df = history_df.copy()

history_df['val_SelectionScore'] = history_df.apply(

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

best_epoch_idx = history_df['val_SelectionScore'].idxmax()

best_epoch = int(history_df.loc[best_epoch_idx, 'epoch']) if 'epoch' in history_df.columns else int(best_epoch_idx) + 1



row = history_df.loc[best_epoch_idx]

best_train_loss = float(row['train_Loss'])

best_val_loss = float(row['val_Loss'])

best_train_acc = float(row['train_Accuracy'])

best_val_acc = float(row['val_Accuracy'])

best_val_class0_recall = float(row['val_Class0_Recall'])

best_val_class0_f2 = float(row['val_Class0_F2'])



print('Best metrics (by class-0-aware validation score):')

print(f'Train Loss:       {best_train_loss:.6f} @ epoch {best_epoch}')

print(f'Val Loss:         {best_val_loss:.6f} @ epoch {best_epoch}')

print(f'Train Acc:        {best_train_acc:.6f} @ epoch {best_epoch}')

print(f'Val Acc:          {best_val_acc:.6f} @ epoch {best_epoch}')

print(f'Val Class0 Recall:{best_val_class0_recall:.6f} @ epoch {best_epoch}')

print(f'Val Class0 F2:    {best_val_class0_f2:.6f} @ epoch {best_epoch}')



summary_df = pd.DataFrame([

    {

        'epoch': best_epoch,

        'train_loss': best_train_loss,

        'val_loss': best_val_loss,

        'train_acc': best_train_acc,

        'val_acc': best_val_acc,

        'val_class0_recall': best_val_class0_recall,

        'val_class0_f2': best_val_class0_f2,

        'selected_class0_threshold': class0_threshold if 'class0_threshold' in globals() else np.nan,

    }

])



training_summary_path = ARTIFACT_DIR / 'gat_training_summary.csv'

summary_df.to_csv(training_summary_path, index=False, encoding='utf-8-sig')

display(summary_df)

print('Saved:', training_summary_path)


# ----------------------------------------
