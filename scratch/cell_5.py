# Sparse row-graph configuration.
# Node la moi dong rating; edge la quan he thua giua cac dong doanh nghiep-thoi diem.
# Edge khong dung label: kNN theo financial features trong cung sector + temporal ticker edges + self-loops.

LAST_Y_EMB_DIM = 8
SECTOR_EMB_DIM = 8
SPARSE_GRAPH_HIDDEN = 96
SPARSE_GRAPH_LAYERS = 3
SPARSE_GRAPH_KNN_K = 12
SPARSE_GRAPH_MIN_SIMILARITY = 0.05
SPARSE_GRAPH_TEMPORAL_WEIGHT = 1.25
SPARSE_GRAPH_SELF_LOOP_WEIGHT = 1.0
PERSISTENCE_PRIOR_SCALE = 2.0


def _add_sparse_edge(edge_store, dst, src, weight, edge_type):
    key = (int(dst), int(src))
    payload = edge_store.setdefault(key, {'raw_weight': 0.0, 'edge_types': set()})
    payload['raw_weight'] = max(float(payload['raw_weight']), float(weight))
    payload['edge_types'].add(str(edge_type))


def build_sparse_credit_graph(frame, feature_cols, knn_k=12, min_similarity=0.05):
    frame = frame.reset_index(drop=True).copy()
    x_np = frame[feature_cols].to_numpy(dtype=np.float32)
    n_nodes = len(frame)
    edge_store = {}

    for node_idx in range(n_nodes):
        _add_sparse_edge(
            edge_store,
            node_idx,
            node_idx,
            SPARSE_GRAPH_SELF_LOOP_WEIGHT,
            'self_loop',
        )

    for _, sector_index in frame.groupby('sector_id', sort=False).groups.items():
        sector_nodes = np.asarray(list(sector_index), dtype=int)
        if sector_nodes.size <= 1:
            continue
        n_neighbors = min(int(knn_k) + 1, int(sector_nodes.size))
        nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        sector_features = x_np[sector_nodes]
        nn_index.fit(sector_features)
        distances, neighbors = nn_index.kneighbors(sector_features, return_distance=True)
        for local_dst, (dist_row, neigh_row) in enumerate(zip(distances, neighbors)):
            dst = int(sector_nodes[local_dst])
            for distance, local_src in zip(dist_row[1:], neigh_row[1:]):
                src = int(sector_nodes[int(local_src)])
                similarity = 1.0 - float(distance)
                weight = max(float(min_similarity), similarity)
                _add_sparse_edge(edge_store, dst, src, weight, 'sector_feature_knn')
                _add_sparse_edge(edge_store, src, dst, weight, 'sector_feature_knn')

    temporal_sort_cols = ['rating_date', '__split__', '__split_row_index__']
    for _, ticker_rows in frame.groupby('ticker', sort=False):
        ordered = ticker_rows.sort_values(temporal_sort_cols, kind='mergesort')
        row_indices = ordered.index.to_numpy(dtype=int)
        if row_indices.size <= 1:
            continue
        for prev_idx, next_idx in zip(row_indices[:-1], row_indices[1:]):
            _add_sparse_edge(edge_store, next_idx, prev_idx, SPARSE_GRAPH_TEMPORAL_WEIGHT, 'ticker_temporal_prev')
            _add_sparse_edge(edge_store, prev_idx, next_idx, SPARSE_GRAPH_TEMPORAL_WEIGHT, 'ticker_temporal_next')

    rows = []
    for (dst, src), payload in edge_store.items():
        rows.append({
            'dst': int(dst),
            'src': int(src),
            'dst_row_id': str(frame.loc[dst, 'row_id']),
            'src_row_id': str(frame.loc[src, 'row_id']),
            'dst_split': str(frame.loc[dst, '__split__']),
            'src_split': str(frame.loc[src, '__split__']),
            'dst_ticker': str(frame.loc[dst, 'ticker']),
            'src_ticker': str(frame.loc[src, 'ticker']),
            'edge_type': '+'.join(sorted(payload['edge_types'])),
            'raw_weight': float(payload['raw_weight']),
        })
    edge_df = pd.DataFrame(rows)
    edge_df = edge_df.sort_values(['dst', 'src']).reset_index(drop=True)

    dst = edge_df['dst'].to_numpy(dtype=np.int64)
    src = edge_df['src'].to_numpy(dtype=np.int64)
    raw_weight = edge_df['raw_weight'].to_numpy(dtype=np.float32)
    row_sum = np.bincount(dst, weights=raw_weight, minlength=n_nodes).astype(np.float32)
    norm_weight = raw_weight / np.maximum(row_sum[dst], 1e-12)
    edge_df['norm_weight'] = norm_weight.astype(float)

    edge_index = torch.tensor(np.vstack([dst, src]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(norm_weight, dtype=torch.float32, device=device)
    return edge_index, edge_weight, edge_df


edge_index, edge_weight, edge_df = build_sparse_credit_graph(
    df,
    MODEL_FEATURES,
    knn_k=SPARSE_GRAPH_KNN_K,
    min_similarity=SPARSE_GRAPH_MIN_SIMILARITY,
)

sparse_graph_contract = {
    'baseline': 'Sparse row graph / pure PyTorch GraphSAGE-style message passing',
    'node_unit': 'one rating row/company observation',
    'node_count': int(len(df)),
    'edge_count': int(edge_index.size(1)),
    'edge_sources': {
        'sector_feature_knn': int(edge_df['edge_type'].str.contains('sector_feature_knn').sum()),
        'ticker_temporal': int(edge_df['edge_type'].str.contains('ticker_temporal').sum()),
        'self_loop': int(edge_df['edge_type'].str.contains('self_loop').sum()),
    },
    'knn_scope': 'within sector, features only, no target labels',
    'knn_k': SPARSE_GRAPH_KNN_K,
    'min_similarity_weight': SPARSE_GRAPH_MIN_SIMILARITY,
    'temporal_weight': SPARSE_GRAPH_TEMPORAL_WEIGHT,
    'normalization': 'row-normalized incoming weights: adj[dst, src]',
    'message_passing': 'torch.sparse.mm(adj, node_embeddings)',
    'hidden_dim': SPARSE_GRAPH_HIDDEN,
    'num_layers': SPARSE_GRAPH_LAYERS,
    'persistence_prior_scale': PERSISTENCE_PRIOR_SCALE,
}
print('Sparse graph contract:', sparse_graph_contract)
print('Sparse graph edge sample:')
display(edge_df.head(10))
