class SparseGraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.20):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.neigh_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_lin.weight)
        nn.init.xavier_uniform_(self.neigh_lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, h, sparse_adj):
        # sparse_adj[dst, src] da row-normalized; sparse.mm gom thong tin lang gieng vao node dst.
        neigh = torch.sparse.mm(sparse_adj, h)
        return self.self_lin(h) + self.neigh_lin(neigh) + self.bias


class SparseCreditGraphBaseline(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        n_sectors,
        n_nodes,
        edge_index,
        edge_weight,
        hidden=96,
        num_layers=3,
        dropout=0.25,
        last_y_emb_dim=8,
        sector_emb_dim=8,
        context_dropout=0.15,
        persistence_prior_scale=0.0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.context_dropout = float(context_dropout)
        self.context_unknown_id = int(n_classes)
        self.persistence_prior_scale = float(persistence_prior_scale)
        self.last_y_emb_dim = int(last_y_emb_dim)
        self.last_y_emb = nn.Embedding(n_classes + 1, last_y_emb_dim)
        self.sector_emb = nn.Embedding(n_sectors, sector_emb_dim)
        in_dim = int(n_features) + int(last_y_emb_dim) + int(sector_emb_dim)
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            SparseGraphSAGELayer(hidden, hidden, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(int(num_layers))])
        self.layer_dropout = nn.Dropout(dropout)
        readout_dim = hidden * (int(num_layers) + 1)
        self.head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.register_buffer('edge_index', edge_index.detach().long().clone())
        self.register_buffer('edge_weight', edge_weight.detach().float().clone())

    def build_sparse_adj(self, edge_index=None, edge_weight=None):
        edge_index = self.edge_index if edge_index is None else edge_index.long()
        edge_weight = self.edge_weight if edge_weight is None else edge_weight.float()
        return torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            size=(self.n_nodes, self.n_nodes),
            device=edge_weight.device,
        ).coalesce()

    def encode_context(self, x, last_y, sector_id, context_mask_prob=0.0):
        if self.training and context_mask_prob > 0.0:
            mask = torch.rand(last_y.shape, device=last_y.device) < float(context_mask_prob)
            last_y = last_y.masked_fill(mask, self.context_unknown_id)
        last_y_emb = self.last_y_emb(last_y)
        if self.training and self.context_dropout > 0:
            last_y_emb = F.dropout(last_y_emb, p=self.context_dropout, training=True)
        h = torch.cat([x, last_y_emb, self.sector_emb(sector_id)], dim=1)
        h = self.input_norm(h)
        return self.input_proj(h)

    def forward(self, x, last_y, sector_id, edge_index=None, edge_weight=None, return_embeddings=False, context_mask_prob=0.0):
        if x.size(0) != self.n_nodes:
            raise ValueError('SparseCreditGraphBaseline yeu cau full-batch x_all dung so node da build graph.')
        sparse_adj = self.build_sparse_adj(edge_index=edge_index, edge_weight=edge_weight)
        h = self.encode_context(x, last_y, sector_id, context_mask_prob=context_mask_prob)
        states = [h]
        for layer, norm in zip(self.layers, self.norms):
            h_next = F.gelu(layer(h, sparse_adj))
            h = norm(h + h_next)
            h = self.layer_dropout(h)
            states.append(h)
        readout = torch.cat(states, dim=1)
        logits = self.head(readout)
        if self.persistence_prior_scale > 0:
            valid_last_y = last_y.clamp(min=0, max=logits.size(1) - 1)
            prior = torch.zeros_like(logits)
            prior.scatter_(1, valid_last_y.view(-1, 1), self.persistence_prior_scale)
            logits = logits + prior
        if return_embeddings:
            return logits, readout
        return logits


CreditGAT = SparseCreditGraphBaseline




model = CreditGAT(
    n_features=len(MODEL_FEATURES),
    n_classes=n_classes,
    n_sectors=n_sectors,
    n_nodes=x_all.size(0),
    edge_index=edge_index,
    edge_weight=edge_weight,
    hidden=SPARSE_GRAPH_HIDDEN,
    num_layers=SPARSE_GRAPH_LAYERS,
    dropout=0.28,
    last_y_emb_dim=LAST_Y_EMB_DIM,
    sector_emb_dim=SECTOR_EMB_DIM,
    context_dropout=0.16,
    persistence_prior_scale=PERSISTENCE_PRIOR_SCALE,
).to(device)

LOSS_CONFIG = {'protocol': LOSS_PROTOCOL, 'ordinal_lambda': ORDINAL_LAMBDA}
criterion = build_loss(
    protocol=LOSS_PROTOCOL,
    ordinal_lambda=ORDINAL_LAMBDA,
).to(device)
OPTIMIZER_CONFIG = {
    'lr': 1.0e-3,
    'weight_decay': 5e-5,
    'plateau_factor': 0.60,
    'plateau_patience': 8,
    'min_lr': 1e-5,
}
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=OPTIMIZER_CONFIG['lr'],
    weight_decay=OPTIMIZER_CONFIG['weight_decay'],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=OPTIMIZER_CONFIG['plateau_factor'],
    patience=OPTIMIZER_CONFIG['plateau_patience'],
    min_lr=OPTIMIZER_CONFIG['min_lr'],
)


def summarize_sparse_graph(edge_df):
    n_nodes = int(len(df))
    non_self = edge_df[edge_df['dst'] != edge_df['src']]
    by_type = edge_df['edge_type'].str.get_dummies(sep='+').sum().sort_values(ascending=False)
    return {
        'nodes': n_nodes,
        'edges': int(len(edge_df)),
        'non_self_edges': int(len(non_self)),
        'avg_in_degree_non_self': round(float(len(non_self) / max(1, n_nodes)), 4),
        'edge_density': round(float(len(non_self) / max(1, n_nodes * max(1, n_nodes - 1))), 8),
        'edge_types': by_type.astype(int).to_dict(),
    }


print('Sparse graph config:', sparse_graph_contract)
print('Loss config:', LOSS_CONFIG)
print('Optimizer config:', OPTIMIZER_CONFIG)
print('Initial sparse graph stats:', summarize_sparse_graph(edge_df))
print(model)
