class FuzzyLayer(nn.Module):
    """Gaussian fuzzy membership expansion with configurable MFs per feature."""
    def __init__(self, input_features, n_mfs=5, init_sigma=1.0):
        super().__init__()
        self.input_features = input_features
        self.n_mfs = n_mfs
        self.centers = nn.Parameter(torch.linspace(-1.0, 1.0, n_mfs).repeat(input_features, 1))
        self.log_sigma = nn.Parameter(torch.full((input_features, n_mfs), math.log(init_sigma)))

    def forward(self, x):
        # x: (B, T, F)
        x_exp = x.unsqueeze(-1)
        centers = self.centers.unsqueeze(0).unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0).unsqueeze(0).clamp(min=1e-3)
        memberships = torch.exp(-0.5 * ((x_exp - centers) / sigma) ** 2)
        return memberships.reshape(x.shape[0], x.shape[1], self.input_features * self.n_mfs)


class TemporalSelfAttentionBlock(nn.Module):
    """Pre-norm Transformer block with learnable relative positional bias."""
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, max_relative_position=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.max_relative_position = int(max_relative_position)
        self.relative_bias = nn.Parameter(torch.zeros(2 * self.max_relative_position + 1))

    def _relative_bias_matrix(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device)
        rel = pos[None, :] - pos[:, None]
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        rel = rel + self.max_relative_position
        return self.relative_bias[rel].to(dtype=dtype)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_bias = self._relative_bias_matrix(x.size(1), x.device, x.dtype)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AttentivePool(nn.Module):
    """Learnable weighted pooling over temporal axis."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        ctx = (w * x).sum(dim=1)
        return ctx, w


class TLSTMFuzzyClassifier(nn.Module):
    """Transformer-LSTM + Fuzzy classifier with sector embedding + transition head."""
    def __init__(
        self,
        n_channels,
        n_classes,
        n_sectors,
        hidden_size=128,
        dropout=0.10,
        n_mfs=5,
        d_model=128,
        n_heads=4,
        n_layers=3,
        sector_emb_dim=16,
        max_relative_position=32,
    ):
        super().__init__()
        self.fuzzy = FuzzyLayer(n_channels, n_mfs=n_mfs)

        fuzzy_dim = n_channels * n_mfs
        self.input_proj = nn.Sequential(
            nn.Linear(fuzzy_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.blocks = nn.ModuleList([
            TemporalSelfAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                max_relative_position=max_relative_position,
            )
            for _ in range(n_layers)
        ])

        self.pre_lstm_norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_pool = AttentivePool(hidden_size * 2)
        self.last_y_embed = nn.Embedding(n_classes, hidden_size)
        self.sector_embed = nn.Embedding(n_sectors, sector_emb_dim)

        transition_in_dim = hidden_size * 2 + sector_emb_dim
        self.transition_head = nn.Sequential(
            nn.Linear(transition_in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        head_in_dim = hidden_size * 3 + sector_emb_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_size * 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, last_y, sector_id, return_aux=False):
        x_fuzzy = self.fuzzy(x)
        x_embed = self.input_proj(x_fuzzy)

        for blk in self.blocks:
            x_embed = blk(x_embed)

        x_embed = self.pre_lstm_norm(x_embed)
        lstm_out, _ = self.lstm(x_embed)
        seq_ctx, _ = self.attn_pool(lstm_out)

        last_y_emb = self.last_y_embed(last_y)
        sector_emb = self.sector_embed(sector_id)

        transition_logits = self.transition_head(torch.cat([seq_ctx, sector_emb], dim=-1)).squeeze(-1)

        out = torch.cat([seq_ctx, last_y_emb, sector_emb], dim=-1)
        logits = self.head(out)

        if return_aux:
            return logits, transition_logits
        return logits


FUZZY_MFS = 5
MODEL_D_MODEL = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3
LSTM_HIDDEN = 128
SECTOR_EMB_DIM = 16
TLSTM_DROPOUT = 0.10
MAX_RELATIVE_POSITION = 32

model = TLSTMFuzzyClassifier(
    n_channels=n_channels,
    n_classes=n_classes,
    n_sectors=n_sectors,
    hidden_size=LSTM_HIDDEN,
    dropout=TLSTM_DROPOUT,
    n_mfs=FUZZY_MFS,
    d_model=MODEL_D_MODEL,
    n_heads=TRANSFORMER_HEADS,
    n_layers=TRANSFORMER_LAYERS,
    sector_emb_dim=SECTOR_EMB_DIM,
    max_relative_position=MAX_RELATIVE_POSITION,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Fuzzy MFs per feature: {FUZZY_MFS}')
print(f'Projected d_model: {MODEL_D_MODEL}')
print(f'Transformer layers/heads: {TRANSFORMER_LAYERS}/{TRANSFORMER_HEADS}')
print(f'Relative position window: +/-{MAX_RELATIVE_POSITION}')
print(f'BiLSTM hidden size: {LSTM_HIDDEN}')
print(f'Sector embedding dim: {SECTOR_EMB_DIM}')
print(f'Channels: {n_channels} (base + delta features)')
print(f'Sectors: {n_sectors}')
print(f'Model parameters: {n_params:,}')
print('\nTLSTM-Fuzzy model created successfully!')
print(model)