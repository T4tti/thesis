"""
model/tlstm_architecture.py — TLSTMFuzzyClassifier architecture definition.

Chứa toàn bộ các class PyTorch cần thiết để load checkpoint đã train.
Không phụ thuộc vào bất kỳ training code nào.

How to Run (standalone sanity check):
    python -m model.tlstm_architecture

Expected Output:
    TLSTMFuzzyClassifier instantiated OK: 1,234,xxx params
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class FuzzyLayer(nn.Module):
    """Gaussian fuzzy membership expansion: (B, T, F) → (B, T, F * n_mfs)."""

    def __init__(self, input_features: int, n_mfs: int = 5, init_sigma: float = 1.0) -> None:
        super().__init__()
        self.input_features = input_features
        self.n_mfs = n_mfs
        self.centers = nn.Parameter(
            torch.linspace(-1.0, 1.0, n_mfs).repeat(input_features, 1)
        )
        self.log_sigma = nn.Parameter(
            torch.full((input_features, n_mfs), math.log(init_sigma))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            memberships: (B, T, F * n_mfs)
        """
        x_exp = x.unsqueeze(-1)                                            # (B, T, F, 1)
        centers = self.centers.unsqueeze(0).unsqueeze(0)                   # (1, 1, F, n_mfs)
        sigma = torch.exp(self.log_sigma).unsqueeze(0).unsqueeze(0).clamp(min=1e-3)
        memberships = torch.exp(-0.5 * ((x_exp - centers) / sigma) ** 2)  # (B, T, F, n_mfs)
        return memberships.reshape(x.shape[0], x.shape[1], self.input_features * self.n_mfs)


class TemporalSelfAttentionBlock(nn.Module):
    """Pre-norm Transformer block with learnable relative positional bias."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_relative_position: int = 32,
    ) -> None:
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

    def _relative_bias_matrix(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        rel = pos[None, :] - pos[:, None]
        rel = rel.clamp(-self.max_relative_position, self.max_relative_position)
        rel = rel + self.max_relative_position
        return self.relative_bias[rel].to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_bias = self._relative_bias_matrix(x.size(1), x.device, x.dtype)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AttentivePool(nn.Module):
    """Learnable weighted pooling over temporal axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.score(x), dim=1)
        ctx = (w * x).sum(dim=1)
        return ctx, w


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class TLSTMFuzzyClassifier(nn.Module):
    """
    Transformer-BiLSTM + Fuzzy classifier with sector embedding and transition head.

    Input:
        x          : (B, T, n_channels)  — scaled financial feature sequences
        last_y     : (B,)  long          — class ID of last known rating
        sector_id  : (B,)  long          — encoded sector index

    Output (default):
        logits     : (B, n_classes)

    Output (return_aux=True):
        (logits, transition_logits)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_sectors: int,
        hidden_size: int = 128,
        dropout: float = 0.10,
        n_mfs: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        sector_emb_dim: int = 16,
        max_relative_position: int = 32,
    ) -> None:
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

        head_in_dim = hidden_size * 3 + sector_emb_dim      # ctx(256) + last_y(128) + sector(16) = 400
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_size * 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        last_y: torch.Tensor,
        sector_id: torch.Tensor,
        return_aux: bool = False,
    ):
        x_fuzzy = self.fuzzy(x)                         # (B, T, F*n_mfs)
        x_embed = self.input_proj(x_fuzzy)              # (B, T, d_model)

        for blk in self.blocks:
            x_embed = blk(x_embed)

        x_embed = self.pre_lstm_norm(x_embed)
        lstm_out, _ = self.lstm(x_embed)                # (B, T, hidden*2)
        seq_ctx, _ = self.attn_pool(lstm_out)           # (B, hidden*2)

        last_y_emb = self.last_y_embed(last_y)          # (B, hidden)
        sector_emb = self.sector_embed(sector_id)       # (B, sector_emb_dim)

        transition_logits = self.transition_head(
            torch.cat([seq_ctx, sector_emb], dim=-1)
        ).squeeze(-1)                                   # (B,)

        out = torch.cat([seq_ctx, last_y_emb, sector_emb], dim=-1)  # (B, 400)
        logits = self.head(out)                         # (B, n_classes)

        if return_aux:
            return logits, transition_logits
        return logits


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    meta_path = Path(__file__).parent / "tlstm_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        hp = meta["model_hparams"]
        m = TLSTMFuzzyClassifier(**{k: v for k, v in hp.items() if k != "input_size"})
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"TLSTMFuzzyClassifier instantiated OK: {n_params:,} params")
    else:
        print("No tlstm_meta.json found — skipping hparam check.")
