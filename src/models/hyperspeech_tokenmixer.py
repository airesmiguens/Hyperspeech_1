"""HyperSpeech TokenMixer architecture for window-level tabular modeling."""

from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim * 2)
        self.down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(torch.nn.functional.silu(x1) * x2))


class MixerBlock(nn.Module):
    def __init__(self, d_token: int, n_tokens: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, n_tokens * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_tokens * 2, n_tokens),
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.channel_ffn = SwiGLU(dim=d_token, hidden_dim=d_token * 2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        z = z.transpose(1, 2)
        z = self.token_mlp(z)
        x = x + z.transpose(1, 2)
        x = x + self.channel_ffn(self.norm2(x))
        return x


class HyperSpeechTokenMixer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.feature_emb = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_token))

        self.blocks = nn.ModuleList([MixerBlock(d_token=d_token, n_tokens=n_features, dropout=dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_emb.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(x)
        logits = self.head(embedding).squeeze(-1)
        return logits
