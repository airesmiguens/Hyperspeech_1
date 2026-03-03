"""HyperSpeech TokenMixer architecture for window-level tabular modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


class GatedMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0, gating: str = "swiglu"):
        super().__init__()
        self.gating = gating
        self.dropout = nn.Dropout(dropout)
        if gating in ("swiglu", "glu"):
            self.fc1 = nn.Linear(d_in, 2 * d_out)
        else:
            self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.gating == "swiglu":
            x = swiglu(x)
        elif self.gating == "glu":
            a, b = x.chunk(2, dim=-1)
            x = torch.sigmoid(a) * b
        else:
            x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TokenMixerBlock(nn.Module):
    def __init__(self, n_tokens: int, d_token: int, dropout: float = 0.0, gating: str = "swiglu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.token_mlp = GatedMLP(n_tokens, n_tokens, dropout=dropout, gating=gating)
        self.norm2 = nn.LayerNorm(d_token)
        self.chan_mlp = GatedMLP(d_token, d_token, dropout=dropout, gating=gating)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + self.dropout(y)

        y = self.norm2(x)
        y = self.chan_mlp(y)
        x = x + self.dropout(y)
        return x


@dataclass
class HyperSpeechTokenMixerConfig:
    n_features: int
    d_token: int = 64
    n_layers: int = 4
    dropout: float = 0.1
    gating: str = "swiglu"
    feature_dropout: float = 0.10
    token_mode: str = "feature"
    groups: Optional[List[List[int]]] = None


class HyperSpeechTokenMixer(nn.Module):
    def __init__(self, cfg: HyperSpeechTokenMixerConfig | int, d_token: int = 64, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        if isinstance(cfg, int):
            self.cfg = HyperSpeechTokenMixerConfig(
                n_features=cfg,
                d_token=d_token,
                n_layers=n_blocks,
                dropout=dropout,
            )
        else:
            self.cfg = cfg

        self.feature_dropout = nn.Dropout(self.cfg.feature_dropout)

        if self.cfg.token_mode == "feature":
            self.groups = None
            self.n_tokens = self.cfg.n_features
        elif self.cfg.token_mode == "group":
            if not self.cfg.groups:
                raise ValueError("For token_mode='group', cfg.groups must be provided.")
            self.groups = self.cfg.groups
            self.n_tokens = len(self.cfg.groups)
        else:
            raise ValueError("token_mode must be 'feature' or 'group'.")

        self.proj = nn.Linear(1, self.cfg.d_token)
        self.blocks = nn.ModuleList(
            [
                TokenMixerBlock(self.n_tokens, self.cfg.d_token, dropout=self.cfg.dropout, gating=self.cfg.gating)
                for _ in range(self.cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.cfg.d_token)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.d_token, self.cfg.d_token),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_token, 1),
        )

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.token_mode == "feature":
            return x.unsqueeze(-1)
        toks = []
        for idxs in self.groups:
            toks.append(x[:, idxs].mean(dim=1, keepdim=True))
        return torch.stack(toks, dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_dropout(x)
        tokens = self._tokenize(x)
        tokens = self.proj(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        emb = tokens.mean(dim=1)
        logits = self.head(emb).squeeze(-1)
        return logits, emb
