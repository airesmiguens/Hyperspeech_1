"""HyperSpeech MIL pooling model for subject-level predictions from window embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .hyperspeech_tokenmixer import HyperSpeechTokenMixer
from .hyperspeech_tokenmixer import HyperSpeechTokenMixerConfig


def make_padded_subject_tensor(
    embeddings: np.ndarray,
    logits: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_subjects = np.unique(subject_ids)
    grouped_emb = [embeddings[subject_ids == sid] for sid in unique_subjects]
    grouped_logits = [logits[subject_ids == sid] for sid in unique_subjects]

    max_windows = max(arr.shape[0] for arr in grouped_emb)
    d_emb = embeddings.shape[1]

    emb_tensor = torch.zeros((len(unique_subjects), max_windows, d_emb), dtype=torch.float32)
    logit_tensor = torch.zeros((len(unique_subjects), max_windows), dtype=torch.float32)
    mask = torch.zeros((len(unique_subjects), max_windows), dtype=torch.bool)

    for idx, (emb_arr, logit_arr) in enumerate(zip(grouped_emb, grouped_logits)):
        n = emb_arr.shape[0]
        emb_tensor[idx, :n] = torch.as_tensor(emb_arr, dtype=torch.float32)
        logit_tensor[idx, :n] = torch.as_tensor(logit_arr, dtype=torch.float32)
        mask[idx, :n] = True

    return emb_tensor, logit_tensor, mask


@dataclass
class HyperSpeechMILConfig:
    encoder: HyperSpeechTokenMixerConfig
    attn_hidden: int = 64
    dropout: float = 0.1
    topk: int = 5


class AttnPool(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.net(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class HyperSpeechMIL(nn.Module):
    """Subject-level model: encode windows -> attention pool + robust logit stats -> subject logit."""

    def __init__(self, cfg: HyperSpeechMILConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = HyperSpeechTokenMixer(self.cfg.encoder)
        d = self.cfg.encoder.d_token
        self.pool = AttnPool(d, self.cfg.attn_hidden)
        self.head = nn.Sequential(
            nn.Linear(d + 4, d),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(d, 1),
        )

    def forward(self, x_windows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, windows, features = x_windows.shape
        x_flat = x_windows.view(bsz * windows, features)

        logits_w, emb_w = self.encoder(x_flat)
        logit_tensor = logits_w.view(bsz, windows)
        emb_tensor = emb_w.view(bsz, windows, -1)

        attn_pool = self.pool(emb_tensor, mask)

        masked_logits = logit_tensor.masked_fill(~mask, 0.0)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_logit = masked_logits.sum(dim=1) / counts
        centered = masked_logits - mean_logit.unsqueeze(1)
        centered = centered.masked_fill(~mask, 0.0)
        std_logit = torch.sqrt((centered.pow(2).sum(dim=1) / counts) + 1e-12)

        max_logit = logit_tensor.masked_fill(~mask, -1e9).max(dim=1).values
        top_k = min(self.cfg.topk, logit_tensor.shape[1])
        topk_vals = logit_tensor.masked_fill(~mask, -1e9).topk(k=top_k, dim=1).values
        topk_mean = topk_vals.mean(dim=1)

        stats = torch.stack([mean_logit, std_logit, max_logit, topk_mean], dim=1)
        subject_feat = torch.cat([attn_pool, stats], dim=1)
        return self.head(subject_feat).squeeze(-1)
