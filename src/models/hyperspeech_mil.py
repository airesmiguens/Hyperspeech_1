"""HyperSpeech MIL pooling model for subject-level predictions from window embeddings."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


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


class HyperSpeechMIL(nn.Module):
    def __init__(self, d_emb: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.subject_head = nn.Sequential(
            nn.Linear(d_emb + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb_tensor: torch.Tensor, logit_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attn(emb_tensor).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(attn_scores, dim=1)

        attn_pool = torch.sum(emb_tensor * weights.unsqueeze(-1), dim=1)

        masked_logits = logit_tensor.masked_fill(~mask, 0.0)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_logit = masked_logits.sum(dim=1) / counts
        centered = masked_logits - mean_logit.unsqueeze(1)
        centered = centered.masked_fill(~mask, 0.0)
        std_logit = torch.sqrt((centered.pow(2).sum(dim=1) / counts) + 1e-12)

        max_logit = logit_tensor.masked_fill(~mask, -1e9).max(dim=1).values
        top_k = min(3, logit_tensor.shape[1])
        topk_vals = logit_tensor.masked_fill(~mask, -1e9).topk(k=top_k, dim=1).values
        topk_mean = topk_vals.mean(dim=1)

        stats = torch.stack([mean_logit, std_logit, max_logit, topk_mean], dim=1)
        subject_feat = torch.cat([attn_pool, stats], dim=1)
        return self.subject_head(subject_feat).squeeze(-1)
