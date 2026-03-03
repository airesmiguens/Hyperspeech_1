"""PyTorch training wrappers for tabular models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


@dataclass
class TorchTrainConfig:
	epochs: int = 80
	batch_size: int = 128
	lr: float = 1e-3
	weight_decay: float = 1e-4
	device: str = "cpu"


def _make_loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
	x_t = torch.as_tensor(x, dtype=torch.float32)
	if y is None:
		ds = TensorDataset(x_t)
	else:
		y_t = torch.as_tensor(y, dtype=torch.float32)
		ds = TensorDataset(x_t, y_t)
	return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_binary_tabular_model(
	model: nn.Module,
	x_train: np.ndarray,
	y_train: np.ndarray,
	config: TorchTrainConfig,
	forward_prob_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> nn.Module:
	device = torch.device(config.device)
	model = model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	criterion = nn.BCEWithLogitsLoss()

	train_loader = _make_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
	model.train()
	for _ in range(config.epochs):
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			logits = model(xb)
			if logits.ndim > 1:
				logits = logits.squeeze(-1)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()

	if forward_prob_fn is not None:
		model.forward_prob = lambda x: forward_prob_fn(model, x)
	return model


def predict_proba_binary(model: nn.Module, x_eval: np.ndarray, batch_size: int = 1024, device: str = "cpu") -> np.ndarray:
	model = model.to(device)
	model.eval()
	loader = _make_loader(x_eval, y=None, batch_size=batch_size, shuffle=False)
	prob_chunks: list[np.ndarray] = []

	with torch.no_grad():
		for (xb,) in loader:
			xb = xb.to(device)
			logits = model(xb)
			if logits.ndim > 1:
				logits = logits.squeeze(-1)
			probs = torch.sigmoid(logits).cpu().numpy()
			prob_chunks.append(probs)
	return np.concatenate(prob_chunks)
