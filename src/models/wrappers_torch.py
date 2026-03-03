"""PyTorch training wrappers for tabular models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def get_device(preference=("cuda", "mps", "cpu")) -> torch.device:
	for item in preference:
		if item == "cuda" and torch.cuda.is_available():
			return torch.device("cuda")
		if item == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			return torch.device("mps")
		if item == "cpu":
			return torch.device("cpu")
	return torch.device("cpu")


class BCEWithLogitsLabelSmoothing(nn.Module):
	def __init__(self, smoothing: float = 0.0):
		super().__init__()
		self.smoothing = float(smoothing)

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		targets = targets.float()
		if self.smoothing > 0:
			targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
		return nn.functional.binary_cross_entropy_with_logits(logits, targets)


@dataclass
class TorchTrainConfig:
	lr: float = 1e-3
	weight_decay: float = 5e-4
	batch_size: int = 256
	max_epochs: int = 200
	patience: int = 25
	label_smoothing: float = 0.0
	device: str = "cpu"


def _make_loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
	x_t = torch.as_tensor(x, dtype=torch.float32)
	if y is None:
		ds = TensorDataset(x_t)
	else:
		y_t = torch.as_tensor(y, dtype=torch.float32)
		ds = TensorDataset(x_t, y_t)
	return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_torch_binary(
	model: nn.Module,
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_val: np.ndarray,
	y_val: np.ndarray,
	cfg: TorchTrainConfig,
	device: torch.device,
) -> nn.Module:
	model = model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	loss_fn = BCEWithLogitsLabelSmoothing(cfg.label_smoothing)

	train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
	val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
	train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

	best_state = None
	best = float("inf")
	bad = 0

	for _ in range(cfg.max_epochs):
		model.train()
		for xb, yb in train_loader:
			xb, yb = xb.to(device), yb.to(device)
			optimizer.zero_grad(set_to_none=True)
			logits = model(xb)
			if logits.ndim > 1:
				logits = logits.squeeze(-1)
			loss = loss_fn(logits, yb)
			loss.backward()
			optimizer.step()

		model.eval()
		losses = []
		with torch.no_grad():
			for xb, yb in val_loader:
				xb, yb = xb.to(device), yb.to(device)
				logits = model(xb)
				if logits.ndim > 1:
					logits = logits.squeeze(-1)
				losses.append(loss_fn(logits, yb).item())

		val_loss = float(np.mean(losses)) if losses else float("inf")
		if val_loss < best - 1e-5:
			best = val_loss
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			bad = 0
		else:
			bad += 1
			if bad >= cfg.patience:
				break

	if best_state is not None:
		model.load_state_dict(best_state)
	return model


def train_binary_tabular_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TorchTrainConfig,
    forward_prob_fn=None,
) -> nn.Module:
    device = torch.device(config.device)
    n_val = max(1, int(0.1 * len(x_train)))
    fit_x, fit_y = x_train[:-n_val], y_train[:-n_val]
    val_x, val_y = x_train[-n_val:], y_train[-n_val:]
    model = train_torch_binary(model, fit_x, fit_y, val_x, val_y, cfg=config, device=device)
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


def predict_proba_torch(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
	return predict_proba_binary(model=model, x_eval=x, batch_size=batch_size, device=str(device))
