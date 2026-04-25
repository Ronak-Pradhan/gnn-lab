"""Training utilities for node-level classification on a single graph."""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def train_epoch(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> float:
    """One full-graph training epoch; returns mean loss on training nodes."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x, edge_index)
    if criterion is None:
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
    else:
        loss = criterion(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """Returns (loss, accuracy) on nodes where ``mask`` is True."""
    model.eval()
    logits = model(x, edge_index)
    loss = F.cross_entropy(logits[mask], y[mask]).item()
    pred = logits[mask].argmax(dim=-1)
    acc = (pred == y[mask]).float().mean().item()
    return loss, acc
