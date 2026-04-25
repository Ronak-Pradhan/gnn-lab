"""Tests for node classification training helpers.

Uses the ``cora_mini`` fixture with ``train_epoch`` / ``evaluate`` to smoke-test
the training loop without long runs or external data.
"""

from pathlib import Path

import torch
import torch.optim as optim

from src.data.cora import load_cora_dir
from src.layers.gcn import GCNConv
from src.train.node_classification import evaluate, set_seed, train_epoch


def test_train_epoch_runs_and_loss_is_finite() -> None:
    """Run several ``train_epoch`` steps and ``evaluate``; assert finite loss and valid accuracy."""
    root = Path(__file__).resolve().parent / "fixtures" / "cora_mini"
    data = load_cora_dir(root)
    set_seed(0)
    num_classes = int(data.y.max().item()) + 1
    model = GCNConv(data.x.size(1), num_classes, bias=True)
    opt = optim.Adam(model.parameters(), lr=0.05)

    losses = [train_epoch(model, data.x, data.edge_index, data.y, data.train_mask, opt) for _ in range(5)]
    assert all(isinstance(x, float) for x in losses)
    assert all(x == x for x in losses)  # not NaN

    loss, acc = evaluate(model, data.x, data.edge_index, data.y, data.train_mask)
    assert 0.0 <= acc <= 1.0
    assert loss == loss
