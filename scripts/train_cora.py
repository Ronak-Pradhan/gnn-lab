"""Train a small GCN on a Cora-style directory (Planetoid files)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from src.data.cora import load_cora_dir
from src.layers.gcn import GCNConv
from src.train.node_classification import evaluate, set_seed, train_epoch


class TinyGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GCN on Cora-style data directory")
    parser.add_argument("data_dir", type=Path, help="Directory with cora.content and cora.cites")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_cora_dir(args.data_dir)
    num_features = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1

    model = TinyGCN(num_features, args.hidden, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data.x, data.edge_index, data.y, data.train_mask, opt)
        if epoch == 1 or epoch % 20 == 0:
            val_loss, val_acc = evaluate(model, data.x, data.edge_index, data.y, data.val_mask)
            if data.val_mask.any():
                print(f"epoch={epoch:04d} train_loss={loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
            else:
                print(f"epoch={epoch:04d} train_loss={loss:.4f}")

    if data.test_mask.any():
        _, test_acc = evaluate(model, data.x, data.edge_index, data.y, data.test_mask)
        print(f"test_acc={test_acc:.3f}")


if __name__ == "__main__":
    main()
