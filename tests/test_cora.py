"""Tests for Cora-style dataset loading.

Covers ``load_cora_dir`` against the local ``cora_mini`` fixture so CI stays
offline and deterministic (no full Planetoid download required).
"""

from pathlib import Path

import torch

from src.data.cora import load_cora_dir


def test_load_cora_mini_fixture() -> None:
    """Load ``tests/fixtures/cora_mini`` and assert tensor shapes and dtypes.

    Verifies node count, feature width, label vector, undirected edge expansion
    (two cite lines become four directed edges), and boolean split masks.
    """
    root = Path(__file__).resolve().parent / "fixtures" / "cora_mini"
    data = load_cora_dir(root)
    assert data.x.shape == (3, 2)
    assert data.y.shape == (3,)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 4  # two undirected pairs -> 4 directed edges
    assert data.train_mask.dtype == torch.bool
