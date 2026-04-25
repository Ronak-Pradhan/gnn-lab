"""Tests for ``GCNConv`` and symmetric normalization helpers.

Checks propagation against a dense adjacency construction and validates that
external ``edge_weights`` are rejected in favor of topology-derived weights.
"""

import torch
import pytest

from src.layers.gcn import GCNConv, symmetric_normalized_gcn


def _dense_norm_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build dense normalized adjacency ``S`` matching sparse ``symmetric_normalized_gcn`` edges."""
    ei, ew = symmetric_normalized_gcn(edge_index, num_nodes, torch.float32, edge_index.device)
    s = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=edge_index.device)
    for e in range(ei.size(1)):
        src = ei[0, e]
        tgt = ei[1, e]
        s[tgt, src] = s[tgt, src] + ew[e]
    return s


def test_gcn_matches_dense_propagation() -> None:
    """``GCNConv`` output should equal ``S @ x`` when the linear is identity."""
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    s = _dense_norm_matrix(edge_index, 2)

    layer = GCNConv(2, 2, bias=False)
    with torch.no_grad():
        layer.lin.weight.copy_(torch.eye(2))

    out = layer(x, edge_index)
    expected = s @ x
    assert torch.allclose(out, expected, atol=1e-5)


def test_gcn_rejects_external_edge_weights() -> None:
    """Passing ``edge_weights`` into ``GCNConv`` must raise ``ValueError``."""
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    layer = GCNConv(2, 2, bias=False)

    with pytest.raises(ValueError, match="does not accept external edge_weights"):
        _ = layer(x, edge_index, edge_weights=edge_weights)
