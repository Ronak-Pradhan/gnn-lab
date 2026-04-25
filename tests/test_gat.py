"""Tests for GATConv."""

import torch
import pytest

from src.layers.gat import GATConv


def test_gat_forward_shapes() -> None:
    """Validate basic output shape contract for GAT forward.

    Uses a tiny directed cycle graph with 4 nodes and input feature size 3.
    The layer is configured with ``out_channels=5`` and no attention dropout,
    then we assert only the structural contract:
    - node count is preserved
    - feature dimension maps from ``in_channels`` to ``out_channels``
    """
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    conv = GATConv(3, 5, dropout=0.0)
    out = conv(x, edge_index)
    assert out.shape == (4, 5)


def test_gat_isolated_uses_dst_linear() -> None:
    """Check isolated-node fallback path.

    With zero incoming edges, there is no neighborhood softmax/aggregation.
    The implementation should return projected self features from ``lin_dst``
    plus optional bias. This test locks in that behavior exactly.
    """
    torch.manual_seed(1)
    x = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    conv = GATConv(3, 2, bias=True, dropout=0.0)
    out = conv(x, edge_index)
    with torch.no_grad():
        expected = conv.lin_dst(x[0]) + conv.bias
    assert torch.allclose(out[0], expected, atol=1e-5)


def test_gat_rejects_external_edge_weights() -> None:
    """Ensure unsupported ``edge_weights`` are rejected clearly.

    This GAT variant learns edge importance via attention coefficients and
    intentionally does not accept externally provided edge weights. Passing
    them should raise a ``ValueError`` with a clear message.
    """
    x = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weights = torch.ones(2)
    conv = GATConv(3, 2, dropout=0.0)

    with pytest.raises(ValueError, match="does not accept external edge_weights"):
        _ = conv(x, edge_index, edge_weights=edge_weights)
