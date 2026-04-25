"""Tests for GraphSAGEConv."""

import torch

from src.layers.graphsage import GraphSAGEConv


def test_graphsage_concat_and_mean() -> None:
    """Verify GraphSAGE mean aggregation and concat-update behavior.

    This test builds a two-node directed graph with one edge ``1 -> 0``:
    - Node 0 has one incoming neighbor (node 1), so its neighbor aggregate is ``x_1``.
    - Node 1 is isolated, so its neighbor aggregate is the zero vector.

    The layer uses ``bias=False`` and an identity projection matrix so output is
    exactly ``concat(x_i, mean_neighbors(i))``. Assertions check both rows match
    these expected concatenations.
    """
    # 0 <- 1 only: node 0 sees neighbor 1; node 1 isolated
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    edge_index = torch.tensor([[1], [0]], dtype=torch.long)

    layer = GraphSAGEConv(2, 4, bias=False)
    with torch.no_grad():
        layer.lin.weight.copy_(torch.eye(4))
        # out = concat(x_i, mean_nb) projected by I
        # node 0: concat([1,0],[0,2]) -> [1,0,0,2]
        # node 1: concat([0,2],[0,0]) -> [0,2,0,0]

    out = layer(x, edge_index)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.0, 2.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 2.0, 0.0, 0.0]))
