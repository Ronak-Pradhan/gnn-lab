"""Tests for the MessagePassing base layer."""

import torch

import pytest
from src.data.graph import Graph
from src.layers.aggregation import aggregate_neighbors
from src.layers.message_passing import MessagePassing


class ScaleMessageLayer(MessagePassing):
    """Multiplies every weighted neighbor row by 2 before aggregation."""

    def message(self, x_i: torch.Tensor, weighted_x_j: torch.Tensor) -> torch.Tensor:
        return weighted_x_j * 2.0


class NoNeighborOnesAggregateLayer(MessagePassing):
    """Returns ones for isolated nodes to verify aggregate hook override."""

    def aggregate(self, messages: torch.Tensor, weights: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        if messages.shape[0] == 0:
            return torch.ones_like(x_i)
        return super().aggregate(messages, weights, x_i)


class AddSelfUpdateLayer(MessagePassing):
    """Uses ``x_i`` in update to verify update override path."""

    def update(self, aggregated: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return aggregated + x_i


class SumOnesAggregateLayer(MessagePassing):
    """Ignores aggr mode and always sums ones per incoming edge."""

    def aggregate(self, messages: torch.Tensor, weights: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        del weights
        if messages.shape[0] == 0:
            return torch.zeros_like(x_i)
        return torch.ones_like(messages).sum(dim=0)


def _constant_three_aggregate(messages: torch.Tensor, weights: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
    """Test double for ``aggregate_fn``: ignore messages and return a constant vector."""
    del messages, weights
    return torch.full_like(x_i, 3.0)


def test_message_hook_changes_aggregation_vs_aggregate_neighbors() -> None:
    """``message`` runs before reduce; aggregate_neighbors ignores it."""
    x = torch.tensor([[1.0], [3.0], [5.0]])
    edge_index = torch.tensor([[0, 2], [1, 1]])
    graph = Graph(x, edge_index, directed=True)

    agg_direct = aggregate_neighbors(graph, 1, "sum")
    assert agg_direct.item() == 6.0  # x[0] + x[2] = 1 + 5

    layer = ScaleMessageLayer(aggr="sum")
    out = layer(x, edge_index)
    assert out[1].item() == 12.0


def test_default_update_returns_aggregate_only() -> None:
    """Default ``update`` is identity; outputs should match ``aggregate_neighbors`` per node."""
    x = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    layer = MessagePassing(aggr="mean")
    out = layer(x, edge_index)
    graph = Graph(x, edge_index)
    for i in range(3):
        expected = aggregate_neighbors(graph, i, "mean")
        assert torch.allclose(out[i], expected)


def test_isolated_node_zero_aggregate() -> None:
    """Nodes with no in-neighbors get a zero aggregate from default ``aggregate``."""
    x = torch.tensor([[1.0], [2.0]])
    edge_index = torch.tensor([[1], [0]])
    layer = MessagePassing(aggr="sum")
    out = layer(x, edge_index)
    assert torch.allclose(out[1], torch.zeros(1))


def test_invalid_aggr_raises() -> None:
    """Unsupported ``aggr`` string should fail at construction time."""
    with pytest.raises(ValueError):
        MessagePassing(aggr="not_a_mode")


def test_aggregate_hook_can_override_no_neighbor_behavior() -> None:
    """Subclass can replace the default zero-vector behavior for empty neighborhoods."""
    x = torch.tensor([[1.0], [2.0]])
    edge_index = torch.tensor([[1], [0]])
    layer = NoNeighborOnesAggregateLayer(aggr="sum")
    out = layer(x, edge_index)
    assert torch.allclose(out[1], torch.ones(1))


def test_update_hook_can_use_self_features() -> None:
    """``update(aggregated, x_i)`` can combine neighbor aggregate with self features."""
    x = torch.tensor([[1.0], [3.0], [5.0]])
    edge_index = torch.tensor([[0, 2], [1, 1]])
    layer = AddSelfUpdateLayer(aggr="sum")
    out = layer(x, edge_index)
    # Node 1 aggregate is 1 + 5 = 6, then update adds x_i=3 -> 9.
    assert out[1].item() == 9.0


def test_aggregate_override_changes_reduction_semantics() -> None:
    """Custom ``aggregate`` can ignore ``aggr`` and implement different reduction rules."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 2], [1, 1]])  # node 1 has two in-neighbors
    layer = SumOnesAggregateLayer(aggr="mean")
    out = layer(x, edge_index)
    # two incoming edges -> aggregate returns [2, 2] regardless of input values.
    assert torch.allclose(out[1], torch.tensor([2.0, 2.0]))


def test_weighted_edges_are_respected_by_message_passing() -> None:
    """``edge_weights`` flow through gather so weighted neighbor rows aggregate correctly."""
    x = torch.tensor([[1.0], [3.0], [5.0]])
    edge_index = torch.tensor([[0, 2], [1, 1]])
    edge_weights = torch.tensor([0.5, 2.0])
    layer = MessagePassing(aggr="sum")
    out = layer(x, edge_index, edge_weights=edge_weights)
    # node1 receives 1*0.5 + 5*2.0 = 10.5
    assert out[1].item() == pytest.approx(10.5)


def test_dtype_and_device_preserved_for_no_neighbor_branch() -> None:
    """Isolated-node branch should not upcast dtype or move tensors off-device."""
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    edge_index = torch.tensor([[1], [0]], dtype=torch.long)
    layer = MessagePassing(aggr="sum")
    out = layer(x, edge_index)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_invalid_edge_weights_length_propagates_graph_error() -> None:
    """Mismatched ``edge_weights`` length should surface as ``Graph`` validation error."""
    x = torch.tensor([[1.0], [2.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_weights = torch.tensor([1.0])  # should be length 2
    layer = MessagePassing(aggr="sum")
    with pytest.raises(ValueError):
        _ = layer(x, edge_index, edge_weights=edge_weights)


def test_custom_aggregate_function_is_used() -> None:
    """``MessagePassing(..., aggregate_fn=...)`` should bypass default reducers."""
    x = torch.tensor([[1.0], [2.0]])
    edge_index = torch.tensor([[0], [1]])
    layer = MessagePassing(aggr="mean", aggregate_fn=_constant_three_aggregate)
    out = layer(x, edge_index)
    assert torch.allclose(out, torch.full_like(x, 3.0))
