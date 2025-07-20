"""Tests for the neighbor aggregation functionality in the GNN implementation.

This module contains comprehensive tests for different aggregation methods (sum, mean, max, min)
across various graph configurations:
- Basic aggregation operations
- Directed vs undirected graphs
- Isolated nodes
- Edge cases (zero features, large graphs)
- Weighted aggregations
"""

import torch
from typing import Tuple
import pytest
from src.data.graph import Graph
from src.layers.aggregation import aggregate_neighbors

@pytest.mark.parametrize("aggr_mode,expected", [
    ("sum", torch.tensor([4.0, 6.0])),   # [1+3, 2+4]
    ("mean", torch.tensor([2.0, 3.0])),  # [(1+3)/2, (2+4)/2]
    ("max", torch.tensor([3.0, 4.0])),   # [max(1,3), max(2,4)]
    ("min", torch.tensor([1.0, 2.0])),   # [min(1,3), min(2,4)]
])
def test_all_aggregation_modes(aggr_mode: str, expected: torch.Tensor) -> None:
    """Test all aggregation modes (sum, mean, max, min) on a simple undirected and unweighted graph.
    
    Graph structure: 0 ↔ 1 ↔ 2 (undirected)
    Tests all four aggregation methods with the same graph structure.
    """
    features = torch.tensor([
        [1.0, 2.0],  # Node 0
        [0.0, 0.0],  # Node 1 (target)
        [3.0, 4.0]   # Node 2
    ])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph(features, edge_index)
    
    agg = aggregate_neighbors(graph, 1, aggr_mode)
    assert torch.allclose(agg, expected)

@pytest.mark.parametrize("weights,aggr_mode,expected", [
    # Positive weights
    (torch.tensor([0.5, 2.0]), "sum", 6.5),                    # 1.0*0.5 + 3.0*2.0 = 0.5 + 6.0 = 6.5
    (torch.tensor([0.5, 2.0]), "mean", pytest.approx(2.6)),   # 6.5 / (0.5 + 2.0) = 6.5 / 2.5 = 2.6
    (torch.tensor([0.5, 2.0]), "max", 6.0),                    # max(1.0*0.5, 3.0*2.0) = max(0.5, 6.0) = 6.0
    (torch.tensor([0.5, 2.0]), "min", 0.5),                    # min(1.0*0.5, 3.0*2.0) = min(0.5, 6.0) = 0.5
    # Negative weights
    (torch.tensor([-0.5, 2.0]), "sum", 5.5),                   # 1.0*(-0.5) + 3.0*2.0 = -0.5 + 6.0 = 5.5
    (torch.tensor([-0.5, 2.0]), "mean", pytest.approx(2.2)),  # 5.5 / (|-0.5| + 2.0) = 5.5 / 2.5 = 2.2
    # Zero weights
    (torch.tensor([0.0, 0.0]), "mean", 0.0),                   # Zero weights should result in zero
    # Mixed zero and negative weights
    (torch.tensor([0.0, -1.0]), "mean", "no_nan_inf"),         # Should not produce NaN or Inf
])
def test_weighted_aggregation_all_modes(weights: torch.Tensor, aggr_mode: str, expected) -> None:
    """Test all weighted aggregation modes with different weight configurations.
    
    Graph structure: 0 -[w1]→ 1 ←[w2]- 2
    Tests sum, mean, max, min with both positive and negative weights.
    """
    features = torch.tensor([[1.0], [999.0], [3.0]])  # Node 1's features unused
    edge_index = torch.tensor([[0, 2], [1, 1]])       # 0→1 and 2→1
    graph = Graph(features, edge_index, directed=True, edge_weights=weights)
    
    agg = aggregate_neighbors(graph, 1, aggr=aggr_mode)
    if expected == "no_nan_inf":
        # Special case: check that result is not NaN or Inf
        assert not torch.isnan(agg)
        assert not torch.isinf(agg)
    elif isinstance(expected, float):
        assert agg.item() == expected
    else:
        assert agg.item() == expected


def test_isolated_node() -> None:
    """Test aggregation for a node with no incoming neighbors.
    
    Graph structure: 1 → 0 (directed, node 1 has no in-neighbors)
    Expected: Should return zero tensor when no neighbors exist.
    """
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[1], [0]])  # 1 → 0
    graph = Graph(features, edge_index, directed=True)
    
    agg = aggregate_neighbors(graph, 1, "sum")
    assert torch.allclose(agg, torch.zeros(2))  # No neighbors → zeros

def test_invalid_aggregation_mode():
    features = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph(features, edge_index, directed=False)

    with pytest.raises(ValueError):
        aggregate_neighbors(graph, 1, "invalid_mode")

def test_node_index_out_of_bounds():
    features = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph(features, edge_index, directed=False)

    with pytest.raises(IndexError):
        aggregate_neighbors(graph, 100, "sum")  # Only 3 nodes exist

def test_zero_feature_dimension():
    features = torch.empty(3, 0)  # 3 nodes, 0 features
    graph = Graph(features, torch.empty((2, 0)))
    agg = aggregate_neighbors(graph, 0, "sum")
    assert agg.shape == (0,)  # Empty but correct shape

def test_large_graph_performance():
    # 10k nodes, 100k edges (test for OOM errors)
    features = torch.randn(10000, 128)
    edge_index = torch.randint(0, 10000, (2, 100000))
    graph = Graph(features, edge_index)
    
    # Should complete without memory issues
    aggregate_neighbors(graph, 5000, "sum")





