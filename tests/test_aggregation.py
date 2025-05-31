import torch
from src.data.graph import Graph
from src.layers.aggregation import aggregate_neighbors
import pytest

def test_sum_aggregation():
    # Graph: 0 ↔ 1 ↔ 2 (undirected)
    features = torch.tensor([
        [1.0, 2.0],  # Node 0
        [0.0, 0.0],  # Node 1 (target)
        [3.0, 4.0]   # Node 2
    ])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph(features, edge_index)
    
    agg = aggregate_neighbors(graph, 1, "sum")
    assert torch.allclose(agg, torch.tensor([4.0, 6.0]))  # [1+3, 2+4]

def test_mean_aggregation():
    # Graph: 0 ↔ 1 ↔ 2 (undirected)
    features = torch.tensor([
        [1.0, 2.0],  # Node 0
        [0.0, 0.0],  # Node 1 (target)
        [3.0, 4.0]   # Node 2
    ])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph(features, edge_index)

    agg = aggregate_neighbors(graph, 1, "mean")
    assert torch.allclose(agg, torch.tensor([2.0, 3.0]))  # (1+3)/2, (2+4)/2

def test_max_aggregation():
    # Graph: 0 ↔ 1 ↔ 2 (undirected)
    features = torch.tensor([
        [1.0, 2.0],  # Node 0
        [0.0, 0.0],  # Node 1 (target)
        [3.0, 4.0]   # Node 2
    ])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph(features, edge_index)

    agg = aggregate_neighbors(graph, 1, "max")
    assert torch.allclose(agg, torch.tensor([3.0, 4.0]))  # max(1,3), max(2,4)

def test_min_aggregation():
    # Graph: 0 ↔ 1 ↔ 2 (undirected)
    features = torch.tensor([
        [1.0, 2.0],  # Node 0
        [0.0, 0.0],  # Node 1 (target)
        [3.0, 4.0]   # Node 2
    ])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph(features, edge_index)

    agg = aggregate_neighbors(graph, 1, "min")
    assert torch.allclose(agg, torch.tensor([1.0, 2.0]))  # min(1,3), min(2,4)

def test_isolated_node():
    # Node 1 has no in-neighbours
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[1], [0]])  # 1 → 0
    graph = Graph(features, edge_index, directed=True)
    
    agg = aggregate_neighbors(graph, 1, "sum")
    assert torch.allclose(agg, torch.zeros(2))  # No neighbors → zeros

def test_single_neighbor():
    # Node 0 → 1 (directed)
    features = torch.tensor([[5.0], [10.0]])
    graph = Graph(
        features,
        edge_index=torch.tensor([[0], [1]]),
        directed=True
    )
    
    agg = aggregate_neighbors(graph, 1, "sum")
    assert agg.item() == 5.0  # Node 1 has one in-neighbor (node 0) with value 5.0

def test_directed_aggregation():
    # Directed edges: 0 → 1 → 2
    features = torch.randn(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    graph = Graph(features, edge_index, directed=True)
    
    # Node 1's in-neighbors: only node 0
    agg = aggregate_neighbors(graph, 1, "sum")
    expected = features[0]  # Only node 0 points to node 1
    assert torch.allclose(agg, expected)

def test_undirected_aggregation():
    # Undirected edges: 0 ↔ 1 ↔ 2
    features = torch.randn(3, 3)
    # For undirected, we need both directions explicitly
    edge_index = torch.tensor([[0, 2, 1, 1], [1, 1, 0, 2]])
    graph = Graph(features, edge_index, directed=False)
    
    # Node 1's in-neighbors: 0 and 2
    agg = aggregate_neighbors(graph, 1, "sum")
    expected = features[0] + features[2]
    assert torch.allclose(agg, expected)

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

def test_mismatched_weights():
    edge_weights = torch.tensor([0.5, 1.0])  # 2 weights but 4 edges
    with pytest.raises(ValueError):
        Graph(
            node_features=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_weights=edge_weights
        )

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

def test_weighted_sum():
    """Weights should scale neighbor contributions"""
    # Node 1 has neighbors 0 (weight=0.5) and 2 (weight=2.0)
    features = torch.tensor([[1.0], [999.0], [3.0]])  # Node 1's features unused
    edge_index = torch.tensor([[0, 2], [1, 1]])        # 0→1 and 2→1
    edge_weights = torch.tensor([0.5, 2.0])
    graph = Graph(features, edge_index, edge_weights=edge_weights)
    
    agg = aggregate_neighbors(graph, 1, aggr="sum")
    assert agg.item() == (1.0*0.5 + 3.0*2.0)  # 0.5 + 6.0 = 6.5

def test_weighted_mean():
    """Weights should control averaging"""
    # Node 1 has neighbors 0 (weight=0.5) and 2 (weight=2.0)
    features = torch.tensor([[1.0], [999.0], [3.0]])  # Node 1's features unused
    edge_index = torch.tensor([[0, 2], [1, 1]])        # 0→1 and 2→1
    edge_weights = torch.tensor([0.5, 2.0])
    graph = Graph(features, edge_index, edge_weights=edge_weights)
    agg = aggregate_neighbors(graph, 1, aggr="mean")
    expected = (1.0*0.5 + 3.0*2.0) / (0.5 + 2.0)  # 6.5 / 2.5 = 2.6
    assert agg.item() == pytest.approx(2.6)

def test_zero_weights():
    """Avoid division by zero in weighted mean"""
    # Node 1 has neighbors 0 (weight=0.5) and 2 (weight=2.0)
    features = torch.tensor([[1.0], [999.0], [3.0]])  # Node 1's features unused
    edge_index = torch.tensor([[0, 2], [1, 1]])        # 0→1 and 2→1
    edge_weights = torch.tensor([0.0, 0.0]) 
    graph = Graph(features, edge_index, edge_weights=edge_weights)
    agg = aggregate_neighbors(graph, 1, aggr="mean")
    assert agg.item() == 0.0  # Not NaN!

def test_mismatched_weights():
    """Validate edge_weights/edge_index alignment"""
    features = torch.tensor([[1.0], [2.0], [3.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges
    edge_weights=torch.rand(2)
    
    with pytest.raises(ValueError):
        # 3 edges but 2 weights
        graph = Graph(
        node_features=features,
        edge_index=edge_index,
        edge_weights=edge_weights
    )
        agg = aggregate_neighbors(graph, 1, aggr="mean")