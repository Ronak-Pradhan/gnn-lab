import torch
from src.data.graph import Graph

def test_node_feature_retrieval():
    # 3 nodes with 5 features each
    features = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [1.1, 1.2, 1.3, 1.4, 1.5],
        [2.1, 2.2, 2.3, 2.4, 2.5]
    ])
    graph = Graph(
        node_features=features,
        edge_index=torch.empty((2, 0)),  # No edges
    )
    
    # Test feature retrieval
    assert torch.allclose(graph.x[0], features[0])
    assert torch.allclose(graph.x[1], features[1])
    assert torch.allclose(graph.x[2], features[2])

def test_single_node_graph():
    graph = Graph(
        node_features=torch.tensor([[10.0]]),  # 1 node, 1 feature
        edge_index=torch.empty((2, 0)),
    )
    assert graph.x[0].item() == 10.0
    neighbors = graph.get_neighbors(0)
    assert neighbors.tolist() == []

def test_edge_weight_storage():
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_weights = torch.tensor([0.5, 2.0])
    
    graph = Graph(
        node_features=torch.randn(3, 2),
        edge_index=edge_index,
        edge_weights=edge_weights,
        directed=True
    )
    
    # Verify weights are stored correctly
    assert torch.allclose(graph.edge_weights, edge_weights)
    assert graph.edge_weights.shape == (2,)

def test_weighted_neighbor_relationships():
    # 0 → 1 ← 2 (node 1 has two in-neighbors: 0 and 2)
    edge_index = torch.tensor([[0, 2], [1, 1]])  # Two edges pointing to node 1
    edge_weights = torch.tensor([0.5, 2.0])
    
    graph = Graph(
        node_features=torch.randn(3, 3),
        edge_index=edge_index,
        edge_weights=edge_weights,
        directed=True
    )
    
    # Check in-neighbors of node 1
    neighbors = graph.get_neighbors(1)
    assert sorted(neighbors.tolist()) == [0, 2]  # Should return sources of edges to node 1
    # Verify corresponding weights
    mask = graph.edge_index[1] == 1  # Find edges where target is node 1
    assert torch.allclose(
        graph.edge_weights[mask],
        torch.tensor([0.5, 2.0])
    )

def test_empty_graph():
    graph = Graph(
        node_features=torch.empty((0, 0)),  # 0 nodes
        edge_index=torch.empty((2, 0)),
    )
    assert graph.num_nodes == 0

def test_no_edge_weights():
    graph = Graph(
        node_features=torch.randn(2, 2),
        edge_index=torch.tensor([[0], [1]]),
        edge_weights=None  # Explicit no weights
    )
    assert graph.edge_weights is None

def test_weight_edge_index_mismatch():
    try:
        Graph(
            node_features=torch.randn(2, 2),
            edge_index=torch.tensor([[0], [1]]),  # 1 edge
            edge_weights=torch.tensor([0.5, 1.0])  # 2 weights
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected behavior

def test_unidirectional_weights():
    """Test weights in a directed graph with one-way edges"""
    # Create a directed graph: 0 → 1 → 2 with different weights
    edge_index = torch.tensor([[0, 1], [1, 2]])  # Two directed edges
    edge_weights = torch.tensor([0.5, 2.0])      # Different weights for each edge
    
    graph = Graph(
        node_features=torch.randn(3, 2),
        edge_index=edge_index,
        edge_weights=edge_weights,
        directed=True
    )
    
    # Node 0 should have no in-neighbors
    n0_neighbors = graph.get_neighbors(0)
    assert n0_neighbors.tolist() == []
    
    # Node 1 should have 0 as in-neighbor with weight 0.5
    n1_neighbors = graph.get_neighbors(1)
    assert n1_neighbors.tolist() == [0]
    assert torch.allclose(
        graph.edge_weights[graph.edge_index[1] == 1],
        torch.tensor([0.5])
    )
    
    # Node 2 should have 1 as in-neighbor with weight 2.0
    n2_neighbors = graph.get_neighbors(2)
    assert n2_neighbors.tolist() == [1]
    assert torch.allclose(
        graph.edge_weights[graph.edge_index[1] == 2],
        torch.tensor([2.0])
    )
