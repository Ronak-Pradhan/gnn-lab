"""Tests for the Graph data structure implementation.

This module contains comprehensive tests for the Graph class functionality:
- Node feature storage and retrieval
- Edge connectivity and weights
- Directed vs undirected graph behavior
- Edge cases (empty graphs, single nodes)
- Error handling for invalid configurations
"""

import torch
from typing import Tuple, List, Optional
from src.data.graph import Graph
import pytest

def create_test_features(num_nodes: int, feature_dim: int) -> torch.Tensor:
    """Create test node features with sequential values.
    
    Args:
        num_nodes: Number of nodes in the graph
        feature_dim: Number of features per node
        
    Returns:
        Tensor of shape (num_nodes, feature_dim) with sequential values
    """
    return torch.tensor([
        [n + f/10 for f in range(feature_dim)]
        for n in range(num_nodes)
    ], dtype=torch.float)

def test_node_feature_retrieval() -> None:
    """Test basic node feature storage and retrieval.
    
    Creates a graph with 3 nodes, each having 5 features,
    and verifies that features can be correctly accessed.
    """
    features = create_test_features(3, 5)
    graph = Graph(
        node_features=features,
        edge_index=torch.empty((2, 0)),  # No edges
    )
    
    # Test feature retrieval
    assert torch.allclose(graph.x[0], features[0])
    assert torch.allclose(graph.x[1], features[1])
    assert torch.allclose(graph.x[2], features[2])

def test_single_node_graph() -> None:
    """Test graph operations on a minimal graph with one node.
    
    Verifies that a single-node graph works correctly and
    properly handles neighbor queries.
    """
    graph = Graph(
        node_features=torch.tensor([[10.0]]),  # 1 node, 1 feature
        edge_index=torch.empty((2, 0)),
    )
    assert graph.x[0].item() == 10.0
    neighbors = graph.get_neighbors(0)
    assert neighbors.tolist() == []

def test_edge_weight_storage() -> None:
    """Test storage and retrieval of edge weights.
    
    Creates a directed graph with weighted edges and verifies
    that weights are stored correctly.
    """
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

def test_weighted_neighbor_relationships() -> None:
    """Test neighbor relationships in a weighted directed graph.
    
    Graph structure: 0 -[0.5]→ 1 ←[2.0]- 2
    Verifies that neighbors and their weights are correctly identified.
    """
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

def test_empty_graph() -> None:
    """Test initialization and operations on an empty graph.
    
    Verifies that a graph with no nodes can be created and
    basic properties are correct.
    """
    graph = Graph(
        node_features=torch.empty((0, 0)),  # 0 nodes
        edge_index=torch.empty((2, 0)),
    )
    assert graph.num_nodes == 0

def test_no_edge_weights() -> None:
    """Test graph creation without edge weights.
    
    Verifies that a graph can be created with unweighted edges
    and the edge_weights attribute is None.
    """
    graph = Graph(
        node_features=torch.randn(2, 2),
        edge_index=torch.tensor([[0], [1]]),
        edge_weights=None  # Explicit no weights
    )
    assert graph.edge_weights is None

def test_weight_edge_index_mismatch() -> None:
    """Test error handling for mismatched edge indices and weights.
    
    Verifies that an error is raised when the number of edge weights
    doesn't match the number of edges.
    """
    with pytest.raises(ValueError):
        Graph(
            node_features=torch.randn(2, 2),
            edge_index=torch.tensor([[0], [1]]),  # 1 edge
            edge_weights=torch.tensor([0.5, 1.0])  # 2 weights
        )

def test_unidirectional_weights() -> None:
    """Test weight handling in a directed graph with one-way edges.
    
    Graph structure: 0 -[0.5]→ 1 -[2.0]→ 2
    Verifies that weights are correctly associated with edges
    and neighbor relationships respect edge directions.
    """
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
