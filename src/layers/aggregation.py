"""Neighbor feature aggregation functionality for Graph Neural Networks."""

import torch
from typing import Optional, Union
from src.data.graph import Graph
from src.registry_and_constants import AGGREGATION_METHODS


def aggregate_neighbors(
    graph: Graph,
    node_idx: int,
    aggr: str = "mean",
) -> torch.Tensor:
    """Aggregate features from neighboring nodes in a graph.
    
    This function combines features from neighboring nodes using the specified
    aggregation method. For weighted graphs, the weights are applied before
    aggregation. If a node has no neighbors, a zero vector is returned.

    The aggregation process:
    1. Identifies neighbors of the target node
    2. Applies edge weights (if any) to neighbor features
    3. Combines features using the specified aggregation method
    
    Args:
        graph: Input graph containing node features and connectivity information
        node_idx: Index of the target node to aggregate neighbors for
        aggr: Aggregation method to use:
             - "sum": Sum of neighbor features
             - "mean": Weighted average of neighbor features
             - "max": Maximum value per feature across neighbors
             - "min": Minimum value per feature across neighbors
        
    Returns:
        Tensor of shape [num_features] containing aggregated neighbor features.
        If the node has no neighbors, returns a zero tensor of appropriate size.
        
    Raises:
        IndexError: If node_idx is not a valid node index in the graph
        ValueError: If aggr is not one of the supported aggregation methods
        
    Notes:
        - For weighted graphs, weights are applied before max/min aggregation
        - For mean aggregation with zero-sum weights, returns zero vector
        - Empty feature dimensions (num_features=0) are handled gracefully


    Example:
        ```python
        graph = Graph(
            node_features=torch.randn(3, 16),  # 3 nodes, 16 features each
            edge_index=torch.tensor([[0, 1], [1, 2]]),  # Two edges: 0→1, 1→2
            edge_weights=torch.tensor([0.5, 2.0])  # Edge weights
        )
        
        # Aggregate features for node 1 using weighted mean
        features = aggregate_neighbors(graph, node_idx=1, aggr="mean")
        ```
    """
    if node_idx < 0 or node_idx >= graph.num_nodes:
        raise IndexError(f"Node index {node_idx} out of range [0, {graph.num_nodes})")
    if aggr not in AGGREGATION_METHODS:
        raise ValueError(f"Unsupported aggregation mode: {aggr}. Use 'sum', 'mean', 'max', or 'min'")
    
    # Handle zero-dimension features early
    if graph.x.shape[1] == 0:
        return torch.empty(0)
    
    # Get neighbor information
    neighbors = graph.get_neighbors(node=node_idx)
    mask = graph.get_neighbors_mask(node=node_idx)
    
    # Handle edge weights
    if graph.edge_weights is not None:
        weights = graph.edge_weights[mask].view(-1, 1)  # Shape: [num_neighbors, 1]
    else:
        weights = torch.ones(len(neighbors), 1)  # Default to all ones if no weights
    
    # Handle isolated nodes
    if len(neighbors) == 0:
        return torch.zeros(graph.x.shape[1])

    # Apply weights to neighbor features
    # For max/min aggregation, weights are applied before reduction
    # Example: max(weight * feature) rather than weight * max(feature)
    messages = graph.x[neighbors] * weights  # Shape: [num_neighbors, feature_dim]
    
    # Perform aggregation based on specified method
    # TODO: Refactor the below if elif block into something that scales better
    if aggr == "sum":
        features = messages.sum(dim=0)
    elif aggr == "max":
        features = messages.max(dim=0)[0]
    elif aggr == "mean":
        # Use absolute values in denominator to handle negative weights properly
        # This treats negative weights as having the same "magnitude" for normalization
        features = messages.sum(dim=0)/(weights.abs().sum(dim=0) + 1e-8)
    elif aggr == "min":
        features = messages.min(dim=0)[0]
    
    return features
