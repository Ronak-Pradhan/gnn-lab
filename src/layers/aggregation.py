import torch
from typing import Literal, Optional
from src.data.graph import Graph

def aggregate_neighbors(
    graph: Graph,
    node_idx: int,
    aggr: Literal["sum", "mean", "max", "min"] = "mean",
) -> torch.Tensor:
    """Aggregate features from neighboring nodes.
    
    Args:
        graph: Input graph instance
        node_idx: Target node index
        aggr: Aggregation mode (sum/mean/max/min)
        
    Returns:
        Aggregated tensor of shape [num_features]
        
    Raises:
        IndexError: If node_idx is invalid
        ValueError: If aggr mode is unsupported
    """
    if node_idx < 0 or node_idx >= graph.num_nodes:
        raise IndexError(f"Node index {node_idx} out of range [0, {graph.num_nodes})")
    if aggr not in ["sum", "mean", "max", "min"]:
        raise ValueError(f"Unsupported aggregation mode: {aggr}. Use 'sum', 'mean', 'max', or 'min'")
    
    # Handle zero-dimension features early
    if graph.x.shape[1] == 0:
        return torch.empty(0)
    
    neighbors = graph.get_neighbors(node=node_idx)
    mask = graph.get_neighbors_mask(node=node_idx)
    if graph.edge_weights is not None:
        weights = graph.edge_weights[mask].view(-1, 1)  # Shape: [num_neighbors, 1]
    else:
        weights = torch.ones(len(neighbors), 1)  # Default to all ones if no weights
    
    if len(neighbors) == 0:
        return torch.zeros(graph.x.shape[1])

    # For max/min aggregation, weights are applied before reduction.
    # Example: max(weight * feature) rather than weight * max(feature).  
    messages = graph.x[neighbors] * weights  # Shape: [num_neighbors, feature_dim]
    
    if aggr == "sum":
        features = messages.sum(dim=0)
    elif aggr == "max":
        features = messages.max(dim=0)[0]
    elif aggr == "mean":
        features = messages.sum(dim=0)/(weights.sum(dim=0) + 1e-8)
    elif aggr == "min":
        features = messages.min(dim=0)[0]
    
    return features
