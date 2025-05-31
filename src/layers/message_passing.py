import torch.nn as nn
from src.data.graph import Graph
from src.layers.aggregation import aggregate_neighbors

aggregation_methods = ["sum", "mean", "max", "min"]

class MessagePassing(nn.Module):
    def __init__(self, aggr: str):
        # Initialize parameters
        super().__init__()
        if aggr not in aggregation_methods:
            raise ValueError(f"Unsupported aggregation mode: {aggr}. Use 'sum', 'mean', 'max', or 'min'")
        self.aggr = aggr
    
    def forward(self, x, edge_index, edge_weights=None):
        # Create graph from input tensors
        graph = Graph(x, edge_index, edge_weights)
        
        # Initialize output tensor
        out = torch.zeros_like(x)
        
        # Process each node
        for node_idx in range(graph.num_nodes):
            # Get aggregated features from neighbors
            aggregated = aggregate_neighbors(graph, node_idx, self.aggr)
            # Update node features
            out[node_idx] = self.update(aggregated)
            
        return out
    
    def message(self, x_j):
        # Default message function - can be overridden by subclasses
        return x_j
    
    def aggregate(self, messages, edge_index):
        # This is handled by aggregate_neighbors
        pass
    
    def update(self, aggregated):
        # Default update function - can be overridden by subclasses
        return aggregated