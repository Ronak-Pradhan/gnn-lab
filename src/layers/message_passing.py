"""Message Passing Neural Network layer implementation.

This module implements a generic message passing layer for Graph Neural Networks (GNNs).
The layer follows the message passing paradigm where node features are updated based on
aggregated information from their neighbors.

The message passing process consists of three steps:
1. Message: Transform neighbor features (can be overridden)
2. Aggregate: Combine messages from neighbors using a specified aggregation method
3. Update: Transform aggregated messages to produce new node features (can be overridden)

Attributes:
    aggr: Aggregation method to use ('sum', 'mean', 'max', or 'min')

Example:
    ```python
    layer = MessagePassing(aggr="sum")
    x = torch.randn(5, 16)  # 5 nodes, 16 features each
    edge_index = torch.tensor([[0, 1], [1, 2]])  # Two edges: 0→1, 1→2
    out = layer(x, edge_index)  # Forward pass
    ```
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from src.data.graph import Graph
from src.layers.aggregation import aggregate_neighbors
from src.registry_and_constants import AGGREGATION_METHODS

class MessagePassing(nn.Module):
    """A generic message passing layer for Graph Neural Networks.
    
    This layer implements the message passing paradigm where each node's features
    are updated based on aggregated information from its neighbors. The layer is
    customizable through different aggregation methods and can be extended by
    overriding the message and update functions.
    
    Attributes:
        aggr: Aggregation method to use ('sum', 'mean', 'max', or 'min')
    """
    
    def __init__(self, aggr: str) -> None:
        """Initialize the message passing layer.
        
        Args:
            aggr: Aggregation method to use for combining neighbor messages.
                 Must be one of: 'sum', 'mean', 'max', 'min'
                 
        Raises:
            ValueError: If aggr is not one of the supported aggregation methods
        """
        super().__init__()
        if aggr not in AGGREGATION_METHODS:
            raise ValueError(
                f"Unsupported aggregation mode: {aggr}. "
                f"Use one of: {', '.join(AGGREGATION_METHODS)}"
            )
        self.aggr = aggr
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform message passing and update node features.
        
        Args:
            x: Node feature matrix of shape [num_nodes, num_features]
            edge_index: Edge connectivity matrix of shape [2, num_edges]
            edge_weights: Optional edge weights of shape [num_edges]
            
        Returns:
            Updated node features of shape [num_nodes, num_features]
        """
        # TODO: Plan to remove Graph initialization inside method and receive the whole graph as argument
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
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Transform neighbor features before aggregation.
        
        This is the default message function that simply passes through the
        neighbor features. Subclasses can override this to implement custom
        message transformations.
        
        Args:
            x_j: Neighbor feature matrix of shape [num_neighbors, num_features]
            
        Returns:
            Transformed neighbor features of same shape as input
        """
        return x_j
    
    def aggregate(self, messages: torch.Tensor, edge_index: torch.Tensor) -> None:
        """Aggregate messages from neighbors.
        
        This method is a placeholder as aggregation is handled by aggregate_neighbors.
        Subclasses should not need to override this.
        
        Args:
            messages: Message matrix from neighbors
            edge_index: Edge connectivity matrix
        """
        pass
    
    def update(self, aggregated: torch.Tensor) -> torch.Tensor:
        """Transform aggregated messages to produce new node features.
        
        This is the default update function that simply passes through the
        aggregated features. Subclasses can override this to implement custom
        update transformations.
        
        Args:
            aggregated: Aggregated neighbor features
            
        Returns:
            Updated node features
        """
        return aggregated