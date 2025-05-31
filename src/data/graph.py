import torch
from typing import Optional

class Graph:
    def __init__(self, 
                 node_features: torch.Tensor,
                 edge_index: torch.Tensor,
                 edge_weights: Optional[torch.Tensor] = None,
                 directed: bool = False):
        """Initialize graph structure
        
        Args:
            node_features: Tensor of shape [num_nodes, num_features]
            edge_index: Edge connections [2, num_edges]
            edge_weights: Optional tensor of shape [num_edges]
            directed: Whether graph is directed
        """
        self.x = node_features
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.directed = directed

        if edge_weights is not None:
            if edge_weights.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"Number of edge weights ({edge_weights.shape[0]}) must match "
                    f"number of edges ({edge_index.shape[1]})"
                )

    def get_neighbors(self, node: int) -> torch.Tensor:
        """Return indices of in-neighboring nodes"""
        return self.edge_index[0][self.edge_index[1] == node]
    
    def get_neighbors_mask(self, node: int) -> torch.Tensor:
        return self.edge_index[1] == node

    @property
    def num_nodes(self) -> int:
        """Number of nodes in graph"""
        return self.x.shape[0]

    def is_directed(self) -> bool:
        """Check if graph is directed"""
        return self.directed
    

# Undirected graph: 0 ↔ 1 ↔ 2
edge_index = torch.tensor([[0, 1, 1, 2],
                          [1, 0, 2, 1]])
graph = Graph(
    node_features=torch.randn(3, 16),  # 3 nodes, 16 features
    edge_index=edge_index
)

assert graph.get_neighbors(1).tolist() == [0, 2]
assert graph.is_directed() == False
assert graph.num_nodes == 3