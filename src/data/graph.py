import torch
from typing import Optional, List, Tuple

class Graph:
    """A graph data structure for GNN computations.
    
    This class represents a graph with node features and optional edge weights.
    It supports both directed and undirected graphs and provides methods for
    accessing node features and neighbor relationships.
    
    Attributes:
        x: Node feature matrix of shape [num_nodes, num_features]
        edge_index: Edge connectivity matrix of shape [2, num_edges]
        edge_weights: Optional edge weight vector of shape [num_edges]
        directed: Boolean indicating if the graph is directed

    Example:
    ```python
    # Create an undirected graph: 0 ↔ 1 ↔ 2
    edge_index = torch.tensor([[0, 1, 1, 2],
                             [1, 0, 2, 1]])
    graph = Graph(
        node_features=torch.randn(3, 16),  # 3 nodes, 16 features
        edge_index=edge_index
    )
    
    # Access graph properties
    neighbors = graph.get_neighbors(1)  # Get neighbors of node 1
    num_nodes = graph.num_nodes       # Get total number of nodes
    ```
    """
    
    def __init__(
        self, 
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        directed: bool = False
    ) -> None:
        """Initialize a new Graph instance.
        
        Args:
            node_features: Tensor of shape [num_nodes, num_features] containing
                         feature vectors for each node
            edge_index: Tensor of shape [2, num_edges] where edge_index[0] contains
                       source nodes and edge_index[1] contains target nodes. 
                       For bidirectional edges, both edges must be mentioned separately
            edge_weights: Optional tensor of shape [num_edges] containing weights
                        for each edge. If None, edges are treated as unweighted
            directed: If True, edges are treated as directed. If False, each edge
                     is treated as bidirectional
                     
        Raises:
            ValueError: If edge_weights is provided and its length doesn't match
                       the number of edges
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
        """Get indices of nodes that are neighbors of the given node.
        
        For directed graphs, returns only in-neighbors (nodes with edges pointing
        to the given node). For undirected graphs, returns all connected nodes.
        
        Args:
            node: Index of the node to get neighbors for
            
        Returns:
            Tensor containing indices of neighboring nodes
        """
        return self.edge_index[0][self.edge_index[1] == node]
    
    def get_neighbors_mask(self, node: int) -> torch.Tensor:
        """Get a boolean mask indicating edges connected to the given node.
        
        Args:
            node: Index of the node to get the mask for
            
        Returns:
            Boolean tensor of shape [num_edges] where True indicates
            edges connected to the given node
        """
        return self.edge_index[1] == node

    @property
    def num_nodes(self) -> int:
        """Get the total number of nodes in the graph.
        
        Returns:
            Number of nodes
        """
        return self.x.shape[0]

    def is_directed(self) -> bool:
        """Check if the graph is directed.
        
        Returns:
            True if the graph is directed, False if undirected
        """
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