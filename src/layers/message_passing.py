"""Message Passing Neural Network layer implementation."""

from typing import Callable, Optional

import torch
import torch.nn as nn

from src.data.graph import Graph
from src.layers.aggregation import gather_neighbor_rows, reduce_neighbor_messages
from src.registry_and_constants import AGGREGATION_METHODS


class MessagePassing(nn.Module):
    r"""A generic message passing layer for Graph Neural Networks.

    Each node's output is computed from aggregated transformed messages from
    in-neighbors, then an optional update that may use the node's own features.

    Steps per target node ``i``:
        1. :func:`~src.layers.aggregation.gather_neighbor_rows` stacks weighted source rows for edges into ``i``.
        2. **Message**: ``m_{j\\to i} = message(x_i, x_j \\odot w)`` (default: pass-through).
        3. **Aggregate**: combine ``m`` with ``aggr`` (same rules as :func:`~src.layers.aggregation.aggregate_neighbors`).
        4. **Update**: ``out_i = update(aggregated, x_i)`` (default: return ``aggregated``).

    Attributes:
        aggr: Aggregation method (``sum``, ``mean``, ``max``, ``min``).
        aggregate_fn: Optional custom aggregation callable. If provided, this is
            used by :meth:`aggregate` instead of the default reducer.

    Implemented per-target node equations:

    .. math::
        m_{j\to i} = \mathrm{message}(x_i, x_j \odot w_{j\to i})

    .. math::
        \bar{m}_i = \mathrm{aggregate}( \{ m_{j\to i} \}_{j\in\mathcal{N}(i)} )

    .. math::
        y_i = \mathrm{update}(\bar{m}_i, x_i)

    Example:
        ``layer = MessagePassing(aggr="sum")`` then ``out = layer(x, edge_index)``.
    """

    def __init__(
        self,
        aggr: str,
        aggregate_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if aggr not in AGGREGATION_METHODS:
            raise ValueError(
                f"Unsupported aggregation mode: {aggr}. Use one of: {', '.join(AGGREGATION_METHODS)}"
            )
        self.aggr = aggr
        self.aggregate_fn = aggregate_fn

    def aggregate(
        self,
        messages: torch.Tensor,
        weights: torch.Tensor,
        x_i: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate transformed neighbor rows for one target node.

        Subclasses can override this to implement custom reduction logic.
        The default path mirrors :func:`src.layers.aggregation.aggregate_neighbors`
        semantics and returns zeros for nodes with no gathered neighbors.
        """
        if self.aggregate_fn is not None:
            return self.aggregate_fn(messages, weights, x_i)
        if messages.shape[0] == 0:
            return torch.zeros_like(x_i)
        return reduce_neighbor_messages(messages, weights, self.aggr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Consider accepting a Graph object directly for callers that
        # already materialize graph structure outside this layer.
        graph = Graph(x, edge_index, edge_weights)
        per_node_outputs = []

        for node_idx in range(graph.num_nodes):
            weighted_x_j, weights = gather_neighbor_rows(graph, node_idx)
            messages = self.message(x[node_idx], weighted_x_j)
            aggregated = self.aggregate(messages, weights, x[node_idx])
            per_node_outputs.append(self.update(aggregated, x[node_idx]))

        if len(per_node_outputs) == 0:
            # Empty graph: preserve dtype/device and keep a valid 2D tensor.
            return torch.empty((0, x.shape[1]), device=x.device, dtype=x.dtype)
        return torch.stack(per_node_outputs, dim=0)

    def message(self, x_i: torch.Tensor, weighted_x_j: torch.Tensor) -> torch.Tensor:
        """Transform weighted neighbor feature rows before aggregation.

        Args:
            x_i: Features of the target node, shape ``[num_features]``.
            weighted_x_j: Rows ``x_j * w_{j\\to i}``, shape ``[num_neighbors, num_features]``.

        Returns:
            Message tensor of the same shape as ``weighted_x_j`` (unless a subclass
            intentionally changes feature dimension before a custom aggregate).
        """
        return weighted_x_j

    def update(self, aggregated: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """Combine aggregated neighbor messages with the node's own features.

        Args:
            aggregated: Reduced messages, shape ``[num_features]`` (or empty).
            x_i: Current node features, shape ``[num_features]``.

        Returns:
            Updated node features for this node, shape ``[num_features]``.
        """
        return aggregated
