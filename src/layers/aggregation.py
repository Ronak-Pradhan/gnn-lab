"""Neighbor feature aggregation functionality for Graph Neural Networks."""

import torch
from typing import Tuple

from src.data.graph import Graph
from src.registry_and_constants import AGGREGATION_METHODS, NEIGHBOR_ROW_REDUCERS


def _empty_neighbor_row_stack(graph: Graph, num_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (rows, weights) with zero rows; ``num_features`` may be 0."""
    dev, dt = graph.x.device, graph.x.dtype
    w = torch.empty(0, 1, device=dev, dtype=dt)
    if num_features == 0:
        return torch.empty(0, 0, device=dev, dtype=dt), w
    return torch.empty(0, num_features, device=dev, dtype=dt), w


def gather_neighbor_rows(
    graph: Graph,
    node_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack weighted source-node rows for every edge whose **target** is ``node_idx``.

    This follows :class:`~src.data.graph.Graph` conventions: ``get_neighbors`` returns
    sources of incoming edges. To aggregate in the opposite direction later, reverse
    ``edge_index`` or extend ``Graph`` with explicit out-neighbor queries; this helper
    stays tied to one clear semantics.

    For each edge ``j -> node_idx``, returns one row ``x[j] * w`` and the matching
    scalar weight as a column (shape ``[num_edges, 1]``).

    Returns:
        Tuple of ``(weighted_rows, weights)``. If there are no such edges, both tensors
        have zero rows (and the same feature width as ``graph.x`` when it is non-zero).
    """
    if node_idx < 0 or node_idx >= graph.num_nodes:
        raise IndexError(f"Node index {node_idx} out of range [0, {graph.num_nodes})")

    num_features = graph.x.shape[1]
    if num_features == 0:
        return _empty_neighbor_row_stack(graph, 0)

    neighbors = graph.get_neighbors(node=node_idx)
    if len(neighbors) == 0:
        return _empty_neighbor_row_stack(graph, num_features)

    mask = graph.get_neighbors_mask(node=node_idx)
    if graph.edge_weights is not None:
        weights = graph.edge_weights[mask].view(-1, 1)
    else:
        weights = torch.ones(len(neighbors), 1, device=graph.x.device, dtype=graph.x.dtype)

    weighted_rows = graph.x[neighbors] * weights
    return weighted_rows, weights


def reduce_neighbor_messages(
    messages: torch.Tensor,
    weights: torch.Tensor,
    aggr: str,
) -> torch.Tensor:
    """Combine stacked per-neighbor rows with ``sum``, ``mean``, ``max``, or ``min``.

    Mean uses ``sum(messages) / (sum(|weights|) + 1e-8)``, matching
    :func:`aggregate_neighbors` when ``messages`` are the same weighted source rows
    (this function does not apply a ``message()`` transform).
    """
    if aggr not in AGGREGATION_METHODS:
        raise ValueError(f"Unsupported aggregation mode: {aggr}. Use 'sum', 'mean', 'max', or 'min'")

    if messages.shape[0] == 0:
        raise ValueError("reduce_neighbor_messages requires at least one message row")

    if messages.shape[1] == 0:
        return torch.empty(0, device=messages.device, dtype=messages.dtype)

    return NEIGHBOR_ROW_REDUCERS[aggr](messages, weights)


def aggregate_neighbors(
    graph: Graph,
    node_idx: int,
    aggr: str = "mean",
) -> torch.Tensor:
    """Aggregate neighbor rows for ``node_idx`` without a ``message()`` transform.

    Equivalent to :func:`gather_neighbor_rows` followed by :func:`reduce_neighbor_messages`.
    :class:`~src.layers.message_passing.MessagePassing` inserts ``message()`` between
    those two steps.

    Args:
        graph: Input graph.
        node_idx: Target node index.
        aggr: ``sum``, ``mean``, ``max``, or ``min``.

    Returns:
        Tensor of shape ``[num_features]``. Isolated nodes yield a zero vector.
    """
    if aggr not in AGGREGATION_METHODS:
        raise ValueError(f"Unsupported aggregation mode: {aggr}. Use 'sum', 'mean', 'max', or 'min'")

    if graph.x.shape[1] == 0:
        return torch.empty(0, device=graph.x.device, dtype=graph.x.dtype)

    weighted_rows, weights = gather_neighbor_rows(graph, node_idx)
    if weighted_rows.shape[0] == 0:
        return torch.zeros(graph.x.shape[1], device=graph.x.device, dtype=graph.x.dtype)

    return reduce_neighbor_messages(weighted_rows, weights, aggr)
