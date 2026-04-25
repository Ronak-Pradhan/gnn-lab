"""Graph Convolutional Network layer (Kipf & Welling)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.layers.message_passing import MessagePassing


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Append one self-loop per node. ``edge_index`` has shape ``[2, num_edges]``."""
    device = edge_index.device
    loop_index = torch.arange(num_nodes, device=device)
    loops = torch.stack([loop_index, loop_index], dim=0)
    return torch.cat([edge_index, loops], dim=1)


def symmetric_normalized_gcn(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return edge_index with self-loops and symmetric normalization weights.

    Weight for edge ``j -> i`` is ``(deg(i)+loop)^{-1/2} (deg(j)+loop)^{-1/2}``
    with degrees counting incoming edges in the augmented graph.

    The ``\\tilde{A}`` term from ``\\tilde{D}^{-1/2}\\tilde{A}\\tilde{D}^{-1/2}``
    is represented implicitly by the edge list itself: weights are only created
    for edges present in ``edge_index`` (after self-loops are appended).
    Non-existent edges are not stored and therefore behave as zeros.
    """
    ei = add_self_loops(edge_index, num_nodes)
    source, target = ei[0], ei[1]
    deg = torch.zeros(num_nodes, device=device, dtype=dtype)
    deg.index_add_(0, target, torch.ones(ei.size(1), device=device, dtype=dtype))
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    edge_weight = deg_inv_sqrt[source] * deg_inv_sqrt[target]
    return ei, edge_weight


class GCNConv(MessagePassing):
    """One GCN layer: ``H' = \\sigma(\\hat{D}^{-1/2} \\hat{A} \\hat{D}^{-1/2} X W + b)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        super().__init__(aggr="sum")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run GCN propagation with topology-derived normalization.

        Args:
            x: Node features of shape ``[num_nodes, in_channels]``.
            edge_index: Edge list of shape ``[2, num_edges]``.
            edge_weights: Optional external edge weights.

        Returns:
            Node features of shape ``[num_nodes, out_channels]``.

        Raises:
            ValueError: If ``edge_weights`` is provided. This implementation always
                computes normalized weights from ``edge_index`` (plus self-loops),
                so external edge weights are not supported.
        """
        if edge_weights is not None:
            raise ValueError(
                "GCNConv does not accept external edge_weights; "
                "normalization weights are derived from edge_index."
            )
        ei, ew = symmetric_normalized_gcn(edge_index, x.size(0), x.dtype, x.device)
        return super().forward(x, ei, ew)

    def update(self, aggregated: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return self.lin(aggregated)
