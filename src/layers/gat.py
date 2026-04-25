"""Graph Attention Network layer (Veličković et al.).

This module implements a readable single-head GAT layer using an explicit
per-target-node loop. The implementation favors clarity over vectorized speed
and is intended as an educational baseline for future optimization.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.message_passing import MessagePassing


class GATConv(MessagePassing):
    r"""Single-head Graph Attention Convolution.

    For each target node ``i``:
    1. Project node features with learnable source/target linears.
    2. Compute attention logits for each incoming edge ``j -> i``.
    3. Apply masked softmax over only ``i``'s incoming neighbors.
    4. Aggregate projected neighbor features with attention weights.

    Notes:
        - This implementation follows the graph convention in ``src.data.Graph``
          where neighbors returned for ``i`` are source nodes of incoming edges.
        - This layer does not accept external ``edge_weights`` because attention
          computes its own normalized per-edge coefficients.
        - Isolated nodes fall back to their projected target representation
          (plus optional bias), so outputs remain well-defined.

    Implemented equations (single head):

    .. math::
        h_i' = W_{dst} x_i,\\quad h_j' = W_{src} x_j

    .. math::
        e_{ij} = \mathrm{LeakyReLU}\big(a_{src}^{\top} h_i' + a_{dst}^{\top} h_j'\big)

    .. math::
        \alpha_{ij} = \mathrm{softmax}_{j \in \mathcal{N}(i)}(e_{ij})

    .. math::
        y_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} h_j' + b
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        """Initialize a single-head GAT layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension per node.
            dropout: Dropout probability applied to normalized attention weights.
            negative_slope: Negative slope for LeakyReLU in attention logits.
            bias: If True, add learnable bias to output features.
        """
        super().__init__(aggr="sum")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, 1, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, 1, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters with Xavier initialization."""
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute one GAT message passing step.

        Args:
            x: Node feature matrix of shape ``[num_nodes, in_channels]``.
            edge_index: Edge list tensor of shape ``[2, num_edges]`` where
                ``edge_index[0]`` are sources and ``edge_index[1]`` are targets.
            edge_weights: Optional external edge weights.

        Returns:
            Tensor of shape ``[num_nodes, out_channels]``.

        Raises:
            ValueError: If ``x`` does not match ``in_channels``.
            ValueError: If ``edge_weights`` is provided.
        """
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected x.shape[1] == in_channels ({self.in_channels}), got {x.size(-1)}"
            )
        if edge_weights is not None:
            raise ValueError(
                "GATConv does not accept external edge_weights; "
                "attention coefficients are derived from node features."
            )
        return super().forward(x, edge_index, edge_weights=None)

    def message(self, x_i: torch.Tensor, weighted_x_j: torch.Tensor) -> torch.Tensor:
        """Project each incoming neighbor row into attention feature space."""
        del x_i
        return self.lin_src(weighted_x_j)

    def aggregate(self, messages: torch.Tensor, weights: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """Apply additive attention and weighted neighbor sum for one target node."""
        del weights  # External edge weights are unsupported for this layer.
        h_i = self.lin_dst(x_i).unsqueeze(0)
        if messages.shape[0] == 0:
            # Keep isolated nodes stable by using projected self features.
            return h_i.squeeze(0)

        h_j = messages
        a_src = self.att_src.view(-1)
        a_dst = self.att_dst.view(-1)
        # Additive attention: a^T[Wh_i || Wh_j] implemented as
        # separate source/target projections summed per incoming edge.
        score_i = (h_i * a_src).sum(dim=-1)
        score_j = (h_j * a_dst).sum(dim=-1)
        e = self.leaky_relu(score_i + score_j)
        # Normalize over this target node's incoming neighborhood only.
        alpha = torch.softmax(e, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (alpha.unsqueeze(-1) * h_j).sum(dim=0)

    def update(self, aggregated: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """Add optional bias after attention aggregation."""
        del x_i
        if self.bias is not None:
            return aggregated + self.bias
        return aggregated
