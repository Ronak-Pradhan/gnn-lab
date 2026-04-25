"""GraphSAGE-style layer (Hamilton et al.), mean pooling variant."""

import torch
import torch.nn as nn

from src.layers.message_passing import MessagePassing


class GraphSAGEConv(MessagePassing):
    r"""GraphSAGE mean-aggregator layer.

    Uses inherited message passing pipeline and custom ``update``:

    .. math::
        \bar{x}_{\mathcal{N}(i)} = \mathrm{mean}_{j\in\mathcal{N}(i)}(x_j)

    .. math::
        y_i = W \,[x_i \,\|\, \bar{x}_{\mathcal{N}(i)}] + b

    where ``[a || b]`` is feature concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        """Initialize GraphSAGE mean layer.

        Args:
            in_channels: Input feature size.
            out_channels: Output feature size.
            bias: Whether to add learnable bias in output linear layer.
        """
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(2 * in_channels, out_channels, bias=bias)

    def update(self, aggregated: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """Apply GraphSAGE update on concatenated self/neighbor representation."""
        return self.lin(torch.cat([x_i, aggregated], dim=-1))
