"""Registry and constants for the GNN implementation.

This module contains shared type definitions, constants, and registry entries
used throughout the GNN codebase to ensure consistency and avoid duplication.
"""

from typing import Callable, Dict, List

import torch

# List of supported aggregation methods (single source of truth)
AGGREGATION_METHODS: List[str] = ["sum", "mean", "max", "min"]

# Per-row reducers for stacked neighbor messages (messages, weights) -> [F]
NEIGHBOR_ROW_REDUCERS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "sum": lambda m, _: m.sum(dim=0),
    "max": lambda m, _: m.max(dim=0)[0],
    "mean": lambda m, w: m.sum(dim=0) / (w.abs().sum(dim=0) + 1e-8),
    "min": lambda m, _: m.min(dim=0)[0],
}
