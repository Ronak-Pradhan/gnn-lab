"""Registry and constants for the GNN implementation.

This module contains shared type definitions, constants, and registry entries
used throughout the GNN codebase to ensure consistency and avoid duplication.
"""

from typing import List

# List of supported aggregation methods (single source of truth)
AGGREGATION_METHODS: List[str] = ["sum", "mean", "max", "min"]