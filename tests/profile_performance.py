"""Performance profiling utilities for the GNN message passing layer.

This module provides tools to benchmark the performance of the MessagePassing layer
across different graph sizes and aggregation methods. It measures execution time
and throughput for various configurations.

Example:
    Run the benchmarks from command line:
    ```bash
    python tests/profile_performance.py
    ```
"""

import time
from typing import List, Dict, Tuple, Union, Any
import torch
import numpy as np
from src.layers.message_passing import MessagePassing
from src.data.graph import Graph
from src.registry_and_constants import AGGREGATION_METHODS

def profile_forward_pass(
    nodes: int,
    edges: int,
    aggr: str = "sum",
    num_runs: int = 5,
    feature_dim: int = 16
) -> Dict[str, Union[int, float, str]]:
    """Profile the forward pass of MessagePassing layer for given configuration.

    Args:
        nodes: Number of nodes in the synthetic graph.
        edges: Number of edges in the synthetic graph.
        aggr: Aggregation method to use ('sum', 'mean', 'max', or 'min').
        num_runs: Number of iterations to run for timing statistics.
        feature_dim: Number of features per node.

    Returns:
        Dict containing profiling statistics:
            - nodes: Number of nodes
            - edges: Number of edges
            - aggr: Aggregation method used
            - median_time: Median execution time across runs
            - mean_time: Mean execution time
            - std_time: Standard deviation of execution times
            - nodes_per_sec: Nodes processed per second (based on median time)
    """
    # Generate synthetic graph
    x = torch.randn(nodes, feature_dim)  # feature_dim features per node
    edge_index = torch.randint(0, nodes, (2, edges))
    
    # Create model
    mp = MessagePassing(aggr=aggr)
    
    # Warm-up
    for _ in range(2):
        _ = mp(x, edge_index)
    
    # Multiple runs so that any anomaly gets averaged out
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = mp(x, edge_index)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    times = np.array(times)
    median_time = np.median(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Return stats for logging
    return {
        "nodes": nodes,
        "edges": edges,
        "aggr": aggr,
        "median_time": median_time,
        "mean_time": mean_time,
        "std_time": std_time,
        "nodes_per_sec": nodes/median_time
    }

def format_markdown_row(stats: Dict[str, Union[int, float, str]]) -> str:
    """Format profiling statistics as a markdown table row.
    
    Args:
        stats: Dictionary containing profiling statistics from profile_forward_pass()
    
    Returns:
        Formatted string representing a markdown table row
    """
    return (f"| {stats['nodes']:,} | {stats['edges']:,} | {stats['aggr']} | "
            f"{stats['median_time']:.4f} | {stats['nodes_per_sec']:,.0f} |")

def get_test_configs() -> List[Tuple[int, int]]:
    """Get list of test configurations for nodes and edges.
    
    Returns:
        List of tuples containing (num_nodes, num_edges) for each test configuration
    """
    return [
        (100, 200),
        (1_000, 5_000),
        (10_000, 50_000),
        (20_000, 100_000)
    ]

def get_aggregation_methods() -> List[str]:
    """Get list of aggregation methods to test.
    
    Returns:
        List of strings representing different aggregation methods
    """
    return AGGREGATION_METHODS

if __name__ == "__main__":
    print("🔍 Performance Profiling - MessagePassing Layer")
    print("-----------------------------------------------")
    
    # Test configs
    configs = get_test_configs()
    aggregations = get_aggregation_methods()
    
    # Print detailed stats
    print(f"\n📊 Profiling Results:")
    print("\n📝 Markdown Table Rows:")
    print("| Nodes | Edges | Aggregation | Time (s) | Nodes/sec |")
    print("|:-----:|:-----:|:-----------:|:--------:|:---------:|")
    
    for nodes, edges in configs:
        for aggr in aggregations:
            stats = profile_forward_pass(nodes, edges, aggr)
            print(format_markdown_row(stats))