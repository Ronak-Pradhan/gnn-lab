"""Phase 2 release-readiness benchmarks.

This script benchmarks:
1) Legacy (Phase 1-style) vs current MessagePassing forward performance.
2) End-to-end Cora pipeline timing (load + train + evaluate) with a tiny GCN.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.cora import load_cora_dir
from src.data.graph import Graph
from src.layers.aggregation import aggregate_neighbors
from src.layers.gcn import GCNConv
from src.layers.message_passing import MessagePassing
from src.registry_and_constants import AGGREGATION_METHODS
from src.train.node_classification import evaluate, set_seed, train_epoch


class LegacyMessagePassingBaseline(nn.Module):
    """Baseline matching pre-refactor Phase 1 forward behavior."""

    def __init__(self, aggr: str) -> None:
        super().__init__()
        self.aggr = aggr

    def update(self, aggregated: torch.Tensor) -> torch.Tensor:
        return aggregated

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        graph = Graph(x, edge_index, edge_weights)
        outputs = []
        for node_idx in range(graph.num_nodes):
            aggregated = aggregate_neighbors(graph, node_idx, self.aggr)
            outputs.append(self.update(aggregated))
        if len(outputs) == 0:
            return torch.empty((0, x.shape[1]), device=x.device, dtype=x.dtype)
        return torch.stack(outputs, dim=0)


class TinyGCN(nn.Module):
    """Small 2-layer GCN for end-to-end pipeline timing."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)


def benchmark_message_passing_compare(
    nodes: int,
    edges: int,
    aggr: str,
    feature_dim: int,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float | int | str]:
    """Compare legacy vs current MessagePassing on one synthetic config."""
    x = torch.randn(nodes, feature_dim)
    edge_index = torch.randint(0, nodes, (2, edges))

    legacy = LegacyMessagePassingBaseline(aggr=aggr)
    current = MessagePassing(aggr=aggr)

    for _ in range(warmup_runs):
        _ = legacy(x, edge_index)
        _ = current(x, edge_index)

    legacy_times: List[float] = []
    current_times: List[float] = []

    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = legacy(x, edge_index)
        legacy_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        _ = current(x, edge_index)
        current_times.append(time.perf_counter() - t1)

    legacy_median = statistics.median(legacy_times)
    current_median = statistics.median(current_times)
    legacy_p10, legacy_p90 = statistics.quantiles(legacy_times, n=10)[0], statistics.quantiles(
        legacy_times, n=10
    )[8]
    current_p10, current_p90 = statistics.quantiles(current_times, n=10)[0], statistics.quantiles(
        current_times, n=10
    )[8]
    slowdown = (current_median / legacy_median) if legacy_median > 0 else float("inf")

    return {
        "nodes": nodes,
        "edges": edges,
        "aggr": aggr,
        "legacy_median_s": legacy_median,
        "legacy_p10_s": legacy_p10,
        "legacy_p90_s": legacy_p90,
        "current_median_s": current_median,
        "current_p10_s": current_p10,
        "current_p90_s": current_p90,
        "relative_current_vs_legacy": slowdown,
    }


def benchmark_cora_end_to_end(
    data_dir: Path,
    epochs: int,
    hidden: int,
    lr: float,
    seed: int,
) -> Dict[str, float | int]:
    """Benchmark Cora loading and a fixed-length GCN training run."""
    set_seed(seed)

    load_times: List[float] = []
    train_times: List[float] = []
    eval_times: List[float] = []
    test_accs: List[float] = []
    last_loss = float("nan")
    num_nodes = 0
    num_edges = 0

    for _ in range(3):
        load_start = time.perf_counter()
        data = load_cora_dir(data_dir)
        load_time = time.perf_counter() - load_start
        load_times.append(load_time)

        num_nodes = int(data.x.size(0))
        num_edges = int(data.edge_index.size(1))
        num_classes = int(data.y.max().item()) + 1
        model = TinyGCN(in_dim=data.x.size(1), hidden=hidden, num_classes=num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        train_start = time.perf_counter()
        for _ in range(epochs):
            last_loss = train_epoch(model, data.x, data.edge_index, data.y, data.train_mask, opt)
        train_time = time.perf_counter() - train_start
        train_times.append(train_time)

        eval_start = time.perf_counter()
        _, eval_acc = evaluate(model, data.x, data.edge_index, data.y, data.test_mask)
        eval_time = time.perf_counter() - eval_start
        eval_times.append(eval_time)
        test_accs.append(float(eval_acc))

    load_median = statistics.median(load_times)
    train_median = statistics.median(train_times)
    eval_median = statistics.median(eval_times)
    load_p10, load_p90 = statistics.quantiles(load_times, n=10)[0], statistics.quantiles(load_times, n=10)[8]
    train_p10, train_p90 = statistics.quantiles(train_times, n=10)[0], statistics.quantiles(train_times, n=10)[8]
    eval_p10, eval_p90 = statistics.quantiles(eval_times, n=10)[0], statistics.quantiles(eval_times, n=10)[8]

    nodes_seen = num_nodes * epochs
    nodes_per_sec = nodes_seen / train_median if train_median > 0 else 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "load_time_s": load_median,
        "load_p10_s": load_p10,
        "load_p90_s": load_p90,
        "train_time_s": train_median,
        "train_p10_s": train_p10,
        "train_p90_s": train_p90,
        "eval_time_s": eval_median,
        "eval_p10_s": eval_p10,
        "eval_p90_s": eval_p90,
        "epochs": epochs,
        "nodes_per_sec_train": nodes_per_sec,
        "final_train_loss": last_loss,
        "test_acc": statistics.median(test_accs),
    }


def get_configs() -> List[Tuple[int, int]]:
    return [
        (100, 200),
        (1_000, 5_000),
        (10_000, 50_000),
    ]


def print_environment() -> None:
    print("## Environment")
    print(f"- Platform: {platform.platform()}")
    print(f"- Python: {platform.python_version()}")
    print(f"- PyTorch: {torch.__version__}")
    print(f"- CUDA available: {torch.cuda.is_available()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 release-readiness benchmarks")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("tests/fixtures/cora_mini"),
        help="Directory with cora.content and cora.cites (default: tests/fixtures/cora_mini)",
    )
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Set torch CPU threads for reproducible microbenchmarks (default: 1).",
    )
    args = parser.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
    print_environment()
    print("\n## MessagePassing: Legacy vs Current")
    print(
        "| Nodes | Edges | Aggr | Legacy median [p10,p90] (s) | "
        "Current median [p10,p90] (s) | Current/Legacy |"
    )
    print("|:-----:|:-----:|:----:|:---------------------------:|:----------------------------:|:--------------:|")
    for nodes, edges in get_configs():
        for aggr in AGGREGATION_METHODS:
            row = benchmark_message_passing_compare(
                nodes=nodes,
                edges=edges,
                aggr=aggr,
                feature_dim=args.feature_dim,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
            )
            print(
                f"| {row['nodes']:,} | {row['edges']:,} | {row['aggr']} | "
                f"{row['legacy_median_s']:.4f} [{row['legacy_p10_s']:.4f},{row['legacy_p90_s']:.4f}] | "
                f"{row['current_median_s']:.4f} [{row['current_p10_s']:.4f},{row['current_p90_s']:.4f}] | "
                f"{row['relative_current_vs_legacy']:.2f}x |"
            )

    print("\n## Cora End-to-End (GCN)")
    cora = benchmark_cora_end_to_end(
        data_dir=args.data_dir,
        epochs=args.epochs,
        hidden=args.hidden,
        lr=args.lr,
        seed=args.seed,
    )
    print(
        "| Data dir | Nodes | Edges | Epochs | Load median [p10,p90] (s) | "
        "Train median [p10,p90] (s) | Eval median [p10,p90] (s) | Train nodes/sec | Test acc |"
    )
    print(
        "|:---------|------:|------:|------:|---------------------------:|----------------------------:|"
        "---------------------------:|----------------:|---------:|"
    )
    print(
        f"| {args.data_dir.as_posix()} | {cora['num_nodes']} | {cora['num_edges']} | {cora['epochs']} | "
        f"{cora['load_time_s']:.4f} [{cora['load_p10_s']:.4f},{cora['load_p90_s']:.4f}] | "
        f"{cora['train_time_s']:.4f} [{cora['train_p10_s']:.4f},{cora['train_p90_s']:.4f}] | "
        f"{cora['eval_time_s']:.4f} [{cora['eval_p10_s']:.4f},{cora['eval_p90_s']:.4f}] | "
        f"{cora['nodes_per_sec_train']:.2f} | {cora['test_acc']:.3f} |"
    )


if __name__ == "__main__":
    main()
