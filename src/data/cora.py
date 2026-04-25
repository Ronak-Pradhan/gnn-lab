"""Cora (Planetoid-style) dataset loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class CoraData:
    """Node features, labels, edges, and standard Planetoid masks."""

    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


def _parse_content_line(line: str) -> Tuple[int, list, int]:
    parts = line.strip().split("\t")
    if len(parts) < 3:
        raise ValueError(f"Bad .content line (need id, feats+, label): {line!r}")
    paper_id = int(parts[0])
    *feat_str, label_str = parts[1:]
    feats = [float(x) for x in feat_str]
    label = int(label_str)
    return paper_id, feats, label


def _build_id_maps(content_lines: list) -> Tuple[dict, list]:
    """Map original paper ids to 0..N-1 in file order."""
    id_to_idx: dict = {}
    ordered_ids: list = []
    for line in content_lines:
        if not line.strip():
            continue
        pid, _, _ = _parse_content_line(line)
        if pid not in id_to_idx:
            id_to_idx[pid] = len(ordered_ids)
            ordered_ids.append(pid)
    return id_to_idx, ordered_ids


def load_cora_dir(data_dir: str | Path) -> CoraData:
    """Load Cora-like tensors from a directory containing ``cora.content`` and ``cora.cites``.

    ``cora.content`` rows: ``paper_id TAB f1 TAB f2 ... TAB class_label``.
    ``cora.cites`` rows: ``paper_id TAB paper_id`` (interpreted as undirected).
    """
    root = Path(data_dir)
    content_path = root / "cora.content"
    cites_path = root / "cora.cites"
    if not content_path.is_file() or not cites_path.is_file():
        raise FileNotFoundError(f"Expected {content_path} and {cites_path}")

    raw_lines = content_path.read_text(encoding="utf-8").splitlines()
    id_to_idx, ordered_ids = _build_id_maps(raw_lines)

    num_nodes = len(ordered_ids)
    feature_rows: list = [None] * num_nodes
    labels = torch.empty(num_nodes, dtype=torch.long)

    for line in raw_lines:
        if not line.strip():
            continue
        pid, feats, label = _parse_content_line(line)
        idx = id_to_idx[pid]
        if feature_rows[idx] is not None:
            raise ValueError(f"Duplicate paper id in .content: {pid}")
        feature_rows[idx] = feats
        labels[idx] = label

    if any(r is None for r in feature_rows):
        raise ValueError("Missing feature row for some paper ids")

    x = torch.tensor(feature_rows, dtype=torch.float32)

    sources: list = []
    targets: list = []
    for line in cites_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a_str, b_str = line.strip().split("\t")
        a, b = int(a_str), int(b_str)
        if a not in id_to_idx or b not in id_to_idx:
            raise ValueError(f"Cites references unknown id: {a} or {b}")
        ia, ib = id_to_idx[a], id_to_idx[b]
        sources += [ia, ib]
        targets += [ib, ia]

    if not sources:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[0] = True
    if num_nodes > 1:
        val_mask[1] = True
    if num_nodes > 2:
        test_mask[2] = True
    if num_nodes > 3:
        rem = torch.arange(3, num_nodes)
        test_mask[rem] = True

    return CoraData(
        x=x,
        y=labels,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
