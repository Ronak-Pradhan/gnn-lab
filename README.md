# 🧠 GNN From Scratch  

Production-grade Graph Neural Networks from first principles

[![Current Phase](https://img.shields.io/badge/Phase-2%3A%20Core%20GNN%20Layers-brightgreen)](docs/ROADMAP.md)

## 🔍 Project Overview  
Implementing fundamental GNN components with production-grade Python practices:  
- Strict type checking & docstrings  
- Comprehensive unit testing
- CI/CD integration  
- Modular architecture  

## 🚀 Current Status: Phase 2 Complete  
Implemented Features
| Component | Description | Status |
|-----------|-------------|--------|
| Graph | Node/edge representation | ✅ |
| MessagePassing | Base class for GNN layers | ✅ |
| Weighted Aggregation | Sum/Mean/Max/Min modes | ✅ |
| CI Pipeline | Automated testing | ✅ |
| GCNConv | Graph convolution layer | ✅ |
| GraphSAGEConv | Mean-aggregator GraphSAGE layer | ✅ |
| GATConv | Single-head graph attention layer | ✅ |
| Cora Loader | Planetoid-style `cora.content` / `cora.cites` parser | ✅ |
| Node Training Utils | `train_epoch` and `evaluate` helpers | ✅ |
| Cora Training Script | End-to-end `scripts/train_cora.py` | ✅ |

**🔜 Coming in Phase 3**: Optimization and extensive benchmarking (including cross-library comparisons) - [View Roadmap](docs/ROADMAP.md)

⚙️ Installation

```bash
git clone https://github.com/Ronak-Pradhan/gnn-lab.git
cd gnn-lab
pip install -r requirements.txt

# Run tests
pytest tests/
```
💻 Basic Usage
```python
import torch
from src.data import Graph
from src.layers import MessagePassing

# Undirected line graph 0 — 1 — 2 (two edges each direction)
node_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
graph = Graph(
    node_features=node_features,
    edge_index=edge_index,
    directed=False,
)

mp = MessagePassing(aggr="mean")
output = mp(graph.x, graph.edge_index)
print(f"Output shape: {output.shape}")  # [3, 2]
```

📊 Performance

Treat **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** as the only place for benchmark numbers. The Phase 2 release baseline there was recorded on **Windows**, **Python 3.13.5**, **PyTorch 2.8.0+cpu**, using `python tests/profile_phase2_release.py` (defaults: 3 warmups, 8 timed runs, 1 CPU thread). Example from that table: **10,000 nodes / 50,000 edges**, `sum` aggregation, legacy vs current medians both ~3.98s, ratio **1.00×**. Cora mini end-to-end timings are in the same doc. Overall, that baseline shows the refactored `MessagePassing` matches the legacy path on the largest synthetic case we timed, with modest slowdowns on some smaller configs and no sign of a large regression; see the tables in the performance doc for detail.

## 📚 Documentation
- [**ARCHITECTURE**](docs/ARCHITECTURE.md) : Component design and data flow  
- [**PERFORMANCE**](docs/PERFORMANCE.md) : Benchmark results and test methodology  
- [**ROADMAP**](docs/ROADMAP.md) : Detailed phase breakdown and future plans  
