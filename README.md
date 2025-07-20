# 🧠 GNN From Scratch  

Production-grade Graph Neural Networks from first principles

[![Current Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-brightgreen)](docs/ROADMAP.md)

## 🔍 Project Overview  
Implementing fundamental GNN components with production-grade Python practices:  
- Strict type checking & docstrings  
- Comprehensive unit testing
- CI/CD integration  
- Modular architecture  

## 🚀 Current Status: Phase 1 Complete  
Implemented Features
| Component | Description | Status |
|-----------|-------------|--------|
| Graph | Node/edge representation | ✅ |
| MessagePassing | Base class for GNN layers | ✅ |
| Weighted Aggregation | Sum/Mean/Max/Min modes | ✅ |
| CI Pipeline | Automated testing | ✅ |

**🔜 Coming in Phase 2**: Core GNN Layers (GCN, GraphSAGE, GAT) - [View Roadmap](docs/ROADMAP.md)

⚙️ Installation

```bash
git clone https://github.com/Ronak-Pradhan/gnn-lab.git
cd gnn-lab
pip install torch pytest

# Run tests
pytest tests/
```
💻 Basic Usage
```python
from src.data import Graph
from src.layers import MessagePassing

# Create sample graph
node_features = torch.tensor([...])
edge_index = torch.tensor([...])
graph = Graph(node_features, edge_index)

# Initialize message passing
mp = MessagePassing(aggr="mean")

# Process graph
output = mp(graph.x, graph.edge_index)
print(f"Output shape: {output.shape}")
```

📊 Performance Baseline
| Scenario | Throughput |
|----------|------------|
| Small Graphs (100 nodes) | >40k nodes/sec |
| Medium Graphs (1k nodes) | >30k nodes/sec |
| Large Graphs (20k nodes) | >13k nodes/sec |

*Note: Current implementation focuses on correctness over optimization - see roadmap for planned improvements*  
[Full benchmark details](docs/PERFORMANCE.md)

## 📚 Documentation
- [**ARCHITECTURE**](docs/ARCHITECTURE.md) : Component design and data flow  
- [**PERFORMANCE**](docs/PERFORMANCE.md) : Benchmark results and test methodology  
- [**ROADMAP**](docs/ROADMAP.md) : Detailed phase breakdown and future plans  
