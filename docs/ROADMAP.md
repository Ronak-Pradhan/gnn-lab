# Development Roadmap  
## Phase 1: Message Passing Foundations (Complete)  
- Graph data structure implementation  
- Weighted aggregation methods  
- MessagePassing base class  
- CI pipeline setup  

## Phase 2: Core GNN Layers (Complete)  
- GCN Layer (Kipf & Welling)  
- GraphSAGE Layer (Hamilton et al.)  
- GAT Layer (Veličković et al.)  
- Cora dataset loader  
- Training pipeline 

## Phase 3: Optimization  
- Vectorized message passing  
- GPU acceleration  
- Sparse matrix operations  
- Memory usage optimization  
- Extensive benchmarking and profiling:
  - Baseline vs optimized MessagePassing comparisons
  - Layer-level benchmarks for GCN/GraphSAGE/GAT (forward + training step)
  - Real-dataset benchmarks on Cora (loader time, epoch time, eval throughput)
  - Regression tracking and reproducibility metadata (seed, device, PyTorch version)
  - Cross-library comparisons (e.g., PyG/DGL) under matched configs
  - Root-cause analysis of performance gaps (kernel fusion, sparse ops, memory layout, Python overhead)

## Phase 4: Productionization   
- Model serialization  
- ONNX export