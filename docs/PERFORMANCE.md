# Performance Benchmarks

## Phase 2 Release Baseline

This section captures the minimum release-readiness benchmark for Phase 2:

1. Legacy (pre-refactor style) vs current `MessagePassing` forward timing.
2. End-to-end Cora pipeline timing (`load_cora_dir` + GCN train + evaluate).

### Test Environment

- **Platform**: Windows-11-10.0.26200-SP0
- **Python**: 3.13.5
- **PyTorch**: 2.8.0+cpu
- **CUDA**: False

### Methodology

- Synthetic benchmark settings:
  - feature dimension: 16
  - warmups: 3
  - timed runs: 8
  - CPU threads: 1
  - configs: `(100, 200)`, `(1_000, 5_000)`, `(10_000, 50_000)` as `(nodes, edges)`
  - aggregation modes: `sum`, `mean`, `max`, `min`
- Comparison metric:
  - `Current/Legacy` ratio where:
    - `1.00x` means equal median runtime
    - `>1.00x` means current is slower
    - `<1.00x` means current is faster
  - Spread reported as `p10/p90` from Python’s `statistics.quantiles(..., n=10)` (with 8 timed runs per config, these are approximate spread bands, not large-sample percentiles).
- Cora end-to-end run:
  - dataset: `tests/fixtures/cora_mini`
  - model: 2-layer `GCNConv` (`hidden=16`)
  - epochs: 30
  - optimizer: Adam (`lr=1e-2`)
  - seed: 42
  - timing: three full repetitions (each repetition loads data, trains, evaluates); reported medians and `p10/p90` are over those three runs.

### MessagePassing: Legacy vs Current

| Nodes | Edges | Aggr | Legacy median [p10,p90] (s) | Current median [p10,p90] (s) | Current/Legacy |
|:-----:|:-----:|:----:|:---------------------------:|:----------------------------:|:--------------:|
| 100 | 200 | sum | 0.0080 [0.0052,0.0098] | 0.0077 [0.0066,0.0104] | 0.96x |
| 100 | 200 | mean | 0.0106 [0.0082,0.0147] | 0.0113 [0.0085,0.0117] | 1.07x |
| 100 | 200 | max | 0.0070 [0.0064,0.0108] | 0.0073 [0.0067,0.0111] | 1.05x |
| 100 | 200 | min | 0.0072 [0.0057,0.0099] | 0.0075 [0.0065,0.0100] | 1.04x |
| 1,000 | 5,000 | sum | 0.0974 [0.0923,0.1091] | 0.1098 [0.1004,0.1149] | 1.13x |
| 1,000 | 5,000 | mean | 0.1465 [0.1282,0.2347] | 0.1509 [0.1388,0.2252] | 1.03x |
| 1,000 | 5,000 | max | 0.1162 [0.1046,0.1313] | 0.1285 [0.1199,0.1494] | 1.11x |
| 1,000 | 5,000 | min | 0.1290 [0.0984,0.1717] | 0.1357 [0.1212,0.1717] | 1.05x |
| 10,000 | 50,000 | sum | 3.9773 [3.8131,4.2633] | 3.9842 [3.9276,4.1958] | 1.00x |
| 10,000 | 50,000 | mean | 4.3864 [4.2396,4.7787] | 4.5620 [4.2177,4.7841] | 1.04x |
| 10,000 | 50,000 | max | 4.0487 [3.9031,4.3064] | 4.1610 [3.9438,4.1983] | 1.03x |
| 10,000 | 50,000 | min | 4.0944 [3.8962,4.6841] | 4.1556 [3.9926,4.2658] | 1.01x |

Conclusion: refactor is roughly performance-neutral at larger scale, with mild regressions in some smaller/mid cases. No major regression.

### Cora End-to-End (GCN)

| Data dir | Nodes | Edges | Epochs | Load median [p10,p90] (s) | Train median [p10,p90] (s) | Eval median [p10,p90] (s) | Train nodes/sec | Test acc |
|:---------|------:|------:|------:|---------------------------:|----------------------------:|---------------------------:|----------------:|---------:|
| `tests/fixtures/cora_mini` | 3 | 4 | 30 | 0.0015 [0.0014,0.0022] | 0.1026 [0.0888,0.1249] | 0.0021 [0.0006,0.0043] | 877.46 | 1.000 |

### Notes and Limitations

- The end-to-end benchmark currently uses the mini fixture for deterministic CI-style reproducibility.
- For release-level external reporting, rerun with full Cora files and record those numbers separately.
- This baseline is CPU-only and should not be generalized to GPU behavior.

## Legacy Synthetic Benchmark (Reference)

The earlier synthetic benchmark and analysis are preserved in `tests/profile_performance.py`.

## Running the Benchmarks

### Phase 2 release baseline

Defaults match the methodology above (`warmup-runs=3`, `timed-runs=8`, `num-threads=1`).

```bash
python tests/profile_phase2_release.py
```

Optional full-Cora run:

```bash
python tests/profile_phase2_release.py --data-dir path/to/cora_dir
```

### Legacy synthetic benchmark

```bash
python tests/profile_performance.py
```
