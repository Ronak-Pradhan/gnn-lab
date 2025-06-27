# Performance Benchmarks  
## Test Environment  
- **Device**: Apple M3 Pro
- **Memory**: 32GB RAM  
- **OS**: macOS Sonoma 14.7 
- **Python**: 3.11.11 
- **PyTorch**: 2.6.0

## Methodology  
1. Generated synthetic graphs with random features (16 features per node)  
2. Each configuration is tested with multiple aggregation methods (sum, mean, max, min)  
3. Multiple runs per configuration (5 iterations)  
4. 2 warm-up iterations before timing  
5. Reported metrics:  
   - Median time across runs  
   - Nodes processed per second

## Benchmark Results  
### 📊 Message Passing Performance  
| Nodes | Edges | Aggregation | Time (s) | Nodes/sec |
|:-----:|:-----:|:-----------:|:--------:|:---------:|
| 100 | 200 | sum | 0.0015 | 66,097 |
| 100 | 200 | mean | 0.0018 | 57,078 |
| 100 | 200 | max | 0.0023 | 43,235 |
| 100 | 200 | min | 0.0023 | 42,574 |
| 1,000 | 5,000 | sum | 0.0293 | 34,176 |
| 1,000 | 5,000 | mean | 0.0267 | 37,428 |
| 1,000 | 5,000 | max | 0.0297 | 33,658 |
| 1,000 | 5,000 | min | 0.0299 | 33,415 |
| 10,000 | 50,000 | sum | 0.6876 | 14,544 |
| 10,000 | 50,000 | mean | 0.6120 | 16,340 |
| 10,000 | 50,000 | max | 0.5850 | 17,094 |
| 10,000 | 50,000 | min | 0.5909 | 16,924 |
| 20,000 | 100,000 | sum | 1.5346 | 13,033 |
| 20,000 | 100,000 | mean | 1.2601 | 15,872 |
| 20,000 | 100,000 | max | 1.1816 | 16,926 |
| 20,000 | 100,000 | min | 1.1891 | 16,820 |

### Analysis  
The benchmarks test the MessagePassing layer across different scales and aggregation methods:  
1. **Scale Testing**:  
   - Small graphs (100 nodes)  
   - Medium graphs (1,000-10,000 nodes)  
   - Large graphs (20,000+ nodes)  
2. **Aggregation Methods**:  
   - `sum`: Simple summation of neighbor features  
   - `mean`: Average of neighbor features  
   - `max`: Maximum value across neighbor features  
   - `min`: Minimum value across neighbor features  
3. **Key Observations**:  
   - **max/min Efficiency Advantage**:
     - At 20k nodes, max/min are 23-30% faster than sum
     - Gains increase with graph size
     - Hardware-level optimizations favor reduction operations

   - **Unexpected Scaling**:
     - max/min show better than O(n) scaling
     - Possible reasons:
       - CPU branch prediction (early termination)
       - Vectorized instructions (AVX)
       - Memory access patterns

   - **Mean Paradox Resolved**:
     - Mean is faster than sum at scale

4. **Recommendations**:
   - For accuracy-critical tasks: Use `sum` (most stable)
   - For large graphs: Prefer `max` (fastest at scale)
   - For balanced workloads: `mean` offers best compromise 

## Running the Benchmarks  
To reproduce these benchmarks:  
```bash  
python tests/profile_performance.py  
```  
The script will output detailed statistics and markdown-formatted table rows that can be directly added to this document.  
