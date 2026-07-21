# Chapter 8: Model Optimization and Quantization

This chapter bridges **exported ONNX graphs** and **deployable inference**: you will learn how to **shrink** models, **accelerate** execution, and **measure** improvements with discipline. Optimization is not a single knob—it is a pipeline of **graph rewrites**, **numerical precision changes**, **sparsity**, and **benchmarking** that must be validated against your accuracy and latency budgets.

---

## Learning Order

| # | Topic | Focus | Time |
|:-:|-------|-------|:----:|
| 1 | [01_Graph_Optimizations](./01_Graph_Optimizations/) | Constant folding, dead-node removal, operator fusion, `onnxoptimizer`, ORT graph optimizers | ~35 min |
| 2 | [02_Quantization_Techniques](./02_Quantization_Techniques/) | FP32 → FP16/INT8, static vs dynamic, QAT vs PTQ, calibration, ONNX/ORT tooling | ~35 min |
| 3 | [03_Pruning_and_Sparsity](./03_Pruning_and_Sparsity/) | Structured/unstructured pruning, sparsity in ONNX, sparse tensors, size/speed impact | ~35 min |
| 4 | [04_Benchmarking](./04_Benchmarking/) | Methodology, metrics, `onnxruntime_perf_test`, custom Python benchmarks, statistics | ~35 min |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Explain** core graph optimizations — constant folding, redundant node elimination, operator fusion — and sketch before/after graphs.
2. **Apply** `onnxoptimizer` passes and ORT `SessionOptions` graph optimization levels.
3. **Define** quantization as a mapping from FP32 to lower-precision types and articulate scale/zero-point representation.
4. **Contrast** static vs dynamic quantization, PTQ vs QAT, and describe when calibration is required.
5. **Navigate** ONNX and ORT quantization toolchains at a practical level.
6. **Describe** structured vs unstructured pruning and realistic expectations for speedups.
7. **Benchmark** models using a repeat–warmup–statistics methodology.
8. **Relate** accuracy, latency, throughput, and portability trade-offs.

---

## Prerequisites

- ONNX graph IR, operators, and opsets
- Basic ONNX Runtime sessions from Chapter 7

### Software

```bash
pip install onnx onnxruntime numpy
# Optional: pip install onnxoptimizer onnxruntime-gpu
```

---

## Runnable Examples

| Path | Purpose |
|------|---------|
| [02_Quantization_Techniques/quantization_example.py](./02_Quantization_Techniques/quantization_example.py) | End-to-end quantization demo |
| [04_Benchmarking/benchmark_example.py](./04_Benchmarking/benchmark_example.py) | Reproducible latency/throughput measurement |

---

## Diagrams

| Path | Description |
|------|-------------|
| [diagrams/optimization_pipeline.md](./diagrams/optimization_pipeline.md) | Mermaid: end-to-end optimization pipeline |

---

*Next: [09_ONNX_Graph_Manipulation →](../09_ONNX_Graph_Manipulation/README.md)*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
