# Chapter 7: ONNX Runtime (ORT)

This chapter moves from the **ONNX file on disk** to **fast, portable inference** on real devices. **ONNX Runtime (ORT)** is the reference runtime for ONNX models: it loads graphs, **optimizes** them, **partitions** subgraphs across **Execution Providers**, and **executes** with a careful **memory** and **threading** model tuned for servers, desktops, and edge devices.

---

## Learning Order

| # | Topic | Focus | Time |
|:-:|-------|-------|:----:|
| 1 | [01_ORT_Architecture](./01_ORT_Architecture/) | Internal pipeline, partitioner, memory, threads, session lifecycle | ~35 min |
| 2 | [02_Execution_Providers](./02_Execution_Providers/) | EP concept, CPU/CUDA/TRT/OpenVINO/DirectML/CoreML/NNAPI, registration, fallback | ~35 min |
| 3 | [03_Inference_Sessions](./03_Inference_Sessions/) | `InferenceSession`, `SessionOptions`, `RunOptions`, `IOBinding`, dynamic shapes | ~35 min |
| 4 | [04_Performance_Tuning](./04_Performance_Tuning/) | Profiling, thread tuning, memory, graph optimization levels, benchmarking | ~35 min |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Explain** how ORT turns an ONNX graph into optimized execution plans through graph optimizations, partitioning across EPs, and kernel execution.
2. **Describe** the session lifecycle (load → optimize → plan → run) and how RunOptions vs SessionOptions shape latency, determinism, and concurrency.
3. **Select and prioritize** EPs for a target platform and reason about fallback behavior.
4. **Configure** `SessionOptions` for threading, graph optimization level, and memory patterns.
5. **Use** `IOBinding` to reduce host–device copies and improve GPU throughput.
6. **Handle** dynamic shapes and apply batch inference strategies.
7. **Profile** ORT runs, tune threads, and benchmark with reproducible methodology.
8. **Contrast** ORT with TensorRT, OpenVINO, and Apache TVM.

---

## Prerequisites

- Chapters on ONNX IR, operators/opsets, and exporting models
- Basic understanding of tensor layouts, batch dimensions, and heterogeneous computing

### Software

```bash
pip install onnx onnxruntime numpy
# Optional GPU: pip install onnxruntime-gpu
```

---

## Diagrams

| Path | Description |
|------|-------------|
| [diagrams/ort_architecture.md](./diagrams/ort_architecture.md) | Mermaid: ORT stack with EPs and execution flow |

---

## Big Picture

```
Training framework    Export              Deployment
  PyTorch / TF /  →  ONNX model    →   ONNX Runtime
  JAX / sklearn      (graph+init)       + Graph optimizations
                                         + Partitioning → EP kernels
                                         + Memory allocators
                                                ↓
                                         Your application
```

---

*Next: [08_Model_Optimization_and_Quantization →](../08_Model_Optimization_and_Quantization/README.md)*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
