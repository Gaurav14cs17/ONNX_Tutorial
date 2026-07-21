# Execution Providers (EPs)

> **Interview Relevance:** HIGH — EP selection and fallback behavior is a top question in ML infrastructure interviews. You must explain how ORT dispatches across hardware backends.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for EP selection flowcharts.

## Why (Motivation)
Execution Providers are ORT's mechanism for running ONNX ops on different hardware. Choosing the right EP (and understanding fallback) directly determines inference **latency**, **throughput**, and **deployment cost**. Misconfigured EPs lead to silent CPU fallback, wasted GPU resources, or runtime failures.

## When (Use Cases)
- Deploying models on NVIDIA GPUs (CUDA/TensorRT EP)
- Optimizing for Intel hardware (OpenVINO EP)
- Building cross-platform inference (DirectML, CoreML, NNAPI)
- Debugging why inference is slower than expected (EP fallback investigation)

## How (Mechanism)
EPs are registered in priority order. For each node in the graph, ORT tries the highest-priority EP first. If it can't handle the node (unsupported op, dtype, or shape), ORT falls to the next EP. CPU EP is always the ultimate fallback. `Memcpy` nodes are inserted at device boundaries.

---

## Prerequisites
- [01_ORT_Architecture](../01_ORT_Architecture/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| EP concept & role | Foundation for all hardware-specific deployment | ⭐⭐⭐ |
| EP priority & fallback | Explains unexpected CPU execution on GPU | ⭐⭐⭐ |
| CUDA / TensorRT EP | Most common GPU deployment path | ⭐⭐⭐ |
| OpenVINO / DirectML / CoreML / NNAPI | Cross-platform deployment decisions | ⭐⭐ |
| Provider options configuration | Production-grade session setup | ⭐⭐ |
| Safe EP factory pattern | Portable code across environments | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What is an Execution Provider?** → An adapter that maps abstract ONNX ops to concrete kernels on a specific compute stack (CPU, CUDA, TRT, etc.).
2. **How does EP fallback work?** → EPs are tried in priority order per node; unclaimed nodes fall to the next EP; CPU EP is the universal fallback.
3. **Why might an op not run on GPU even though CUDA EP is registered?** → Dtype not implemented, dynamic shape incompatibility, or graph structure forcing host/device transitions.
4. **How do you configure TensorRT EP?** → Via provider options tuple with cache settings; always pair with CUDA EP and CPU EP as fallbacks.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Execution_Providers_Deep_Dive.ipynb](01_Execution_Providers_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Execution_Providers_Apply.ipynb](02_Execution_Providers_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Assuming all ops run on the preferred EP — fallback can silently move ops to CPU.
- Not checking `get_available_providers()` before using an EP — wrong ORT wheel = missing EP.
- Ignoring TensorRT engine caching — first-run latency includes engine compilation.
- Treating EP performance rankings as universal — always benchmark on your specific model and hardware.

---

## Next Steps
→ [03_Inference_Sessions](../03_Inference_Sessions/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
