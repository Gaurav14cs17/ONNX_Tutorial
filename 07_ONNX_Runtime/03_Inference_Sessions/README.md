# Inference Sessions

> **Interview Relevance:** HIGH — The `InferenceSession` API is the day-to-day surface for ORT. Interviewers expect you to configure sessions, handle dynamic shapes, and explain the session lifecycle.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for session lifecycle diagrams.

## Why (Motivation)
`InferenceSession` is where ONNX models meet production code. Understanding session configuration, IO binding, dynamic shapes, and error handling separates a prototype from production-grade inference. Most ORT bugs in production stem from misconfigured sessions.

## When (Use Cases)
- Building inference services (REST/gRPC endpoints)
- Embedding inference in applications (Python, C++, C#)
- Handling variable-length inputs (NLP, time series)
- Optimizing GPU inference with IOBinding

## How (Mechanism)
Session creation loads the ONNX protobuf, builds an internal IR, runs graph transforms, and partitions across EPs — this is expensive. `run()` reuses the compiled plan with user-provided input tensors. `IOBinding` bypasses host-device copies for GPU workloads. Dynamic shapes are handled via symbolic dimensions exported from the training framework.

---

## Prerequisites
- [01_ORT_Architecture](../01_ORT_Architecture/)
- [02_Execution_Providers](../02_Execution_Providers/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| InferenceSession creation & config | Core API for all ORT usage | ⭐⭐⭐ |
| SessionOptions vs RunOptions | Session-wide vs per-request controls | ⭐⭐⭐ |
| IOBinding | Eliminates GPU copy overhead | ⭐⭐ |
| Dynamic shapes handling | Real-world models have variable dimensions | ⭐⭐⭐ |
| Batch inference strategies | Throughput vs latency tradeoffs | ⭐⭐ |
| Error handling patterns | Production robustness | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What is the difference between SessionOptions and RunOptions?** → SessionOptions configures session-wide behavior (threading, optimization level); RunOptions provides per-invocation controls (logging, cancellation).
2. **Why use IOBinding instead of session.run()?** → IOBinding avoids host-device copies by binding pre-allocated device buffers, critical for high-QPS GPU inference.
3. **How do you handle dynamic shapes in ORT?** → Read `sess.get_inputs()[0].shape` for symbolic dims; allocate numpy arrays with actual shapes per request; use padding + masks for NLP.
4. **What batch inference strategy would you use for a latency-sensitive service?** → Micro-batching with bounded wait time; balances throughput and tail latency.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Inference_Sessions_Deep_Dive.ipynb](01_Inference_Sessions_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Inference_Sessions_Apply.ipynb](02_Inference_Sessions_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Recreating sessions per request — session creation includes graph optimization and is expensive.
- Ignoring warm-up runs — first runs include one-time EP compilation and allocator setup.
- Not validating input shapes before `run()` — leads to cryptic runtime errors.
- Assuming `IOBinding` is always faster — it helps on GPU with large tensors but adds complexity.

---

## Next Steps
→ [04_Performance_Tuning](../04_Performance_Tuning/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
