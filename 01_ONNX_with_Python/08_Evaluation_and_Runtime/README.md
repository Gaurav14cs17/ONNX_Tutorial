# Evaluation and Runtime

> **Interview Relevance:** HIGH — Understanding inference options is critical for deployment decisions

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
ONNX provides two inference paths: the pure-Python `ReferenceEvaluator` for debugging (verbose output, custom ops) and the C++-optimized ONNX Runtime for production (10-100x faster, GPU support). Knowing when to use each is essential for efficient ML workflows.

## When (Use Cases)
- Use `ReferenceEvaluator` during model development and debugging
- Use ONNX Runtime for production inference and benchmarking
- Use custom operators to prototype operator fusion patterns
- Use verbose mode to trace data flow and catch numerical issues

## How (Mechanism)
`ReferenceEvaluator` executes each node in Python with NumPy, supporting verbose output at 4 levels. ONNX Runtime uses C++ kernels with hardware-specific optimizations. Custom operators subclass `OpRun` for the reference evaluator.

---

## Prerequisites
- [Parsing and Checker](../07_Parsing_and_Checker/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `ReferenceEvaluator` | Debug models step-by-step | ⭐⭐ |
| ONNX Runtime inference | Production deployment | ⭐⭐⭐ |
| Verbose debugging | Trace data flow through nodes | ⭐⭐ |
| Custom operators | Extend ONNX with new ops | ⭐⭐ |
| Benchmarking | Compare evaluator performance | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you run inference with ONNX?** → Use `onnxruntime.InferenceSession` for production; `ReferenceEvaluator` for debugging.
2. **What is the ReferenceEvaluator?** → A pure-Python evaluator that executes ONNX ops with NumPy; supports verbose output for step-by-step debugging.
3. **How do you implement custom operators?** → Subclass `OpRun`, implement `_run()`, and pass to `ReferenceEvaluator(model, new_ops=[...])`.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Evaluation_and_Runtime_Deep_Dive.ipynb](01_Evaluation_and_Runtime_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Evaluation_and_Runtime_Apply.ipynb](02_Evaluation_and_Runtime_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Using `ReferenceEvaluator` for performance benchmarking (it's 10-100x slower than ORT)
- Forgetting to specify `providers=['CPUExecutionProvider']` in ORT
- Not understanding that protobuf v4+ means Python/C++ boundary has serialization overhead

---

## Next Steps
→ Return to [Chapter Overview](../README.md)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
