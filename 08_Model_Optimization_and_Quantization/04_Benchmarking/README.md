# Benchmarking

> **Interview Relevance:** HIGH — Credible benchmarking separates junior from senior ML engineers. Interviewers expect you to know warm-up, percentile reporting, and how to avoid common measurement traps.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for benchmarking pipeline diagrams.

## Why (Motivation)
Benchmarking prevents "optimization folklore" from driving production decisions. Without rigorous measurement, you can't distinguish real improvements from noise. A good benchmark answers: **how fast**, **under what load**, **with what tail latency**, and **on which hardware**.

## When (Use Cases)
- Validating optimization improvements (quantization, pruning, fusion)
- Setting production SLAs with confidence
- Comparing execution providers or hardware targets
- Regression testing after ORT or model upgrades

## How (Mechanism)
Freeze the system, run warm-up iterations (discard), collect 100+ timed runs with `time.perf_counter`, report p50/p95/p99 percentiles. Control batch size, threading, EP, input distribution, and optimization level. Use paired runs for A/B comparisons.

---

## Prerequisites
- [01_Graph_Optimizations](../01_Graph_Optimizations/)
- [03_Pruning_and_Sparsity](../03_Pruning_and_Sparsity/)
- Basic ONNX Runtime session usage

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Warm-up methodology | Eliminates one-time compilation noise | ⭐⭐⭐ |
| Percentile reporting (p50/p95/p99) | SLO-aligned latency measurement | ⭐⭐⭐ |
| Controlled variables | Reproducible comparisons | ⭐⭐⭐ |
| `onnxruntime_perf_test` tool | Standardized ORT benchmarking | ⭐⭐ |
| Statistical analysis | Distinguish signal from noise | ⭐⭐ |
| Common benchmarking traps | Avoid biased measurements | ⭐⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you benchmark an ONNX model?** → Warm-up (discard early runs), 100+ timed iterations, report p50/p95/p99 percentiles, control batch size/threading/EP.
2. **Why report percentiles instead of means?** → Means hide tail latency; p95/p99 captures the experience of worst-case users and is what SLAs measure.
3. **What variables must you control in a benchmark?** → Batch size, thread counts, EP, input shapes/distribution, optimization level, power state, and other running processes.
4. **What's wrong with timing only the first run?** → It captures one-time costs (session creation, TRT engine build, allocator warmup) that don't represent steady-state.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Benchmarking_Deep_Dive.ipynb](01_Benchmarking_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Benchmarking_Apply.ipynb](02_Benchmarking_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Reporting means instead of percentiles — hides real tail latency.
- Benchmarking on a laptop on battery — thermal throttling makes results unstable.
- Including preprocessing time inside the inference timer — misattributes bottlenecks.
- Omitting warm-up — captures one-time compilation costs as "inference latency."

---

## Next Steps
→ [09_ONNX_Graph_Manipulation](../../09_ONNX_Graph_Manipulation/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
