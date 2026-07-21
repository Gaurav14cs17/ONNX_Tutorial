# Performance Tuning

> **Interview Relevance:** HIGH — Performance tuning questions are standard in ML engineering interviews. You must explain profiling, threading, optimization levels, and benchmarking methodology.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for profiling workflow diagrams.

## Why (Motivation)
Without profiling and tuning, you're guessing — and guessing ships regressions. Performance tuning transforms a "working" ORT deployment into one that meets **latency SLAs** and **cost efficiency targets**. The difference between a tuned and untuned session can be 2-10x.

## When (Use Cases)
- Preparing for production SLAs (p50/p95/p99 latency)
- Optimizing cost per inference on cloud infrastructure
- Debugging latency spikes or memory pressure
- Validating ORT upgrades haven't regressed performance

## How (Mechanism)
Enable ORT profiling to identify bottleneck operators. Sweep `intra_op_num_threads` and `inter_op_num_threads` with production-representative inputs. Test graph optimization levels (DISABLE → BASIC → EXTENDED → ALL). Use IOBinding on GPU to eliminate copy overhead. Always benchmark with warm-up, sufficient repetitions, and percentile statistics.

---

## Prerequisites
- [01_ORT_Architecture](../01_ORT_Architecture/)
- [02_Execution_Providers](../02_Execution_Providers/)
- [03_Inference_Sessions](../03_Inference_Sessions/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| ORT profiling | Find bottlenecks instead of guessing | ⭐⭐⭐ |
| Thread tuning (intra/inter-op) | Biggest CPU performance lever | ⭐⭐⭐ |
| Graph optimization levels | Control optimization aggressiveness | ⭐⭐ |
| Memory optimization | Prevent OOM and reduce allocator churn | ⭐⭐ |
| Benchmarking methodology | Credible measurements for production decisions | ⭐⭐⭐ |
| IOBinding for GPU throughput | Eliminate copy overhead in tight loops | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you profile ORT inference?** → Enable `so.enable_profiling = True`, run warm-up + profiled runs, call `sess.end_profiling()`, analyze the chrome trace.
2. **What benchmarking methodology do you use?** → Warm-up (discard early runs), 100+ timed iterations, report p50/p95/p99, fix clocks, separate preprocessing from inference time.
3. **How do you tune threading?** → Fix batch/input shapes, sweep intra_op first (1,2,4,...), then inter_op, re-test under load for oversubscription.
4. **What are graph optimization levels?** → DISABLE (debug), BASIC (safe rewrites), EXTENDED (aggressive fusions), ALL (widest passes); higher levels trade session creation time for faster execution.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Performance_Tuning_Deep_Dive.ipynb](01_Performance_Tuning_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Performance_Tuning_Apply.ipynb](02_Performance_Tuning_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Timing the first run only — captures one-time compilation, not steady-state latency.
- Reporting means instead of percentiles — means hide tail latency spikes.
- Thread oversubscription — 16 workers × 4 intra-op threads on 32 cores causes contention.
- Changing input shapes every iteration during benchmarks — allocator noise dominates measurements.

---

## Next Steps
→ [08_Model_Optimization_and_Quantization](../../08_Model_Optimization_and_Quantization/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
