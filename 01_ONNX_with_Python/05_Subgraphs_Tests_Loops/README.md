# Subgraphs: Tests and Loops

> **Interview Relevance:** MEDIUM — Control flow in ONNX is advanced but asked in ML infrastructure roles

---

## Visual Overview
![If Operator](dot_if_py.png)
![Scan Operator](dot_scan_py.png)

## Why (Motivation)
Real-world models sometimes need conditional logic (If), iteration over sequences (Scan), and loops (Loop). ONNX implements control flow by embedding entire sub-graphs as operator attributes, which is a unique and powerful design pattern.

## When (Use Cases)
- Conditional model outputs based on input characteristics
- Iterating over variable-length sequences (NLP, time series)
- Implementing pairwise distance computations (KNN via Scan)

## How (Mechanism)
Control flow operators (`If`, `Scan`, `Loop`) take `GraphProto` objects as attributes. The `If` operator selects between `then_branch` and `else_branch` based on a boolean input. `Scan` iterates over rows of a tensor. `Loop` combines for/while semantics with loop-carried state.

---

## Prerequisites
- [Opset and Metadata](../04_Opset_and_Metadata/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `If` operator | Conditional branching in ONNX | ⭐⭐ |
| `Scan` operator | Row-wise iteration over tensors | ⭐⭐ |
| `Loop` operator | For/while loops in ONNX | ⭐ |
| Subgraphs as attributes | Unique ONNX design pattern | ⭐⭐ |
| When to avoid control flow | Performance implications | ⭐⭐⭐ |

## Key Interview Questions Answered Here
1. **How does ONNX handle control flow?** → Through operators (If, Scan, Loop) that take entire sub-graphs as attributes.
2. **When should you avoid ONNX control flow?** → When vectorized alternatives exist (e.g., `Where` instead of `If`), since control flow prevents graph optimizations.
3. **How does the Scan operator work?** → Iterates over rows of a tensor, executing a body subgraph per row, collecting outputs.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Subgraphs_Tests_Loops_Deep_Dive.ipynb](01_Subgraphs_Tests_Loops_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Subgraphs_Tests_Loops_Apply.ipynb](02_Subgraphs_Tests_Loops_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Using `If` when `Where` (element-wise conditional) would be faster
- Forgetting that control flow prevents graph-level optimizations
- Not matching subgraph output types between then/else branches

---

## Next Steps
→ [Functions](../06_Functions/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
