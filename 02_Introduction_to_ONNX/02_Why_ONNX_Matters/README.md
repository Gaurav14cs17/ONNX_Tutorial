# Why ONNX Matters

> **Interview Relevance:** HIGH — The N×M to N+M argument is the most important ONNX value proposition

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
ML fragmentation creates an N×M integration burden: N training frameworks times M deployment targets. ONNX acts as a hub, reducing this to N+M integrations. This means fewer bespoke translators, clearer ownership boundaries, and reusable model artifacts.

## When (Use Cases)
- When your organization uses multiple ML frameworks
- When targeting multiple deployment platforms (cloud, edge, mobile, browser)
- When separating ML research teams from platform engineering teams
- When evaluating hardware options with fair comparisons

## How (Mechanism)
Each framework contributes one export path to ONNX. Each runtime consumes ONNX as an input contract. The ONNX file becomes the stable interchange layer between producers and consumers.

---

## Prerequisites
- [What is ONNX](../01_What_is_ONNX/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| N×M fragmentation problem | Core motivation for ONNX | ⭐⭐⭐ |
| N+M hub pattern | How ONNX solves fragmentation | ⭐⭐⭐ |
| Portability vs performance | Semantic vs performance portability | ⭐⭐ |
| When not to use ONNX | Trade-off awareness | ⭐⭐ |

## Key Interview Questions Answered Here
1. **Why does ONNX matter?** → Reduces N×M integration burden to N+M by acting as a hub between frameworks and runtimes.
2. **What are the benefits of ONNX?** → Portability, runtime optimization opportunities, flexible deployment, and clearer team boundaries.
3. **When should you NOT use ONNX?** → When exotic operators aren't supported, when fully committed to one vendor's pipeline, or when dynamic shapes cause export difficulties.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Why_ONNX_Matters_Deep_Dive.ipynb](01_Why_ONNX_Matters_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Why_ONNX_Matters_Apply.ipynb](02_Why_ONNX_Matters_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Assuming ONNX eliminates all conversion work (export paths still need engineering)
- Conflating semantic portability with performance portability
- Overlooking operator coverage gaps for specific model architectures

---

## Next Steps
→ [ONNX_Ecosystem_Overview](../03_ONNX_Ecosystem_Overview/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
