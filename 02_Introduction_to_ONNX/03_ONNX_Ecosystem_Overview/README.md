# ONNX Ecosystem Overview

> **Interview Relevance:** MEDIUM — Knowing the ecosystem components shows breadth of ONNX understanding

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
ONNX is more than a file format — it's an ecosystem of model collections, runtimes, converters, visualization tools, and hardware partnerships. Understanding these pieces helps you navigate the landscape and choose the right tools for your deployment scenario.

## When (Use Cases)
- When selecting a runtime for your deployment target
- When choosing a converter for your training framework
- When debugging exported models with visualization tools
- When evaluating hardware acceleration options

## How (Mechanism)
The ecosystem connects frameworks (producers) to runtimes (consumers) through ONNX as the interchange format. Converters export models to ONNX; runtimes execute them; Netron visualizes them; Model Zoo provides pre-trained baselines.

---

## Prerequisites
- [Why ONNX Matters](../02_Why_ONNX_Matters/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| ONNX Runtime | Primary inference engine | ⭐⭐⭐ |
| Execution Providers | Hardware-specific acceleration | ⭐⭐⭐ |
| Converters (torch.onnx, tf2onnx) | How models enter ONNX format | ⭐⭐⭐ |
| Netron | Visual debugging tool | ⭐⭐ |
| Model Zoo | Pre-trained baselines | ⭐ |

## Key Interview Questions Answered Here
1. **What is ONNX Runtime?** → A high-performance inference engine that executes ONNX graphs with CPU/GPU acceleration via Execution Providers.
2. **How do you convert a PyTorch model to ONNX?** → Use `torch.onnx.export()` (or newer `torch.export` workflows).
3. **What is Netron?** → A visualization tool for inspecting ONNX model graphs, debugging exports, and documenting model structure.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [ONNX_Ecosystem_Overview_Deep_Dive.ipynb](01_ONNX_Ecosystem_Overview_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [ONNX_Ecosystem_Overview_Apply.ipynb](02_ONNX_Ecosystem_Overview_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Confusing ONNX (format) with ONNX Runtime (inference engine)
- Assuming all operators work on all Execution Providers
- Not checking converter compatibility before choosing an export path

---

## Next Steps
→ [Installation_and_Setup](../04_Installation_and_Setup/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
