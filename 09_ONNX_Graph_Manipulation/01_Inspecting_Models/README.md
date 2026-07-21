# Inspecting Models

> **Interview Relevance:** HIGH — Model inspection is the first skill for debugging ONNX issues. Interviewers expect you to programmatically summarize a model's IO, node histogram, and initializer stats.

---

## Visual Overview
![ONNX Linear Regression Graph — Y = XA + B with MatMul and Add operators](assets/dot_linreg.png)

![ONNX Scan Operator — KNN regression using iterative scan](assets/dot_scan_py.png)
See `assets/` for inspection workflow diagrams.

## Why (Motivation)
Visualization gives intuition; Python APIs give scale. Inspecting models lets you understand IO contracts, find exporter oddities, verify quantization markers, and generate CI-ready summaries. It's the foundation for all graph manipulation.

## When (Use Cases)
- Debugging export issues (unexpected ops, wrong shapes)
- Generating model summaries for documentation or CI logs
- Verifying quantization patterns (QDQ nodes)
- Understanding model structure before modification
- Extracting subgraphs for debugging or unit tests

## How (Mechanism)
Load `ModelProto` via `onnx.load()`, then walk `graph.input`, `graph.output`, `graph.node`, and `graph.initializer`. Build op histograms, parameter counts, and dtype breakdowns. Use Netron for visual exploration. Extract subgraphs by traversing value producers backward from target outputs.

---

## Prerequisites
- Basic ONNX graph understanding (nodes, initializers, value names)
- NumPy for shapes and dtypes

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Netron visualization | Quick structural understanding | ⭐⭐ |
| Programmatic model inspection | Automated CI/CD integration | ⭐⭐⭐ |
| Op histogram generation | Profile model complexity | ⭐⭐ |
| Initializer/parameter counting | Model size analysis | ⭐⭐⭐ |
| Subgraph extraction | Debugging and unit testing | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you inspect an ONNX model programmatically?** → Load with `onnx.load()`, walk `graph.input/output/node/initializer`, generate op histograms and parameter counts.
2. **What tools do you use to visualize ONNX models?** → Netron for interactive exploration; `onnx` Python API for automated summaries and CI integration.
3. **How do you find the parameter count of an ONNX model?** → Sum `numpy_helper.to_array(init).size` for all float initializers in `graph.initializer`.
4. **How do you extract a subgraph?** → Identify target output tensor, traverse backward through node outputs to collect reachable nodes, build new GraphProto.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Inspecting_Models_Deep_Dive.ipynb](01_Inspecting_Models_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Inspecting_Models_Apply.ipynb](02_Inspecting_Models_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Forgetting to set `load_external_data=True` for models with external data files.
- Printing every node for large models instead of using histograms and sampling.
- Confusing graph inputs (true feeds) with initializer-backed inputs (weights appear in `graph.input` in some ONNX versions).
- Not checking `dim_param` (symbolic dimensions) when reading shapes.

---

## Next Steps
→ [02_Modifying_Graphs](../02_Modifying_Graphs/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
