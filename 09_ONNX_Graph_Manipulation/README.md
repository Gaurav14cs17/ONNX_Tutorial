# Chapter 9: ONNX Graph Manipulation

An ONNX file is not a black box: it is a **typed computation graph** you can **inspect**, **summarize**, **rewrite**, **reshape-infer**, and **validate** programmatically. This chapter teaches the practical skills that turn ONNX from an exported artifact into a **maintained interface** inside automated pipelines—CI, quantization tooling, and custom fusion passes.

---

## Learning Order

| # | Topic | Focus | Time |
|:-:|-------|-------|:----:|
| 1 | [01_Inspecting_Models](./01_Inspecting_Models/) | Netron visualization, `onnx` Python inspection, summaries, subgraph extraction | ~35 min |
| 2 | [02_Modifying_Graphs](./02_Modifying_Graphs/) | Add/remove nodes, rewire I/O, replace operators, merge models with `onnx.compose` | ~35 min |
| 3 | [03_Shape_Inference](./03_Shape_Inference/) | Propagate tensor shapes, `onnx.shape_inference`, dynamic shapes, partial inference | ~35 min |
| 4 | [04_Model_Validation](./04_Model_Validation/) | `onnx.checker`, conformance mindset, common errors, organizational policy tests | ~35 min |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Inspect** ONNX models using Netron and the `onnx` Python API for automation.
2. **Generate** human-readable summaries: op histograms, parameter counts, dtype breakdowns.
3. **Extract** meaningful subgraphs for debugging or unit tests.
4. **Modify** graphs safely: add/remove nodes, rewire edges, rename IO, replace operator patterns.
5. **Merge** models using `onnx.compose` while managing name collisions.
6. **Apply** `onnx.shape_inference` and interpret dynamic dimensions and partial inference.
7. **Validate** models with `onnx.checker` and explain common failures.
8. **Design** lightweight conformance tests for your organization's ONNX contract.

---

## Prerequisites

- ONNX graph basics: nodes, initializer vs activations, value names
- NumPy for shapes and dtypes

### Software

```bash
pip install onnx numpy
# Optional: pip install netron
```

---

## Companion Scripts

| Path | Purpose |
|------|---------|
| [01_Inspecting_Models/inspect_model.py](./01_Inspecting_Models/inspect_model.py) | Summarize nodes, dtypes, initializer stats |

---

## Diagrams

| Path | Description |
|------|-------------|
| [diagrams/graph_manipulation.md](./diagrams/graph_manipulation.md) | Mermaid: manipulation operations map |

---

## Big Picture

```
Load ModelProto → Inspect → Modify → Shape Infer → Validate
```

---

*Next: [Chapter_09_Deployment →](../Chapter_09_Deployment/README.md)*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
