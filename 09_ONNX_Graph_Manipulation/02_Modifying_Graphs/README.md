# Modifying Graphs

> **Interview Relevance:** MEDIUM — Graph modification skills show up in ML platform and tooling roles. Key: safe node insertion, edge rewiring, and model merging.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for before/after rewrite diagrams.

## Why (Motivation)
Once you can read a model, you can reshape it for deployment: fixing exporter quirks, rewriting ops for a specific opset, stitching preprocessing/postprocessing into the graph, or combining models end-to-end. Graph modification automates what would otherwise be manual re-export.

## When (Use Cases)
- Fixing exporter quirks (unwanted Cast, Identity, Transpose)
- Injecting preprocessing/postprocessing into the ONNX graph
- Rewriting operators for a specific opset or EP
- Merging multi-stage models into a single artifact
- Custom fusion passes for optimization

## How (Mechanism)
Add/remove `NodeProto` entries in `graph.node`, rewire `node.input`/`node.output` strings to maintain edge consistency, update `graph.input`/`graph.output` `ValueInfoProto` entries. Use `onnx.compose.merge_models` with `io_map` for model stitching. Always validate with `check_model` and shape inference after edits.

---

## Prerequisites
- [01_Inspecting_Models](../01_Inspecting_Models/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Adding/removing nodes | Core graph surgery skill | ⭐⭐⭐ |
| Rewiring edges safely | Prevents orphaned values | ⭐⭐ |
| Replacing operators | Opset migration, EP compatibility | ⭐⭐ |
| Merging models with `onnx.compose` | Single deployment artifact | ⭐⭐ |
| Name collision handling | Safe model composition | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you add a node to an ONNX graph?** → Create with `helper.make_node()`, rewire downstream inputs to use the new node's output, insert at correct position, validate.
2. **How do you merge two ONNX models?** → Use `onnx.compose.merge_models()` with `io_map` connecting output tensors of model 1 to input tensors of model 2.
3. **What must you verify after modifying a graph?** → Run `onnx.checker.check_model()`, apply shape inference, and numerically validate outputs on representative inputs.
4. **How do you avoid name collisions when merging?** → Use `prefix1`/`prefix2` parameters or `onnx.compose.add_prefix` to namespace internal values.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Modifying_Graphs_Deep_Dive.ipynb](01_Modifying_Graphs_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Modifying_Graphs_Apply.ipynb](02_Modifying_Graphs_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Forgetting to rewire downstream consumers when removing a node — leaves dangling references.
- Not running `check_model` after edits — structural issues go undetected until runtime.
- Name collisions when merging models — use prefix utilities to namespace internal values.
- Inserting nodes without ensuring topological consistency — can cause invalid execution order.

---

## Next Steps
→ [03_Shape_Inference](../03_Shape_Inference/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
