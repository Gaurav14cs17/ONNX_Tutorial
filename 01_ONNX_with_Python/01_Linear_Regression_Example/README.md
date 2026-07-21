# Linear Regression Example

> **Interview Relevance:** HIGH — Building ONNX graphs from scratch is a fundamental skill tested in ML infrastructure interviews

---

## Visual Overview
![ONNX Linear Regression Graph — Y = XA + B with MatMul and Add operators](assets/dot_linreg.png)
![Linear Regression ONNX Graph](dot_linreg.png)

## Why (Motivation)
Understanding how to construct ONNX graphs manually gives you deep insight into the ONNX format. Linear regression (`Y = XA + B`) is the simplest model that demonstrates all four core helper functions: `make_tensor_value_info`, `make_node`, `make_graph`, and `make_model`.

## When (Use Cases)
- When you need to build custom ONNX models without a framework exporter
- When debugging exported models by comparing against hand-built references
- When creating minimal test models for runtime validation

## How (Mechanism)
Decompose `Y = XA + B` into two ONNX operator nodes (`MatMul` → `Add`), declare typed inputs/outputs, wire them into a graph, wrap in a model, and validate with `check_model`.

---

## Prerequisites
- Python 3.8+ with `onnx` and `onnxruntime` installed
- Basic linear algebra (matrix multiplication)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `make_tensor_value_info` | Declares typed graph interfaces | ⭐⭐⭐ |
| `make_node` | Creates operator instances | ⭐⭐⭐ |
| `make_graph` / `make_model` | Assembles complete models | ⭐⭐⭐ |
| `check_model` | Validates structural correctness | ⭐⭐ |
| Graph inspection | Read nodes, inputs, outputs programmatically | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you build an ONNX model from scratch?** → Use `make_node` for ops, `make_graph` to wire them, `make_model` to wrap, `check_model` to validate.
2. **What are the minimum components of an ONNX graph?** → Inputs (ValueInfoProto), nodes (operators), outputs (ValueInfoProto).
3. **How does data flow through an ONNX graph?** → Through named edges — a node's output name matches the next node's input name (SSA-like).

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Linear_Regression_Example_Deep_Dive.ipynb](01_Linear_Regression_Example_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Linear_Regression_Example_Apply.ipynb](02_Linear_Regression_Example_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Forgetting that node input/output **order matters** — it must match the operator schema
- Using `None` in shapes without understanding it means "dynamic dimension"
- Not calling `check_model()` after construction — catches wiring bugs early

---

## Next Steps
→ [Serialization](../02_Serialization/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
