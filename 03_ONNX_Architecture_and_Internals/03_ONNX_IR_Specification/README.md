# ONNX IR Specification

> **Interview Relevance:** HIGH — Understanding the IR hierarchy and versioning is essential for ONNX compatibility

---

## Visual Overview
![If Operator](dot_if_py.png)
![Scan Operator](dot_scan_py.png)

## Why (Motivation)
The ONNX Intermediate Representation (IR) is a versioned language built on Protocol Buffers. Understanding the `ModelProto` → `GraphProto` → `NodeProto` hierarchy, attribute types, subgraph embedding, and version compatibility is essential for building robust ONNX pipelines.

## When (Use Cases)
- When debugging IR version mismatches between tools
- When inspecting nested subgraphs (If/Loop/Scan control flow)
- When building tools that manipulate ONNX models programmatically
- When managing opset compatibility across export/runtime versions

## How (Mechanism)
`ModelProto` is the root containing IR version, opset imports, and the primary `GraphProto`. Graphs contain `NodeProto` instances, `ValueInfoProto` for typing, and `TensorProto` for constants. Attributes encode static parameters including nested `GraphProto` for control flow.

---

## Prerequisites
- [Nodes, Edges, and Tensors](../02_Nodes_Edges_and_Tensors/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Proto hierarchy | Navigate any ONNX model | ⭐⭐⭐ |
| IR version vs OpSet | Two independent version axes | ⭐⭐⭐ |
| Attribute types | Static parameters (including graphs) | ⭐⭐ |
| Nested subgraphs | Control flow (If, Loop, Scan) | ⭐⭐ |
| FunctionProto | Reusable operator definitions | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What is the ONNX IR?** → A versioned specification defining how models are serialized as Protocol Buffers, with ModelProto → GraphProto → NodeProto hierarchy.
2. **What is the difference between IR version and OpSet version?** → IR version controls container grammar (nesting, metadata); OpSet controls operator definitions (schemas, attributes).
3. **Where do nested graphs live in ONNX?** → Inside `GraphProto`-typed attributes of control flow operators (If, Loop, Scan) and in model-level function definitions.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [ONNX_IR_Specification_Deep_Dive.ipynb](01_ONNX_IR_Specification_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [ONNX_IR_Specification_Apply.ipynb](02_ONNX_IR_Specification_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Confusing IR version with OpSet version (they evolve independently)
- Not realizing a model can contain multiple GraphProto objects (nested in attributes)
- Assuming the checker catches all runtime issues (it validates structure, not numerical correctness)

---

## Next Steps
→ [Type_System_and_Shapes](../04_Type_System_and_Shapes/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
