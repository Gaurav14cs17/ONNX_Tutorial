# Nodes, Edges, and Tensors

> **Interview Relevance:** HIGH — Understanding NodeProto structure and tensor types is essential for ONNX work

---

## Visual Overview
![Graph with Attributes](dot_att.png)

## Why (Motivation)
Every operator instance in ONNX is a `NodeProto` that references named values. Understanding how nodes wire together through name identity, how attributes differ from inputs, and the full tensor data type catalog is critical for reading, debugging, and building ONNX models.

## When (Use Cases)
- When reading exported models in Netron or Python
- When debugging operator input/output mismatches
- When choosing the right data type for quantization or mixed precision
- When building custom ONNX graph manipulation tools

## How (Mechanism)
`NodeProto` contains `op_type`, ordered `input`/`output` name lists, and `attribute` entries for static parameters. Edges are implicit through name identity (SSA). Tensor types are enumerated in `TensorProto.DataType`.

---

## Prerequisites
- [Computation Graph Basics](../01_Computation_Graph_Basics/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `NodeProto` anatomy | Read any ONNX node | ⭐⭐⭐ |
| Implicit edges (SSA) | Understand data flow without edge objects | ⭐⭐⭐ |
| Tensor data types | Choose types for quantization/precision | ⭐⭐ |
| Attributes vs inputs | Static params vs dynamic data | ⭐⭐⭐ |
| Input order matters | Swapping inputs rewires the op | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What are the key fields of an ONNX NodeProto?** → `op_type`, `domain`, `input` (ordered names), `output` (ordered names), `attribute` (static params).
2. **How are edges represented in ONNX?** → Through name identity; no explicit edge objects. Each name is produced once (SSA).
3. **What data types does ONNX support?** → FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT32, INT64, BOOL, STRING, and FP8/INT4 variants.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Nodes_Edges_and_Tensors_Deep_Dive.ipynb](01_Nodes_Edges_and_Tensors_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Nodes_Edges_and_Tensors_Apply.ipynb](02_Nodes_Edges_and_Tensors_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Thinking ONNX has explicit edge objects (it doesn't — edges are name-based)
- Swapping input order (each position has specific meaning per the op schema)
- Confusing `domain=""` (standard ONNX) with custom domains

---

## Next Steps
→ [ONNX_IR_Specification](../03_ONNX_IR_Specification/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
