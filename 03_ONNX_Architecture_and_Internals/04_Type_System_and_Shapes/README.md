# Type System and Shapes

> **Interview Relevance:** HIGH — Shape inference and dynamic dimensions are critical for ONNX deployment

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
ONNX's type system goes beyond dense tensors to include Sequences, Maps, and Optionals. Shape inference propagates static shape information through the graph, enabling runtime optimizations like memory pre-allocation, operator fusion, and in-place computation.

## When (Use Cases)
- When exporting models with dynamic batch sizes or variable-length sequences
- When debugging shape mismatch errors in exported models
- When optimizing models (shape info enables fusion and memory planning)
- When working with non-standard types (sequences, maps in preprocessing)

## How (Mechanism)
`TypeProto` carries one of `tensor_type`, `sequence_type`, `map_type`, `optional_type`, or `sparse_tensor_type`. Tensor shapes use `dim_value` (static), `dim_param` (symbolic), or unset (unknown). `onnx.shape_inference.infer_shapes()` walks the graph using operator schemas to propagate shape info.

---

## Prerequisites
- [ONNX IR Specification](../03_ONNX_IR_Specification/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Tensor type system | Foundation for all ONNX types | ⭐⭐⭐ |
| Symbolic dimensions | Dynamic batch sizes | ⭐⭐⭐ |
| Shape inference | Enable runtime optimizations | ⭐⭐⭐ |
| Non-tensor types | Sequence, Map, Optional | ⭐⭐ |
| dim_value vs dim_param | Static vs symbolic dimensions | ⭐⭐⭐ |

## Key Interview Questions Answered Here
1. **How does ONNX handle dynamic shapes?** → Through symbolic dimensions (`dim_param`) and unknown dimensions, allowing variable batch sizes and sequence lengths.
2. **What is shape inference?** → A schema-driven static analysis that propagates shapes through the graph without running inference.
3. **What types does ONNX support beyond tensors?** → Sequences (lists), Maps (key-value), Optionals (nullable), and SparseTensors.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Type_System_and_Shapes_Deep_Dive.ipynb](01_Type_System_and_Shapes_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Type_System_and_Shapes_Apply.ipynb](02_Type_System_and_Shapes_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Expecting shape inference to resolve data-dependent shapes (it can't)
- Confusing `dim_param` (symbolic name) with unset dimensions (unknown)
- Assuming non-tensor types work uniformly across all runtimes (coverage varies)

---

## Next Steps
→ Return to [Chapter Overview](../README.md)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
