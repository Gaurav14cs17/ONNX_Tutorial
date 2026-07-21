# Initializers and Attributes

> **Interview Relevance:** HIGH — Distinguishing initializers from inputs and attributes from inputs is a core ONNX concept

---

## Visual Overview
![Linear Regression with Initializers](dot_linreg2.png)
![Graph with Attributes](dot_att.png)

## Why (Motivation)
In real models, trained weights must be stored inside the model (initializers), and operators like Transpose need fixed parameters (attributes). Understanding the difference between dynamic inputs, constant initializers, and static attributes is fundamental to ONNX graph construction and debugging.

## When (Use Cases)
- Embedding trained weights into exported models
- Setting operator parameters like `perm`, `axis`, `alpha`
- Debugging exported models where weights appear as inputs vs constants

## How (Mechanism)
**Initializers** are numpy arrays converted to `TensorProto` via `numpy_helper.from_array()` and passed as the 5th argument to `make_graph`. **Attributes** are keyword arguments to `make_node()` (e.g., `perm=[1, 0]`).

---

## Prerequisites
- [Serialization](../02_Serialization/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Initializers | Store trained weights inside the model | ⭐⭐⭐ |
| Attributes | Fixed operator parameters (perm, axis) | ⭐⭐⭐ |
| Inputs vs Initializers vs Attributes | Core ONNX design distinction | ⭐⭐⭐ |
| `numpy_helper.from_array()` | Convert numpy arrays to ONNX tensors | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What is the difference between an input and an initializer?** → Inputs are provided at runtime; initializers are constants stored in the model file.
2. **What are attributes in ONNX?** → Fixed parameters set at graph construction time (e.g., `perm` for Transpose), cannot change at runtime.
3. **Can an initializer be overridden?** → Yes, if the name appears in both inputs and initializers, the user can override it at runtime.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Initializers_and_Attributes_Deep_Dive.ipynb](01_Initializers_and_Attributes_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Initializers_and_Attributes_Apply.ipynb](02_Initializers_and_Attributes_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Listing weights as graph inputs instead of initializers (forces callers to provide them)
- Confusing attributes (static, compile-time) with inputs (dynamic, runtime)
- Forgetting that the `Constant` operator is the only way to convert an attribute into a data-flow value

---

## Next Steps
→ [Opset_and_Metadata](../04_Opset_and_Metadata/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
