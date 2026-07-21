# Serialization

> **Interview Relevance:** HIGH — Understanding protobuf serialization is essential for model deployment pipelines

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
ONNX uses Protocol Buffers for serialization, enabling models to be saved as portable binary files. Understanding serialization is critical for model deployment, storage, and distribution across different environments.

## When (Use Cases)
- Saving trained models to disk for later inference
- Distributing models across teams or services
- Storing test data and calibration tensors alongside models
- Building model versioning and registry systems

## How (Mechanism)
Every ONNX proto object (`ModelProto`, `TensorProto`, etc.) supports `SerializeToString()` to convert to bytes and `ParseFromString()` to reconstruct. Models are typically saved as `.onnx` files, tensors as `.pb` files.

---

## Prerequisites
- [Linear Regression Example](../01_Linear_Regression_Example/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `SerializeToString()` | Core method for saving any ONNX object | ⭐⭐⭐ |
| `onnx.load()` | Standard way to load models | ⭐⭐⭐ |
| Tensor serialization | Save/load weights and test data separately | ⭐⭐ |
| 2 GB protobuf limit | Determines when to use external data format | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you save and load an ONNX model?** → `model.SerializeToString()` to bytes → write to file; `onnx.load()` to read back.
2. **What is the size limit for ONNX models?** → Protobuf has a 2 GB limit; use external data format for larger models.
3. **Can you serialize individual tensors?** → Yes, use `numpy_helper.from_array()` → `SerializeToString()`.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Serialization_Deep_Dive.ipynb](01_Serialization_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Serialization_Apply.ipynb](02_Serialization_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Forgetting to open files in binary mode (`'wb'` / `'rb'`)
- Hitting the 2 GB protobuf limit with large models without knowing about external data
- Confusing `onnx.load()` (from file) with `ParseFromString()` (from bytes)

---

## Next Steps
→ [Initializers_and_Attributes](../03_Initializers_and_Attributes/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
