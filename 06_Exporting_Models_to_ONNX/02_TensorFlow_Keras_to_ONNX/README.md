# TensorFlow / Keras to ONNX

> **Interview Relevance:** HIGH — Converting TF models via `tf2onnx`, handling NHWC/NCHW layout differences, and SavedModel workflows are common in production ML pipelines.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Many organizations train with TensorFlow/Keras but deploy with ONNX Runtime for better cross-platform performance. `tf2onnx` bridges this gap by converting SavedModel artifacts to portable ONNX graphs.

## When (Use Cases)
- Deploying Keras models via ONNX Runtime for speed
- Migrating from TF Serving to a unified ONNX-based serving stack
- Converting TFLite models for server-side inference
- Cross-framework model sharing

## How (Mechanism)
`tf2onnx` reads TensorFlow graph structures (SavedModel, GraphDef, or Keras) and rewrites TF ops into ONNX primitives. It handles NHWC→NCHW layout translation by inserting Transpose nodes where needed.

---

## Prerequisites
- [01_PyTorch_to_ONNX](../01_PyTorch_to_ONNX/) — General export concepts

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `tf2onnx` conversion paths | Primary TF→ONNX tool | ⭐⭐⭐ |
| SavedModel as intermediate | Recommended export format | ⭐⭐⭐ |
| NHWC↔NCHW handling | Layout translation awareness | ⭐⭐⭐ |
| Signature selection | Multi-signature models | ⭐⭐ |
| TFLite → ONNX path | Mobile model migration | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you convert a Keras model to ONNX?** → Save as SavedModel, then use `tf2onnx.convert.from_keras()` or CLI `python -m tf2onnx.convert --saved-model`.
2. **Why do TF→ONNX models have extra Transpose nodes?** → TensorFlow uses NHWC layout; ONNX Conv typically expects NCHW. tf2onnx inserts layout conversions.
3. **What format should TF models be in before ONNX conversion?** → SavedModel is recommended; it provides clear serving signatures and variable serialization.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [TensorFlow_Keras_to_ONNX_Deep_Dive.ipynb](01_TensorFlow_Keras_to_ONNX_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [TensorFlow_Keras_to_ONNX_Apply.ipynb](02_TensorFlow_Keras_to_ONNX_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Exporting training-mode graphs (includes optimizer state, iterators)
- Not specifying signatures for multi-signature SavedModels
- Expecting NHWC tensors in ONNX output (they may be transposed to NCHW)
- Forgetting to validate parity — layout changes can shift numerical results

---

## Next Steps
→ [03_Scikit_Learn_to_ONNX](../03_Scikit_Learn_to_ONNX/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
