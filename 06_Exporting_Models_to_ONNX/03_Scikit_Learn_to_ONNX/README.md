# Scikit-Learn to ONNX

> **Interview Relevance:** MEDIUM — Converting classical ML pipelines to ONNX for unified deployment shows breadth in ML infrastructure knowledge.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Many production systems use scikit-learn for tabular data (classifiers, regressors, preprocessing pipelines). Converting to ONNX enables unified serving infrastructure and often faster inference via ONNX Runtime's optimized tree ensemble kernels.

## When (Use Cases)
- Deploying sklearn classifiers/regressors in production
- Unifying DL and classical ML under one serving framework
- Embedding preprocessing (scaling, encoding) in the model file
- Migrating from pickle-based serving to ONNX

## How (Mechanism)
`skl2onnx` inspects sklearn estimator topology and reimplements each step as ONNX operators (from the `ai.onnx.ml` domain for trees, standard domain for linear algebra). The resulting graph can be served without Python/sklearn at inference time.

---

## Prerequisites
- [01_PyTorch_to_ONNX](../01_PyTorch_to_ONNX/) — General export concepts

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `convert_sklearn` API | Primary sklearn→ONNX tool | ⭐⭐ |
| `initial_types` declaration | Input shape/dtype contract | ⭐⭐ |
| Pipeline conversion | End-to-end deployment | ⭐⭐⭐ |
| `ai.onnx.ml` domain ops | Tree ensemble representation | ⭐⭐ |
| Probability parity validation | Correctness assurance | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you deploy a sklearn pipeline with ONNX?** → Use `skl2onnx.convert_sklearn(pipeline, initial_types=..., target_opset=17)` — it chains scaler + model into one ONNX graph.
2. **Does ONNX execution use Python sklearn?** → No. `skl2onnx` creates a static ONNX graph that reimplements sklearn behavior using ONNX operators.
3. **What are `initial_types`?** → They declare input tensor names, shapes, and dtypes (e.g., `FloatTensorType([None, n_features])`) to guide the ONNX graph builder.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Scikit_Learn_to_ONNX_Deep_Dive.ipynb](01_Scikit_Learn_to_ONNX_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Scikit_Learn_to_ONNX_Apply.ipynb](02_Scikit_Learn_to_ONNX_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Using float64 training data but declaring FloatTensorType (float32) — dtype mismatch
- Forgetting ColumnTransformer column order must match training
- Custom estimators require custom converters (not auto-supported)
- Not validating both `predict` and `predict_proba` outputs

---

## Next Steps
→ [04_Other_Frameworks](../04_Other_Frameworks/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
