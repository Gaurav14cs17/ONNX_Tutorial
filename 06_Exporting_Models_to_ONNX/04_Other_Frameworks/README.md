# Other Frameworks

> **Interview Relevance:** MEDIUM — Knowing the broader converter ecosystem (XGBoost, LightGBM, PaddlePaddle, Hummingbird) demonstrates breadth in ML deployment.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Not every model comes from PyTorch or TensorFlow. Tree-based models (XGBoost, LightGBM), PaddlePaddle networks, and MATLAB models all have paths to ONNX. Understanding the ecosystem helps you deploy any model family through a unified runtime.

## When (Use Cases)
- Deploying XGBoost/LightGBM models via ONNX Runtime
- Migrating PaddlePaddle models to cross-platform serving
- Choosing between native tree ONNX vs tensorized compilation (Hummingbird)
- Evaluating which converter fits your tech stack

## How (Mechanism)
Each framework has a dedicated converter that understands its internal representation and maps it to ONNX operators. Tree models use `ai.onnx.ml` domain operators; DL frameworks map to standard ONNX ops. Hummingbird offers an alternative path by compiling trees to tensor operations.

---

## Prerequisites
- [03_Scikit_Learn_to_ONNX](../03_Scikit_Learn_to_ONNX/) — Classical ML conversion patterns

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| ONNXMLTools for XGBoost/LightGBM | Production tree deployment | ⭐⭐ |
| Hummingbird tensorization | GPU acceleration for trees | ⭐⭐ |
| paddle2onnx | PaddlePaddle ecosystem | ⭐ |
| Converter comparison table | Technology selection | ⭐⭐ |
| Universal validation playbook | Quality assurance | ⭐⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you deploy XGBoost models with ONNX?** → Use `onnxmltools.convert.convert_xgboost()` for native tree ONNX, or Hummingbird for tensorized GPU execution.
2. **What is Hummingbird?** → A compiler that converts tree-based models into PyTorch/ONNX tensor graphs, enabling GPU acceleration via tensor execution providers.
3. **What validation should you do after any ONNX conversion?** → Schema check (`onnx.checker`), numerical parity (source vs ORT on realistic inputs), and performance measurement on target hardware.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Other_Frameworks_Deep_Dive.ipynb](01_Other_Frameworks_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Other_Frameworks_Apply.ipynb](02_Other_Frameworks_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Assuming all XGBoost parameters are supported (version coupling matters)
- Not pinning converter versions alongside model artifacts
- Choosing Hummingbird without benchmarking (overhead may exceed tree EP)
- Forgetting that `ai.onnx.ml` domain ops need compatible runtime support

---

## Next Steps
→ Return to chapter overview for the quality checklist

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
