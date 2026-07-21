# Installation and Setup

> **Interview Relevance:** LOW — Setup knowledge is practical but rarely asked directly in interviews

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
A properly configured environment is the foundation for all ONNX work. Mismatched versions between `onnx`, `onnxruntime`, CUDA drivers, and framework converters are the #1 source of deployment failures.

## When (Use Cases)
- Setting up a new development environment for ONNX work
- Debugging import errors and provider failures
- Preparing reproducible environments for CI/CD pipelines

## How (Mechanism)
Install `onnx` (model creation/validation) and `onnxruntime` (inference) via pip. Verify with a minimal build-and-run test. For GPU workflows, ensure CUDA toolkit version matches the `onnxruntime-gpu` wheel.

---

## Prerequisites
- [ONNX Ecosystem Overview](../03_ONNX_Ecosystem_Overview/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Core package installation | Foundation for all ONNX work | ⭐ |
| CPU vs GPU runtime | Choosing the right wheel | ⭐⭐ |
| Version compatibility | Avoiding deployment failures | ⭐⭐ |
| Verification testing | Confirming correct setup | ⭐ |

## Key Interview Questions Answered Here
1. **What packages do you need for ONNX?** → `onnx` (creation/validation), `onnxruntime` (inference), `numpy` (data handling).
2. **What is the most common GPU setup issue?** → Mismatched CUDA driver, CUDA toolkit, and onnxruntime-gpu wheel versions.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Installation_and_Setup_Deep_Dive.ipynb](01_Installation_and_Setup_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Installation_and_Setup_Apply.ipynb](02_Installation_and_Setup_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Installing both `onnxruntime` and `onnxruntime-gpu` (pick one)
- Not matching CUDA driver version with the GPU wheel
- Mixing Python environments (system vs venv vs conda)

---

## Next Steps
→ Return to [Chapter Overview](../README.md)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
