# Pruning and Sparsity

> **Interview Relevance:** MEDIUM — Pruning is less commonly asked than quantization but important for edge/mobile roles. Key distinction: unstructured vs structured pruning and why sparsity alone doesn't guarantee speedups.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for structured vs unstructured pruning diagrams.

## Why (Motivation)
Pruning removes parameters or structural components to achieve smaller storage, fewer FLOPs, or better mapping to sparse hardware. However, the gap between "sparse weights" and "faster inference" is often wider than expected — understanding when pruning actually helps is critical.

## When (Use Cases)
- Reducing model size for bandwidth-constrained deployment
- Preparing models for hardware with sparse compute units
- Combining with quantization for maximum compression
- Structured pruning to reduce actual compute on standard hardware

## How (Mechanism)
**Unstructured pruning** zeros individual weights (flexible but needs sparse kernels). **Structured pruning** removes entire neurons/channels/filters, shrinking dimensions and reducing dense BLAS compute. In ONNX, pruned models typically use the same operators with sparser initializer tensors.

---

## Prerequisites
- [02_Quantization_Techniques](../02_Quantization_Techniques/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Unstructured vs structured pruning | Fundamentally different impact on speed | ⭐⭐⭐ |
| Sparsity in ONNX representation | How pruned models are stored | ⭐⭐ |
| Sparse tensor support | ONNX capabilities and limitations | ⭐ |
| Size vs speed impact | Why sparse ≠ fast without kernel support | ⭐⭐⭐ |
| Framework-side pruning + re-export | Most common high-accuracy path | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What's the difference between structured and unstructured pruning?** → Unstructured zeros individual weights (scattered); structured removes whole channels/neurons, shrinking tensor dimensions.
2. **Does pruning automatically make inference faster?** → No. Unstructured sparsity needs sparse kernels; only structured pruning reliably reduces compute on standard dense BLAS.
3. **How is sparsity represented in ONNX?** → Typically as dense tensors with many zeros. ONNX defines sparse tensor encodings but not all runtimes accelerate them.
4. **How do you prune an ONNX model?** → Usually prune in the training framework (PyTorch pruning APIs), fine-tune, then re-export to ONNX.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Pruning_and_Sparsity_Deep_Dive.ipynb](01_Pruning_and_Sparsity_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Pruning_and_Sparsity_Apply.ipynb](02_Pruning_and_Sparsity_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Assuming unstructured sparsity speeds up inference on standard hardware — it usually doesn't without sparse kernels.
- Forgetting to fine-tune after pruning — accuracy can drop significantly.
- Expecting ONNX sparse tensor types to automatically yield faster inference — runtimes may densify internally.
- Only measuring file size reduction without checking wall-clock latency.

---

## Next Steps
→ [04_Benchmarking](../04_Benchmarking/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
