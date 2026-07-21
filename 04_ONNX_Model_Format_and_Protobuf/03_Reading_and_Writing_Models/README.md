# Reading and Writing ONNX Models

> **Interview Relevance:** HIGH — Load/save APIs, external data handling, and memory strategies for large models are common in ML infrastructure interviews.

---

## Visual Overview
![ONNX Graph with Initializers — Constants stored as model weights](assets/dot_linreg2.png)

## Why (Motivation)
Every ONNX workflow starts with loading or saving a model. Understanding the APIs, external data patterns, and memory-efficient strategies is critical for working with models from small prototypes to multi-gigabyte production checkpoints.

## When (Use Cases)
- Loading models for inference, optimization, or inspection
- Saving exported models with externalized weights for large checkpoints
- Building CI/CD pipelines that validate model artifacts
- Debugging "model too large" or "missing weight" errors

## How (Mechanism)
`onnx.load_model` deserializes Protobuf bytes into a `ModelProto`. For large models, weights can be stored externally via `save_as_external_data`, keeping the main `.onnx` file as a compact graph skeleton. The `load_external_data` flag controls whether weight blobs are materialized into memory.

---

## Prerequisites
- [02_ONNX_Proto_Structure](../02_ONNX_Proto_Structure/) — Understanding ModelProto and TensorProto

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `load_model` / `save_model` API | Core workflow for any ONNX task | ⭐⭐⭐ |
| External data format | Handle models >2GB | ⭐⭐⭐ |
| `load_external_data=False` pattern | Fast structural inspection | ⭐⭐ |
| Memory-efficient loading strategies | Production-scale model handling | ⭐⭐⭐ |
| Protobuf 2GB limit and workarounds | System design awareness | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you handle ONNX models larger than 2GB?** → Use `save_as_external_data=True` to store weights in companion binary files; the `.onnx` file keeps only the graph skeleton.
2. **How can you inspect a large model without loading all weights?** → `onnx.load_model(path, load_external_data=False)` loads only structure and metadata.
3. **What is the difference between `onnx.load` and `onnx.load_model`?** → They are aliases; `load_model` is the modern explicit name.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Reading_and_Writing_Models_Deep_Dive.ipynb](01_Reading_and_Writing_Models_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Reading_and_Writing_Models_Apply.ipynb](02_Reading_and_Writing_Models_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Moving external weight files without updating paths → "missing initializer" errors
- Forgetting that `load_external_data=True` requires the `.onnx` file path (not a stream)
- Confusing `size_threshold` (bytes) — set to 0 to externalize all tensors
- Not treating `.onnx` + sidecar files as a bundle during deployment

---

## Next Steps
→ [04_Model_Metadata_and_Versioning](../04_Model_Metadata_and_Versioning/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
