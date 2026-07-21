# Model Metadata and Versioning

> **Interview Relevance:** HIGH — Understanding IR version vs opset version, model identity fields, and governance metadata is frequently tested in ML platform and infrastructure interviews.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Metadata transforms an anonymous model file into an auditable, reproducible artifact. Proper versioning prevents subtle compatibility bugs when deploying across different runtime versions.

## When (Use Cases)
- Setting up model registries and CI/CD validation
- Debugging "unsupported opset" runtime errors
- Auditing model provenance for compliance
- Choosing target opset for cross-platform deployment

## How (Mechanism)
`ModelProto` carries `ir_version` (container schema), `opset_import` (operator definitions), producer information, and arbitrary `metadata_props`. These two versioning axes are independent: IR version governs proto structure, while opset version governs operator semantics.

---

## Prerequisites
- [03_Reading_and_Writing_Models](../03_Reading_and_Writing_Models/) — How to load/save models

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| IR version vs opset version | Two independent versioning axes | ⭐⭐⭐ |
| `opset_import` semantics | Which ops a model may invoke | ⭐⭐⭐ |
| `metadata_props` for governance | Auditable artifacts in production | ⭐⭐ |
| Producer and domain fields | Model identity and provenance | ⭐⭐ |
| Version history table | Choose compatible targets | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What is the difference between `ir_version` and `opset_import`?** → `ir_version` defines the Protobuf container rules; `opset_import` defines which operator definitions (semantics) apply to nodes.
2. **How do you ensure a model works on a specific runtime?** → Match `opset_import` version to the runtime's supported opset; run `onnx.checker.check_model`.
3. **What metadata should a production ONNX model carry?** → Producer info, domain + model_version, commit hash, dataset fingerprint, quantization recipe in `metadata_props`.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Model_Metadata_and_Versioning_Deep_Dive.ipynb](01_Model_Metadata_and_Versioning_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Model_Metadata_and_Versioning_Apply.ipynb](02_Model_Metadata_and_Versioning_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Confusing `ir_version` with `opset_import` version — they are independent
- Setting opset too high for target runtime → "unsupported operator" errors
- Forgetting to bump `model_version` when retraining with same architecture
- Not recording exporter version in `producer_version` → irreproducible artifacts

---

## Next Steps
→ [05_ONNX_Operators_and_OpSets](../../05_ONNX_Operators_and_OpSets/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
