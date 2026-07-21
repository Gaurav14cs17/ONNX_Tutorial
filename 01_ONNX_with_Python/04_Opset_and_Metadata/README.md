# Opset and Metadata

> **Interview Relevance:** HIGH — OpSet versioning is critical for model compatibility across runtimes

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
ONNX operators evolve across versions — their schemas, inputs, and behaviors can change. The OpSet version acts as a "language version" ensuring operators are interpreted consistently. Model metadata provides traceability for production deployments.

## When (Use Cases)
- Ensuring models are compatible with target runtimes
- Debugging operator mismatch errors between exporter and runtime
- Tagging models with producer info, authorship, and versioning for ML registries

## How (Mechanism)
Each model declares `opset_import` entries mapping domains to version numbers. At runtime, the most recent operator spec ≤ the declared opset version is used. Metadata fields like `producer_name`, `doc_string`, and `metadata_props` are set directly on the `ModelProto`.

---

## Prerequisites
- [Initializers and Attributes](../03_Initializers_and_Attributes/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| OpSet versioning | Determines operator behavior | ⭐⭐⭐ |
| IR version vs OpSet | Two independent version axes | ⭐⭐⭐ |
| Model metadata | Traceability in production | ⭐⭐ |
| Multiple domains | Using standard + ML + custom ops | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What is an OpSet in ONNX?** → A versioned set of operator definitions; determines which spec version of each op is used.
2. **How does ONNX resolve which operator version to use?** → Uses the most recent version ≤ the declared opset version.
3. **What is the difference between IR version and OpSet version?** → IR version controls the container format (graph structure); OpSet controls individual operator semantics.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Opset_and_Metadata_Deep_Dive.ipynb](01_Opset_and_Metadata_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Opset_and_Metadata_Apply.ipynb](02_Opset_and_Metadata_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Confusing IR version (container format) with OpSet version (operator semantics)
- Using an opset that doesn't support the operators you need
- Forgetting that operator behavior can change between opset versions (e.g., Reshape)

---

## Next Steps
→ [Subgraphs_Tests_Loops](../05_Subgraphs_Tests_Loops/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
