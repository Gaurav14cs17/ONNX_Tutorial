# Parsing and Checker

> **Interview Relevance:** HIGH — Model validation and shape inference are critical production skills

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
The ONNX parser provides a concise text format for defining models (6 lines vs 12+). The checker validates structural correctness, catching type mismatches and wiring errors. Shape inference propagates shape information, enabling runtime optimizations like memory pre-allocation and operator fusion.

## When (Use Cases)
- Quickly prototyping ONNX models without verbose API calls
- Validating exported models before deployment
- Running shape inference to enable runtime optimizations
- Debugging conversion errors by inspecting inferred shapes

## How (Mechanism)
`onnx.parser.parse_model()` converts a text representation to a `ModelProto`. `onnx.checker.check_model()` validates against the OpSet spec. `onnx.shape_inference.infer_shapes()` walks the graph and computes output shapes from input shapes using operator schemas.

---

## Prerequisites
- [Functions](../06_Functions/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `onnx.parser` | Concise model definition | ⭐⭐ |
| `onnx.checker` | Structural validation | ⭐⭐⭐ |
| Shape inference | Propagate shapes through graph | ⭐⭐⭐ |
| Symbolic dimensions | Dynamic batch sizes | ⭐⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you validate an ONNX model?** → Use `onnx.checker.check_model()` to validate types, operator specs, and graph structure.
2. **What is shape inference in ONNX?** → A schema-driven static analysis that computes output shapes from input shapes without running inference.
3. **What are symbolic dimensions?** → String names (e.g., "batch") in dim_param that represent dynamic dimensions known only at runtime.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Parsing_and_Checker_Deep_Dive.ipynb](01_Parsing_and_Checker_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Parsing_and_Checker_Apply.ipynb](02_Parsing_and_Checker_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Assuming shape inference can resolve data-dependent shapes (it can't)
- Not running `check_model()` after every graph modification
- Confusing the parser text format with Python syntax

---

## Next Steps
→ [Evaluation_and_Runtime](../08_Evaluation_and_Runtime/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
