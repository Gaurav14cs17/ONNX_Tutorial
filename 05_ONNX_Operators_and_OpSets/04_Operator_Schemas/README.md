# Operator Schemas

> **Interview Relevance:** MEDIUM — Understanding how to read and query operator schemas shows deep ONNX expertise and is valuable for debugging export/inference issues.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Operator schemas are the "legal system" of ONNX — they define what inputs, outputs, attributes, and types are valid for each op. Reading schemas transforms debugging from guesswork into structured deduction.

## When (Use Cases)
- Debugging type mismatch or attribute errors in exported models
- Verifying which ops are available at a given opset version
- Understanding broadcasting rules and optional inputs
- Building custom tooling that programmatically validates graphs

## How (Mechanism)
Each operator has an `OpSchema` accessible via `onnx.defs.get_schema(op, opset, domain)`. The schema specifies formal parameters (with type variables), attributes (with types and defaults), and type constraints (allowed element types per type variable).

---

## Prerequisites
- [03_Custom_Operators](../03_Custom_Operators/) — Domain and op_type concepts

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `onnx.defs.get_schema` API | Primary schema query tool | ⭐⭐ |
| Type constraints (T, T1, Tind) | Type validation logic | ⭐⭐ |
| Attribute types and defaults | Debug incorrect attribute usage | ⭐⭐ |
| `since_version` tracking | Op availability history | ⭐⭐ |
| Formal parameters and variadics | Understand op arity rules | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you find what attributes a Conv op accepts?** → `onnx.defs.get_schema("Conv", opset, "")` returns the full schema including attributes, inputs, type constraints.
2. **What are type constraints in ONNX?** → They bind type variables (like `T`) to lists of allowed tensor element types, enforcing that connected inputs/outputs share compatible types.
3. **How do you check if an operator is available at a given opset?** → Call `get_schema(op, opset, domain)` — it raises if unavailable.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Operator_Schemas_Deep_Dive.ipynb](01_Operator_Schemas_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Operator_Schemas_Apply.ipynb](02_Operator_Schemas_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Assuming attribute names are stable across opset versions (they can change)
- Ignoring optional inputs (e.g., Conv bias is optional)
- Not checking type constraints when mixing precisions (float16 + float32)
- Confusing schema docs for one opset with behavior at another

---

## Next Steps
→ [06_Exporting_Models_to_ONNX](../../06_Exporting_Models_to_ONNX/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
