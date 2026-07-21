# Model Validation

> **Interview Relevance:** HIGH — Validation is the final gate before deployment. Interviewers expect you to explain `onnx.checker`, common errors, and how to build conformance tests in CI.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for validation flowchart diagrams.

## Why (Motivation)
Validation answers: "Is this ONNX file well-formed and internally consistent?" Without it, structural bugs pass silently to production where they cause runtime failures. `onnx.checker` catches schema violations, type mismatches, and malformed attributes. Conformance tests add organizational policies on top.

## When (Use Cases)
- CI/CD gate before deploying ONNX models
- After any graph modification (merging, node insertion, op replacement)
- Debugging export failures or runtime errors
- Enforcing organizational standards (opset bounds, forbidden ops, size budgets)

## How (Mechanism)
`onnx.checker.check_model()` validates protobuf constraints, IR invariants, and operator schema conformance for declared opsets. Combine with `shape_inference` for completeness. Add custom Python policy checks for organizational requirements (opset bounds, forbidden ops, quantization contracts, size budgets).

---

## Prerequisites
- [01_Inspecting_Models](../01_Inspecting_Models/)
- [02_Modifying_Graphs](../02_Modifying_Graphs/)
- [03_Shape_Inference](../03_Shape_Inference/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `onnx.checker.check_model` | Standard ONNX validation | ⭐⭐⭐ |
| Common validation errors | Debug export/manipulation issues | ⭐⭐⭐ |
| Conformance testing | Organizational CI gates | ⭐⭐ |
| Policy checks (forbidden ops, opset bounds) | Production deployment standards | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you validate an ONNX model?** → Use `onnx.checker.check_model()` for schema conformance; combine with `shape_inference` for shape consistency; add custom policy checks for organizational requirements.
2. **What does `check_model` actually check?** → Protobuf field validity, graph wiring (inputs/outputs), node schema conformance for declared opsets, type constraints, and attribute validity.
3. **When should you validate?** → After every graph modification, as a CI gate before deployment, and after model merging (which can create name collisions).
4. **How do you implement custom conformance tests?** → Write Python functions that walk `graph.node` checking for forbidden ops, verify opset bounds, enforce size budgets, and validate quantization patterns.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Model_Validation_Deep_Dive.ipynb](01_Model_Validation_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Model_Validation_Apply.ipynb](02_Model_Validation_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Skipping validation after graph modifications — structural bugs pass silently to runtime.
- Not validating after model merging — `merge_models` can create name collisions.
- Treating `check_model` as a full correctness guarantee — it validates structure, not numerical correctness.
- Forgetting `load_external_data=True` for models with external data files — causes "missing data" errors.

---

## Next Steps
→ [Chapter_09_Deployment](../../Chapter_09_Deployment/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
