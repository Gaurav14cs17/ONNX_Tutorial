# Shape Inference

> **Interview Relevance:** MEDIUM — Shape inference demonstrates understanding of ONNX's type system and is essential for debugging shape mismatches. Common in ML platform roles.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*
See `assets/` for shape propagation flow diagrams.

## Why (Motivation)
Shape inference fills in tensor shape information for intermediate values by propagating known input shapes through operator shape rules. It catches structural mismatches early, makes inspection tools more informative, and helps runtimes plan memory allocation.

## When (Use Cases)
- Validating exported models before deployment
- Debugging shape mismatches (Concat axis, Reshape incompatibility)
- Making Netron visualizations more informative
- Preparing models for runtimes that use inferred shapes for memory planning

## How (Mechanism)
`onnx.shape_inference.infer_shapes()` walks the graph topologically, applying each operator's shape inference rules. Static dimensions are resolved to integers; dynamic dimensions remain as symbolic parameters (`dim_param`). Partial inference is normal when input ranks are unknown.

---

## Prerequisites
- [01_Inspecting_Models](../01_Inspecting_Models/)
- [02_Modifying_Graphs](../02_Modifying_Graphs/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `onnx.shape_inference` API | Core shape analysis tool | ⭐⭐⭐ |
| Static vs symbolic dimensions | Understanding dynamic models | ⭐⭐⭐ |
| Partial inference limitations | Why some shapes remain unknown | ⭐⭐ |
| Interactions with `onnx.checker` | Combined validation workflow | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What does shape inference do in ONNX?** → Propagates known input shapes through operator rules to fill in intermediate tensor shapes; catches mismatches early.
2. **What's the difference between `dim_value` and `dim_param`?** → `dim_value` is a static integer; `dim_param` is a symbolic name (e.g., "batch") representing a dynamic dimension.
3. **Why might shape inference be incomplete?** → Unknown input ranks, malformed attributes, or ops with data-dependent output shapes can block propagation.
4. **Should you use shape inference or checker first?** → Use `infer_shapes` first to fill in shape info, then `checker.check_model` for full validation.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Shape_Inference_Deep_Dive.ipynb](01_Shape_Inference_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Shape_Inference_Apply.ipynb](02_Shape_Inference_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Expecting shape inference to resolve symbolic dimensions — they remain symbolic by design.
- Treating shape inference as a substitute for `check_model` — they serve different purposes.
- Not handling partial inference — some intermediate shapes may remain unknown.
- Forgetting that shape inference is version-sensitive — API kwargs differ across ONNX releases.

---

## Next Steps
→ [04_Model_Validation](../04_Model_Validation/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
