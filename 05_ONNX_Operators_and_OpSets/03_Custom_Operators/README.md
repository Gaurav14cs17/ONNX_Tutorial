# Custom Operators

> **Interview Relevance:** MEDIUM — Understanding when and why custom ops are needed, their trade-offs, and the registration workflow shows advanced ONNX knowledge.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Custom operators let you express computations that the ONNX standard doesn't (yet) cover — fused kernels, proprietary algorithms, or domain-specific primitives. They bridge the gap between "what ONNX defines today" and "what your model needs."

## When (Use Cases)
- Fused attention or normalization kernels for performance
- Proprietary algorithms that cannot be decomposed into standard ops
- Vendor-specific hardware intrinsics
- Research ops not yet standardized

## How (Mechanism)
Custom ops use a private domain string (e.g., `com.myorg.fused`) with versioned opset imports. The model file carries the schema reference; the runtime must have a registered kernel implementation. Without the kernel, the model is unrunnable.

---

## Prerequisites
- [02_OpSet_Versions](../02_OpSet_Versions/) — How domains and versions work

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Custom domain + op_type pattern | How to reference non-standard ops | ⭐⭐ |
| Portability vs expressiveness trade-off | Architecture decisions | ⭐⭐⭐ |
| ORT kernel registration workflow | Production deployment | ⭐⭐ |
| Schema definition requirements | Interoperability discipline | ⭐⭐ |
| When NOT to use custom ops | Avoiding unnecessary complexity | ⭐⭐ |

## Key Interview Questions Answered Here
1. **When would you use a custom ONNX operator?** → When no standard op combination achieves required performance, numerical parity, or expressiveness, AND you control both the export and deployment runtimes.
2. **What are the trade-offs of custom ops?** → You gain expressiveness/performance but lose portability — the model only runs where your kernel is registered.
3. **How does a runtime know how to execute a custom op?** → The kernel must be registered with the runtime (e.g., ORT) before session creation, mapping (domain, op_type) to an implementation.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Custom_Operators_Deep_Dive.ipynb](01_Custom_Operators_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Custom_Operators_Apply.ipynb](02_Custom_Operators_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Forgetting to add custom domain to `opset_import` in ModelProto
- Not versioning your custom domain → silent schema breaks
- Assuming custom ops are portable across different ORT builds
- Shipping model without the corresponding runtime extension

---

## Next Steps
→ [04_Operator_Schemas](../04_Operator_Schemas/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
