# OpSet Versions

> **Interview Relevance:** HIGH — Understanding opset versioning, domain scoping, and the compatibility triangle is critical for production ONNX deployment and is commonly asked in ML infrastructure interviews.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
OpSet versions determine whether a runtime can execute your model. Choosing the wrong opset leads to "unsupported operator" failures. Understanding version semantics prevents subtle cross-tool incompatibilities.

## When (Use Cases)
- Choosing target opset for model export
- Debugging "model not supported" runtime errors
- Converting models between opset versions for deployment
- Planning runtime upgrades across a fleet

## How (Mechanism)
Each `opset_import` entry in ModelProto binds a domain to a version integer. Nodes reference their domain, and the runtime looks up operator schemas at the imported version. Forward compatibility is not guaranteed — newer opsets may require newer runtimes.

---

## Prerequisites
- [01_Standard_Operators](../01_Standard_Operators/) — What operators are

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| OpSet = domain + version binding | Core versioning mechanism | ⭐⭐⭐ |
| Default domain vs custom domains | Standard vs vendor ops | ⭐⭐⭐ |
| `version_converter` usage | Practical upgrade/downgrade | ⭐⭐ |
| Compatibility triangle | Runtime/model/converter alignment | ⭐⭐⭐ |
| `since_version` per operator | When ops became available | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What is an ONNX OpSet?** → A versioned collection of operator definitions scoped to a domain; the model declares which version it targets via `opset_import`.
2. **Can you run a model built for opset 18 on a runtime that supports opset 15?** → Not reliably. Forward compatibility is not guaranteed; newer schemas may use features the older runtime lacks.
3. **How do you convert a model to a different opset?** → Use `onnx.version_converter.convert_version(model, target_version)` for the default domain.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [OpSet_Versions_Deep_Dive.ipynb](01_OpSet_Versions_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [OpSet_Versions_Apply.ipynb](02_OpSet_Versions_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Confusing `ir_version` with `opset_import` — they are independent axes
- Assuming higher opset is always better — target the lowest your runtime supports
- Forgetting custom domain ops require separate runtime registration
- Not re-running checker after version conversion

---

## Next Steps
→ [03_Custom_Operators](../03_Custom_Operators/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
