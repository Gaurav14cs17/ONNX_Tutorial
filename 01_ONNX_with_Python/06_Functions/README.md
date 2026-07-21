# Functions

> **Interview Relevance:** MEDIUM — Custom functions demonstrate advanced ONNX knowledge

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Functions let you define reusable operator combinations — like creating your own "macro operator" from existing ONNX primitives. This makes graphs shorter, more readable, and potentially faster as runtimes may optimize known function patterns.

## When (Use Cases)
- Creating reusable operator patterns (e.g., a custom activation)
- Making graphs more readable by encapsulating common subgraph patterns
- Extending ONNX with domain-specific operators without writing C++ kernels

## How (Mechanism)
Use `make_function()` to define a function with a custom domain, name, inputs, outputs, and internal nodes. Reference it in the graph with `make_node('FuncName', ..., domain='custom')`. Register it in the model with `functions=[...]`.

---

## Prerequisites
- [Subgraphs Tests Loops](../05_Subgraphs_Tests_Loops/)

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| `make_function` | Create reusable operator combos | ⭐⭐ |
| Custom domains | Namespace for custom operators | ⭐⭐ |
| Function attributes | Pass fixed params to functions | ⭐ |
| Functions vs subgraphs | Different mechanisms for different needs | ⭐⭐ |

## Key Interview Questions Answered Here
1. **How do you create a custom operator in ONNX without C++?** → Use `make_function` to compose existing operators under a custom domain.
2. **What is the difference between functions and subgraphs?** → Functions are reusable, model-level definitions; subgraphs are control-flow bodies tied to specific nodes.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Functions_Deep_Dive.ipynb](01_Functions_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Functions_Apply.ipynb](02_Functions_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Forgetting to register the function in the model's `functions` list
- Not including the custom domain in `opset_imports`
- Confusing function attributes with node inputs

---

## Next Steps
→ [Parsing_and_Checker](../07_Parsing_and_Checker/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
