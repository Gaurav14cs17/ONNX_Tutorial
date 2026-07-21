# 05 — ONNX Operators and OpSets

This chapter bridges **what** an ONNX graph *is* and **how** it is *executed and evolved*. ONNX represents computation as a DAG of **operators**, each identified by a name, domain, and versioned through **operator sets (OpSets)**.

---

## Learning Order

| # | Topic | Focus | Time |
|:-:|-------|-------|:----:|
| 1 | [01_Standard_Operators](./01_Standard_Operators/) | Operator categories, common ops, data flow patterns | ~35 min |
| 2 | [02_OpSet_Versions](./02_OpSet_Versions/) | Version numbering, domains, upgrade/downgrade workflows | ~35 min |
| 3 | [03_Custom_Operators](./03_Custom_Operators/) | When/how to define custom ops, registration, ORT kernels | ~35 min |
| 4 | [04_Operator_Schemas](./04_Operator_Schemas/) | Reading schemas, type constraints, programmatic queries | ~35 min |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Explain** what an ONNX operator is and how it relates to NodeProto
2. **Navigate** major categories of standard operators
3. **Define** an OpSet and interpret opset imports
4. **Perform** version conversions and choose target opsets
5. **Motivate** custom operators and understand registration workflow
6. **Query** operator schemas programmatically

---

## Prerequisites

- Basic familiarity with ONNX graphs (GraphProto, tensors vs initializers)
- Python 3.9+ with `pip install onnx`

## See Also

- [diagrams/operator_categories.md](./diagrams/operator_categories.md) — Mermaid map of operator categories
- Official: [ONNX Operators](https://onnx.ai/onnx/operators/)

---

*Next: [06_Exporting_Models_to_ONNX](../06_Exporting_Models_to_ONNX/)*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
