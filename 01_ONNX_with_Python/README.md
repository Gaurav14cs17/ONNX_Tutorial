# 01 — ONNX with Python: Hands-On Quick Start

> **Source:** This chapter is based on the official ONNX documentation at
> [onnx.ai/onnx/intro/python.html](https://onnx.ai/onnx/intro/python.html)
> with enhanced explanations, diagrams, and runnable examples.

---

## Chapter Overview

This is the **practical quick-start chapter** — jump straight into building ONNX graphs
with Python before diving into theory. Every concept is shown through working code and
official ONNX documentation diagrams.

---

## Learning Order

| # | Section | Key Concepts | Time |
|---|---------|-------------|:----:|
| 1 | [Linear Regression Example](01_Linear_Regression_Example/) | `make_node`, `make_graph`, `make_model`, first ONNX graph | ~35 min |
| 2 | [Serialization](02_Serialization/) | `SerializeToString`, `load`, saving `.onnx` files | ~35 min |
| 3 | [Initializers & Attributes](03_Initializers_and_Attributes/) | Constants, `numpy_helper.from_array`, operator attributes | ~35 min |
| 4 | [Opset & Metadata](04_Opset_and_Metadata/) | OpSet versions, IR version, model metadata | ~35 min |
| 5 | [Subgraphs: Tests & Loops](05_Subgraphs_Tests_Loops/) | `If`, `Scan`, `Loop` operators with subgraphs | ~35 min |
| 6 | [Functions](06_Functions/) | Custom reusable functions, `make_function` | ~35 min |
| 7 | [Parsing & Checker](07_Parsing_and_Checker/) | `onnx.parser`, `onnx.checker`, shape inference | ~35 min |
| 8 | [Evaluation & Runtime](08_Evaluation_and_Runtime/) | `ReferenceEvaluator`, custom operators, benchmarking | ~35 min |

Each section contains:
- **`{Topic}_Deep_Dive.ipynb`** — Theory and internals (~20 min)
- **`{Topic}_Apply.ipynb`** — Hands-on exercises (~15 min)
- **`README.md`** — Enhanced reference with interview questions

---

## Learning Objectives

After completing this chapter, you will be able to:

1. Build ONNX graphs from scratch using the Python API
2. Serialize and deserialize ONNX models
3. Use initializers and operator attributes
4. Understand and set OpSet versions and metadata
5. Implement control flow with If, Scan, and Loop
6. Create reusable ONNX functions
7. Validate models and perform shape inference
8. Evaluate ONNX models using the Reference Evaluator

## Prerequisites

- Python 3.8+
- `pip install onnx onnxruntime numpy`

---

## Official ONNX Diagrams

| Diagram | Description | Location |
|---------|-------------|----------|
| ![Linear Regression](01_Linear_Regression_Example/dot_linreg.png) | Simple linear regression graph `Y = XA + B` | Section 1 |
| ![Linear Regression with Initializers](03_Initializers_and_Attributes/dot_linreg2.png) | Linear regression with constants as initializers | Section 3 |
| ![Attributes (Transpose)](03_Initializers_and_Attributes/dot_att.png) | Graph with operator attributes (Transpose) | Section 3 |
| ![If Operator](05_Subgraphs_Tests_Loops/dot_if_py.png) | Conditional subgraph (If operator) | Section 5 |
| ![Scan Operator](05_Subgraphs_Tests_Loops/dot_scan_py.png) | KNN regression using Scan operator | Section 5 |

---

**[Next: Section 1 — Linear Regression Example →](01_Linear_Regression_Example/)**

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
