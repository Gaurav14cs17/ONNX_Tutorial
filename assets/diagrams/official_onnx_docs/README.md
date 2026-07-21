# Official ONNX Documentation Diagrams

These diagrams are downloaded from the official ONNX documentation at
[onnx.ai/onnx/intro/python.html](https://onnx.ai/onnx/intro/python.html).

## Diagram Index

| # | File | Description | Used In |
|---|------|-------------|---------|
| 1 | ![dot_linreg.png](dot_linreg.png) | **Linear Regression Graph** — Simple `Y = XA + B` with MatMul and Add nodes. Shows inputs X, A, B flowing through the graph to produce output Y. | Chapter 0.1, Chapter 2.1 |
| 2 | ![dot_linreg2.png](dot_linreg2.png) | **Linear Regression with Initializers** — Same regression but with A and C as constants (initializers) stored in the model instead of inputs. Shows the difference between inputs and initializers. | Chapter 0.3, Chapter 2.1 |
| 3 | ![dot_att.png](dot_att.png) | **Graph with Attributes (Transpose)** — Shows `Y = X·A' + B` where A is transposed. The Transpose node has a `perm=[1,0]` attribute, illustrating the difference between inputs (dynamic data) and attributes (fixed parameters). | Chapter 0.3, Chapter 4.1 |
| 4 | ![dot_if_py.png](dot_if_py.png) | **If Operator (Conditional Subgraph)** — Computes sum of matrix, then branches: returns 1 if sum > 0, else -1. Shows `then_branch` and `else_branch` subgraphs embedded as attributes of the If node. | Chapter 0.5, Chapter 2.3 |
| 5 | ![dot_scan_py.png](dot_scan_py.png) | **Scan Operator (KNN Regression)** — Complete K-Nearest Neighbors regression model using the Scan operator for pairwise distance computation. Shows the main graph and the nested subgraph. | Chapter 0.5, Chapter 8.1 |
| 6 | ![scanop.png](scanop.png) | **Scan Iteration Mechanism** — Shows how the Scan operator iterates: green = first iteration, blue = second iteration. The state is passed forward while scan outputs are collected row-by-row. | Chapter 0.5 |

## Source

All images are from: `https://onnx.ai/onnx/_images/`

Licensed under the Apache 2.0 License as part of the ONNX project.
