# Computation Graph Basics

> **Interview Relevance:** HIGH — Understanding DAGs, nodes, edges, and topological order is fundamental to ML systems

---

## Visual Overview
![Linear Regression ONNX Graph](dot_linreg.png)
![Linear Regression with Initializers](dot_linreg2.png)

## Why (Motivation)
ONNX represents models as directed acyclic graphs (DAGs) of operator nodes. Understanding this representation is essential for debugging exported models, reasoning about execution order, and performing graph optimizations.

## When (Use Cases)
- When inspecting exported ONNX models in Netron
- When debugging conversion errors from framework exports
- When hand-building or modifying ONNX graphs
- When reasoning about execution order and parallelism

## How (Mechanism)
Nodes represent operator invocations; edges are implicit through name identity (SSA discipline). A forward pass evaluates nodes in topological order. Initializers provide constant weights; graph inputs provide dynamic data.

---

## Prerequisites
- [Chapter 2 Overview](../README.md)
- Basic linear algebra

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| DAG representation | Foundation of ONNX model format | ⭐⭐⭐ |
| Nodes and edges | How computation is structured | ⭐⭐⭐ |
| SSA naming | How data flow is tracked | ⭐⭐ |
| Topological order | How forward pass executes | ⭐⭐⭐ |
| Initializers vs inputs | Constants vs dynamic data | ⭐⭐⭐ |

## Key Interview Questions Answered Here
1. **What is a computation graph?** → A DAG where vertices are operations and directed edges are data dependencies.
2. **How are edges represented in ONNX?** → Implicitly through name identity (SSA); each tensor name is produced exactly once.
3. **What is the difference between graph inputs and initializers?** → Inputs are provided at runtime; initializers are constants stored in the model.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Computation_Graph_Basics_Deep_Dive.ipynb](01_Computation_Graph_Basics_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Computation_Graph_Basics_Apply.ipynb](02_Computation_Graph_Basics_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Thinking ONNX edges are explicit objects (they're implicit via name matching)
- Confusing "dynamic graph" (framework) with ONNX graphs (always static structure once exported)
- Assuming node order in the proto file is execution order (it's topological, but the runtime may reorder)

---

## Next Steps
→ [Nodes_Edges_and_Tensors](../02_Nodes_Edges_and_Tensors/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
