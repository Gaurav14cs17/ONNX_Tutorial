# 03 — ONNX Architecture and Internals

This chapter peels back ONNX's outer packaging and studies how models are represented as **structured data**: directed computation graphs, typed tensors, and a versioned **Intermediate Representation (IR)** built from Protocol Buffers.

---

## Learning Order

| # | Section | Focus | Time |
|---|---------|-------|:----:|
| 1 | [Computation Graph Basics](01_Computation_Graph_Basics/) | DAGs, nodes and edges, static vs dynamic, building graphs with `onnx.helper` | ~35 min |
| 2 | [Nodes, Edges, and Tensors](02_Nodes_Edges_and_Tensors/) | `NodeProto` anatomy, data flow, tensor types, programmatic inspection | ~35 min |
| 3 | [ONNX IR Specification](03_ONNX_IR_Specification/) | IR versions, proto hierarchy, attributes, sub-graphs, functions, compatibility | ~35 min |
| 4 | [Type System and Shapes](04_Type_System_and_Shapes/) | Tensors, sequences, maps, optionals, shape inference, symbolic dimensions | ~35 min |

Each section contains:
- **`{Topic}_Deep_Dive.ipynb`** — Theory and internals (~20 min)
- **`{Topic}_Apply.ipynb`** — Hands-on exercises (~15 min)
- **`README.md`** — Enhanced reference with interview questions

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Define** a computation graph as a DAG with named value edges
2. **Map** neural network layers to ONNX operator nodes
3. **Interpret** an ONNX node: `op_type`, inputs, outputs, and attributes
4. **Navigate** the IR hierarchy: `ModelProto` → `GraphProto` → `NodeProto`
5. **Enumerate** ONNX tensor element types (FLOAT, FLOAT16, INT8, BFLOAT16, etc.)
6. **Explain** ONNX's type system: Sequence, Map, Optional, and shape inference
7. **Inspect** `.onnx` files in Python: list nodes, read types, run shape inference

---

## Prerequisites

### Knowledge
- Completion of **Chapter 2** (environment setup and ONNX motivation)
- Basic linear algebra: matrix multiply, elementwise ops
- Python: comfortable with small scripts and printing object fields

### Software
- Python 3.8+ with `onnx` installed
- Optional: Netron for visual graph inspection

### How to Use This Chapter
Read sections **01**→**04** in order. Keep a Python REPL open and run the snippets.

---

## Files and Assets

| Path | Description |
|------|-------------|
| `diagrams/architecture_overview.md` | Mermaid diagram: ModelProto → GraphProto → NodeProto |
| Section `01` | Standalone graph construction example |
| Sections `02`–`04` | Inspection and inference helpers |

---

*Next: [01 — Computation Graph Basics →](01_Computation_Graph_Basics/)*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
