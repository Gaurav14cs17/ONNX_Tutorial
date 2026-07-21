# Standard Operators

> **Interview Relevance:** HIGH — Knowing common ONNX operators, their categories, and how data flows through them is essential for debugging exports and optimizing inference graphs.

---

## Visual Overview
![ONNX Operator Attributes — Transpose with perm=[1,0]](assets/dot_att.png)

## Why (Motivation)
ONNX represents computation as a DAG of operators. Understanding the standard operator vocabulary lets you read any exported model, debug shape errors, and reason about runtime optimization (fusion, quantization).

## When (Use Cases)
- Reading and understanding exported ONNX graphs
- Debugging "unsupported op" errors during export or inference
- Optimizing models by understanding which ops can be fused
- Answering interview questions about ONNX computation model

## How (Mechanism)
Each operator has a canonical name, typed inputs/outputs, and static attributes. Operators are grouped into categories (math, NN, tensor manipulation). A node in the graph references an op_type and wires data via string names.

---

## Prerequisites
- [04_ONNX_Model_Format_and_Protobuf](../../04_ONNX_Model_Format_and_Protobuf/) — ModelProto and GraphProto basics

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Operator categories (math, NN, tensor) | Navigate the ONNX op vocabulary | ⭐⭐⭐ |
| Conv, MatMul, Relu, Softmax semantics | Core NN building blocks | ⭐⭐⭐ |
| Attributes vs inputs distinction | Static config vs dynamic data | ⭐⭐⭐ |
| Broadcasting rules | Shape compatibility | ⭐⭐ |
| Operator fusion patterns | Runtime optimization awareness | ⭐⭐ |

## Key Interview Questions Answered Here
1. **What are the main categories of ONNX operators?** → Elementwise math, linear algebra, reduction, tensor reorganization, NN layers (conv, activation, normalization), and control flow.
2. **How does Conv work in ONNX?** → Takes input X, weight W, optional bias B; attributes control kernel_shape, strides, pads, dilations, group; outputs spatial feature map.
3. **What is the difference between attributes and inputs?** → Attributes are static (fixed at export), inputs are dynamic (provided at runtime).

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Standard_Operators_Deep_Dive.ipynb](01_Standard_Operators_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Standard_Operators_Apply.ipynb](02_Standard_Operators_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Confusing NCHW (ONNX default for Conv) with NHWC (TensorFlow default)
- Forgetting that `Gemm` and `MatMul + Add` are semantically similar but not identical
- Ignoring broadcasting rules for elementwise ops → shape errors
- Assuming `Dropout` does something at inference (usually optimized away)

---

## Next Steps
→ [02_OpSet_Versions](../02_OpSet_Versions/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
