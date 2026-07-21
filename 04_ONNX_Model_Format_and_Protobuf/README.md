# 04 — ONNX Model Format and Protocol Buffers

This chapter moves from *what ONNX is* to *how it is represented on disk and in memory*. ONNX models follow a strict, versioned **schema** expressed as **Google Protocol Buffers (Protobuf)**. Understanding that schema is the key to debugging converters, authoring diagnostic models, and reasoning about large deployments.

---

## Learning Order

| # | Topic | Focus | Time |
|:-:|-------|-------|:----:|
| 1 | [01_Protobuf_Primer](./01_Protobuf_Primer/) | Serialization fundamentals, why ML uses binary schemas | ~35 min |
| 2 | [02_ONNX_Proto_Structure](./02_ONNX_Proto_Structure/) | ModelProto → GraphProto → NodeProto hierarchy | ~35 min |
| 3 | [03_Reading_and_Writing_Models](./03_Reading_and_Writing_Models/) | load/save, external data, large model strategies | ~35 min |
| 4 | [04_Model_Metadata_and_Versioning](./04_Model_Metadata_and_Versioning/) | IR version, operator sets, practical versioning | ~35 min |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Explain** what Protocol Buffers are and why ONNX uses them
2. **Navigate** the ONNX message hierarchy from ModelProto down through NodeProto
3. **Load and save** models with external weights and memory-efficient patterns
4. **Interpret and set** model metadata and version information

---

## Prerequisites

- Basic ONNX vocabulary (interchange format, training vs inference)
- Python 3.8+ with `pip install onnx`

## Files and Assets

| Path | Description |
|------|-------------|
| [03_Reading_and_Writing_Models/read_write_example.py](./03_Reading_and_Writing_Models/read_write_example.py) | Runnable script: round-trip save/load, external data |
| [diagrams/protobuf_structure.md](./diagrams/protobuf_structure.md) | Mermaid diagram of the ONNX Protobuf hierarchy |

---

*Next: [05_ONNX_Operators_and_OpSets](../05_ONNX_Operators_and_OpSets/)*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
