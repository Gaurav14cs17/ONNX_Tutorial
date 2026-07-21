# Protobuf Primer

> **Interview Relevance:** HIGH — Understanding why ONNX uses Protocol Buffers (compact binary, schema-first, cross-language) is a foundational question in ML infrastructure interviews.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

## Why (Motivation)
Protocol Buffers are the serialization backbone of ONNX. Without understanding Protobuf, errors from converters and runtimes remain mysterious. Knowing the *define → compile → serialize* workflow demystifies how `.onnx` files encode graphs and tensors.

## When (Use Cases)
- Debugging ONNX model corruption or validation errors
- Understanding why model files are compact yet not human-readable
- Contributing to ONNX tooling or writing custom serialization logic
- Explaining ONNX's design decisions in system design interviews

## How (Mechanism)
ONNX packages its intermediate representation as Protobuf messages defined in `onnx.proto`. A `.onnx` file is a serialized `ModelProto` message. The Python `onnx` package ships pre-generated stubs so you never need to run `protoc` yourself.

---

## Prerequisites
- [Chapter overview](../README.md) — ONNX as an interchange format
- Basic Python environment with `pip install onnx`

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Protobuf as binary schema-first format | Explains ONNX file structure | ⭐⭐⭐ |
| `.proto` syntax (messages, fields, enums) | Read ONNX schema definitions | ⭐⭐ |
| Serialize / Deserialize lifecycle | Debug load/save issues | ⭐⭐⭐ |
| Protobuf vs JSON vs XML trade-offs | Justify ONNX design choices | ⭐⭐⭐ |
| ONNX's proto2 dialect | Understand `optional`/`repeated` semantics | ⭐⭐ |

## Key Interview Questions Answered Here
1. **Why does ONNX use Protocol Buffers instead of JSON?** → Compact binary, fast parsing, strong schema enforcement, cross-language codegen, and built-in evolution rules for versioning.
2. **What is a `.proto` file?** → A human-readable schema that defines message types, field numbers, and types; compiled into language-specific serialization code.
3. **How does Protobuf handle backward compatibility?** → Field numbers are never reused; new fields get new numbers; parsers skip unknown fields gracefully.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Protobuf_Primer_Deep_Dive.ipynb](01_Protobuf_Primer_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Protobuf_Primer_Apply.ipynb](02_Protobuf_Primer_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions
- Confusing `ir_version` (container schema) with `opset_import` (operator semantics)
- Thinking you need to compile `.proto` yourself — the `onnx` package ships pre-generated code
- Assuming Protobuf is human-readable — use Netron or helper printers for debugging
- Forgetting that field numbers are wire identifiers, not ordinal positions

---

## Next Steps
→ [02_ONNX_Proto_Structure](../02_ONNX_Proto_Structure/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
