# ONNX Protobuf message hierarchy (Mermaid)

This diagram mirrors Chapter **02** and condenses the ONNX IR into a **Mermaid class-style** graph. It is intended for quick navigation, not as a replacement for reading **`onnx.proto`**.

> **Rendering:** GitHub-style Markdown renders Mermaid. VS Code + a Mermaid preview extension works locally. Export to SVG/PNG using [`mermaid-cli`](https://github.com/mermaid-js/mermaid-cli) if you need a static image for slides.

```mermaid
classDiagram
    class ModelProto {
        int64 ir_version
        OperatorSetIdProto[] opset_import
        string producer_name
        string producer_version
        string domain
        int64 model_version
        string doc_string
        GraphProto graph
        StringStringEntryProto[] metadata_props
        TrainingInfoProto[] training_info
        FunctionProto[] functions
    }

    class OperatorSetIdProto {
        string domain
        int64 version
    }

    class GraphProto {
        string name
        string doc_string
        NodeProto[] node
        TensorProto[] initializer
        SparseTensorProto[] sparse_initializer
        ValueInfoProto[] input
        ValueInfoProto[] output
        ValueInfoProto[] value_info
        TensorAnnotation[] quantization_annotation
        StringStringEntryProto[] metadata_props
    }

    class NodeProto {
        string name
        string op_type
        string domain
        string overload
        string[] input
        string[] output
        AttributeProto[] attribute
        string doc_string
        StringStringEntryProto[] metadata_props
    }

    class AttributeProto {
        string name
        string ref_attr_name
        AttributeType type
        float f
        int64 i
        bytes s
        TensorProto t
        GraphProto g
        SparseTensorProto sparse_tensor
        TypeProto tp
        float[] floats
        int64[] ints
        bytes[] strings
        TensorProto[] tensors
        GraphProto[] graphs
    }

    class ValueInfoProto {
        string name
        TypeProto type
        string doc_string
        StringStringEntryProto[] metadata_props
    }

    class TensorProto {
        int64[] dims
        int32 data_type
        float[] float_data
        int32[] int32_data
        bytes[] string_data
        int64[] int64_data
        double[] double_data
        bytes raw_data
        string name
        string doc_string
        StringStringEntryProto[] external_data
        int32 data_location
    }

    class SparseTensorProto {
    }

    class TypeProto {
    }

    class FunctionProto {
    }

    class StringStringEntryProto {
        string key
        string value
    }

    class TrainingInfoProto {
        GraphProto initialization
        GraphProto algorithm
        StringStringEntryProto[] initialization_binding
        StringStringEntryProto[] update_binding
    }

    ModelProto "1" o-- "many" OperatorSetIdProto : opset_import
    ModelProto "1" *-- "1" GraphProto : graph
    ModelProto "1" o-- "many" StringStringEntryProto : metadata_props
    ModelProto "1" o-- "many" TrainingInfoProto : training_info
    ModelProto "1" o-- "many" FunctionProto : functions

    GraphProto "1" *-- "many" NodeProto : node
    GraphProto "1" o-- "many" TensorProto : initializer
    GraphProto "1" o-- "many" SparseTensorProto : sparse_initializer
    GraphProto "1" o-- "many" ValueInfoProto : input
    GraphProto "1" o-- "many" ValueInfoProto : output
    GraphProto "1" o-- "many" ValueInfoProto : value_info

    NodeProto "1" *-- "many" AttributeProto : attribute

    AttributeProto ..> TensorProto : optional tensor / tensors
    AttributeProto ..> GraphProto : optional graph / graphs
    AttributeProto ..> TypeProto : optional type / type_protos
    AttributeProto ..> SparseTensorProto : optional sparse_tensor / sparse_tensors

    ValueInfoProto ..> TypeProto : type

    TensorProto "1" o-- "many" StringStringEntryProto : external_data
```

### Legend

- **Solid lines with diamond** — primary containment (`ModelProto` owns its `GraphProto`).
- **Dashed arrows** — “may embed” relationships for **attributes** (subgraphs, constant tensors).
- **`StringStringEntryProto`** — reused map-like entries for **metadata_props** and **TensorProto.external_data** key/value pairs.
- This diagram **does not** expand **`TypeProto`**, **`SparseTensorProto`**, or **`FunctionProto`** bodies—those messages are sizable on their own; open `onnx.proto` for exhaustive field lists.

---

*Chapter: [README.md](../README.md)*
