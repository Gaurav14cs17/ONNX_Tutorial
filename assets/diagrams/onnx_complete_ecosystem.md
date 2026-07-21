# ONNX Complete Ecosystem Diagram

```mermaid
graph TB
    subgraph Training["Training Frameworks"]
        PT[PyTorch]
        TF[TensorFlow/Keras]
        SK[Scikit-Learn]
        XG[XGBoost]
        LG[LightGBM]
        PP[PaddlePaddle]
        MX[MXNet]
    end

    subgraph Converters["ONNX Converters"]
        TE[torch.onnx.export]
        T2O[tf2onnx]
        S2O[skl2onnx]
        OML[onnxmltools]
        P2O[paddle2onnx]
    end

    subgraph ONNX_Core["ONNX Standard"]
        direction TB
        FORMAT[".onnx Model File"]
        IR["IR Specification"]
        OPSET["Operator Sets"]
        PROTO["Protobuf Schema"]

        FORMAT --- IR
        FORMAT --- OPSET
        FORMAT --- PROTO
    end

    subgraph Optimization["Optimization Tools"]
        GOPT[Graph Optimizations]
        QUANT[Quantization]
        PRUNE[Pruning/Sparsity]
        OPTIM[onnxoptimizer]
    end

    subgraph Runtime["ONNX Runtime"]
        direction TB
        ORT_CORE["ORT Core Engine"]
        CPU_EP["CPU EP (MLAS)"]
        CUDA_EP["CUDA EP"]
        TRT_EP["TensorRT EP"]
        OV_EP["OpenVINO EP"]
        DML_EP["DirectML EP"]
        CM_EP["CoreML EP"]
        NNAPI_EP["NNAPI EP"]

        ORT_CORE --> CPU_EP
        ORT_CORE --> CUDA_EP
        ORT_CORE --> TRT_EP
        ORT_CORE --> OV_EP
        ORT_CORE --> DML_EP
        ORT_CORE --> CM_EP
        ORT_CORE --> NNAPI_EP
    end

    subgraph Deployment["Deployment Targets"]
        CLOUD["☁️ Cloud Servers"]
        EDGE["⚡ Edge Devices"]
        MOBILE["📱 Mobile"]
        WEB["🌐 Web Browser"]
    end

    subgraph Tools["Ecosystem Tools"]
        NETRON["Netron (Visualizer)"]
        ZOO["ONNX Model Zoo"]
        HUB["ONNX Hub"]
        CHECKER["Model Checker"]
    end

    PT --> TE
    TF --> T2O
    SK --> S2O
    XG --> OML
    LG --> OML
    PP --> P2O
    MX --> OML

    TE --> FORMAT
    T2O --> FORMAT
    S2O --> FORMAT
    OML --> FORMAT
    P2O --> FORMAT

    FORMAT --> GOPT
    FORMAT --> QUANT
    FORMAT --> PRUNE
    GOPT --> OPTIM

    FORMAT --> ORT_CORE
    OPTIM --> ORT_CORE

    CPU_EP --> CLOUD
    CUDA_EP --> CLOUD
    TRT_EP --> CLOUD
    OV_EP --> EDGE
    NNAPI_EP --> MOBILE
    CM_EP --> MOBILE
    DML_EP --> WEB

    FORMAT --> NETRON
    ZOO --> FORMAT
    HUB --> FORMAT
    FORMAT --> CHECKER

    style ONNX_Core fill:#ff6b35,stroke:#333,color:#fff
    style Runtime fill:#2196F3,stroke:#333,color:#fff
    style Training fill:#4CAF50,stroke:#333,color:#fff
    style Deployment fill:#9C27B0,stroke:#333,color:#fff
    style Optimization fill:#FF9800,stroke:#333,color:#fff
    style Tools fill:#607D8B,stroke:#333,color:#fff
```

## Lifecycle Diagram

```mermaid
sequenceDiagram
    participant DS as Data Scientist
    participant TF as Training Framework
    participant ONNX as ONNX Format
    participant OPT as Optimizer
    participant ORT as ONNX Runtime
    participant PROD as Production

    DS->>TF: Train Model
    TF->>TF: Evaluate & Tune
    TF->>ONNX: Export to .onnx
    ONNX->>ONNX: Validate (onnx.checker)
    ONNX->>OPT: Optimize Graph
    OPT->>OPT: Quantize (FP32→INT8)
    OPT->>ORT: Load Optimized Model
    ORT->>ORT: Select Execution Provider
    ORT->>PROD: Deploy (Cloud/Edge/Mobile)
    PROD->>PROD: Serve Inference Requests
```

## Model Architecture (Internal Structure)

```mermaid
classDiagram
    class ModelProto {
        +int64 ir_version
        +repeated OperatorSetIdProto opset_import
        +string producer_name
        +string producer_version
        +string domain
        +int64 model_version
        +string doc_string
        +GraphProto graph
        +repeated StringStringEntryProto metadata_props
    }

    class GraphProto {
        +string name
        +repeated NodeProto node
        +repeated TensorProto initializer
        +repeated ValueInfoProto input
        +repeated ValueInfoProto output
        +string doc_string
    }

    class NodeProto {
        +string op_type
        +repeated string input
        +repeated string output
        +string name
        +string domain
        +repeated AttributeProto attribute
    }

    class TensorProto {
        +repeated int64 dims
        +int32 data_type
        +string name
        +repeated float float_data
        +repeated bytes raw_data
    }

    class ValueInfoProto {
        +string name
        +TypeProto type
        +string doc_string
    }

    class AttributeProto {
        +string name
        +AttributeType type
        +float f
        +int64 i
        +string s
        +TensorProto t
        +GraphProto g
    }

    ModelProto "1" --> "1" GraphProto : graph
    GraphProto "1" --> "*" NodeProto : nodes
    GraphProto "1" --> "*" TensorProto : initializers
    GraphProto "1" --> "*" ValueInfoProto : inputs/outputs
    NodeProto "1" --> "*" AttributeProto : attributes
```
