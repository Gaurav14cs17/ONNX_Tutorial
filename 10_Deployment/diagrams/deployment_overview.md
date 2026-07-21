# Deployment overview (Chapter 9)

This diagram situates **ONNX** as a portable artifact that can be executed across **edge**, **cloud**, **mobile**, and **web** stacks using **ONNX Runtime** family products and appropriate **Execution Providers (EPs)**.

```mermaid
flowchart TB
  subgraph train["Train & Export"]
    PT[PyTorch / TensorFlow / JAX]
    EX[Export to ONNX]
    PT --> EX
  end

  subgraph artifact["Portable artifact"]
    M[model.onnx + weights]
    META[Metadata: inputs, opset, normalization]
    EX --> M
    EX --> META
  end

  subgraph edge["Edge / IoT"]
    RPI[Raspberry Pi / ARM Linux]
    JET[NVIDIA Jetson]
    INT[Intel NCS / OpenVINO EP]
    M --> RPI
    M --> JET
    M --> INT
  end

  subgraph cloud["Cloud"]
    DK[Docker image]
    K8s[Kubernetes Service]
    AML[Azure ML]
    SM[AWS SageMaker]
    VAI[Google Vertex AI]
    M --> DK
    DK --> K8s
    M --> AML
    M --> SM
    M --> VAI
  end

  subgraph mobile["Mobile"]
    AND[Android + NNAPI EP]
    IOS[iOS + CoreML EP]
    M --> AND
    M --> IOS
  end

  subgraph web["Browser"]
    OW[ONNX Runtime Web]
    WASM[WASM]
    WG[WebGPU / WebGL]
    M --> OW
    OW --> WASM
    OW --> WG
  end

  ORT[ONNX Runtime family]
  ORT -. orchestrates .-> edge
  ORT -. orchestrates .-> cloud
  ORT -. orchestrates .-> mobile
  ORT -. orchestrates .-> web
```

---

## Reading guide

- **Solid lines** show the **model artifact** flowing into each platform.
- **Dotted lines** emphasize that **ONNX Runtime** (including mobile/web variants) is the common execution layer—actual EP availability depends on **device**, **build**, and **graph operators**.
