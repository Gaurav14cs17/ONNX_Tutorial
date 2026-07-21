# ONNX Ecosystem Overview (Mermaid)

This diagram positions **training frameworks** on the left, the **ONNX** interchange in the center, and **inference runtimes / deployment targets** on the right—matching the mental model used throughout Chapter 1.

```mermaid
flowchart LR
  subgraph FW["Training frameworks"]
    direction TB
    PyTorch["PyTorch"]
    TensorFlow["TensorFlow / Keras"]
    Sklearn["scikit-learn"]
    Other["Other frameworks & tools"]
  end

  subgraph Hub["ONNX interchange"]
    direction TB
    ONNXFile["ONNX model (.onnx)\ngraph + weights + opset"]
  end

  subgraph RT["Inference runtimes & deployment"]
    direction TB
    ORT["ONNX Runtime"]
    TensorRT["TensorRT (NVIDIA)"]
    OpenVINO["OpenVINO (Intel)"]
    Mobile["Mobile / embedded SDKs"]
    Web["Web / WASM runtimes"]
  end

  PyTorch --> |"export (torch.onnx / torch.export)"| ONNXFile
  TensorFlow --> |"convert (tf2onnx, etc.)"| ONNXFile
  Sklearn --> |"convert (skl2onnx)"| ONNXFile
  Other --> |"custom converters"| ONNXFile

  ONNXFile --> ORT
  ONNXFile --> TensorRT
  ONNXFile --> OpenVINO
  ONNXFile --> Mobile
  ONNXFile --> Web
```

## How to render

- **GitHub / GitLab / many Markdown viewers:** Mermaid is rendered natively in several products.
- **Local preview:** use a Mermaid-capable editor or the Mermaid CLI to export PNG/SVG if you need a static image for slides.

## Legend

| Lane | Role |
|------|------|
| **Training frameworks** | Where parameters are learned and native graphs live |
| **ONNX** | Portable, declarative computation contract between teams |
| **Inference runtimes** | Where latency, throughput, and hardware mapping are optimized |
