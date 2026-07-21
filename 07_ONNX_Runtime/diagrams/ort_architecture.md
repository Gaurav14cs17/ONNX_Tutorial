# ONNX Runtime: Architecture (Mermaid)

This diagram summarizes how an ONNX model flows through **ONNX Runtime (ORT)**, from **loading** through **optimization**, **partitioning** to **Execution Providers**, and **execution** with **allocators** and **kernels**.

> **Reading tip:** follow the numbered path for the “steady state” inference pipeline; the **fallback** edge shows what happens when a preferred EP declines a subgraph.

```mermaid
flowchart TB
  subgraph App["Application layer"]
    PY["Python / C / C# / Java / Objective-C API"]
  end

  subgraph ORT["ONNX Runtime core"]
    LOAD["Model load\n(onnx protobuf → graph IR)"]
    GS["Graph optimization\nrewrites + fusions"]
    PART["Partitioning\nassign subgraphs to EPs"]
    PLAN["Execution plan\norder + memory reuse"]
    EXEC["Session run\n(scheduling + kernels)"]
  end

  subgraph MEM["Memory & execution services"]
    ALLOC["Allocators\n(CPU/GPU pools, arenas)"]
    THREADS["Thread pools\n(intra-op / inter-op)"]
  end

  subgraph EPs["Execution Providers (examples)"]
    EP_CPU["CPU EP\n(MLAS / Eigen / parallel loops)"]
    EP_CUDA["CUDA EP"]
    EP_TRT["TensorRT EP"]
    EP_OV["OpenVINO EP"]
    EP_DML["DirectML EP"]
    EP_OTHER["… / CoreML / NNAPI / ROCm / …"]
  end

  PY --> LOAD
  LOAD --> GS
  GS --> PART
  PART --> PLAN
  PLAN --> EXEC

  EXEC --> ALLOC
  EXEC --> THREADS

  PART --> EP_CPU
  PART --> EP_CUDA
  PART --> EP_TRT
  PART --> EP_OV
  PART --> EP_DML
  PART --> EP_OTHER

  EP_CPU -. fallback .-> EP_CUDA
  EP_CUDA -. fallback .-> EP_CPU
  EP_TRT -. partial support .-> EP_CUDA

  EXEC --> EP_CPU
  EXEC --> EP_CUDA
  EXEC --> EP_TRT
  EXEC --> EP_OV
  EXEC --> EP_DML
  EXEC --> EP_OTHER
```

## Legend

| Symbol | Meaning |
|--------|---------|
| **Graph optimization** | Hardware-agnostic graph transforms (constant folding, fusion, layout planning where applicable). |
| **Partitioning** | Each ONNX node/subgraph is assigned to an EP that claims it; some EPs compile **fused** regions. |
| **Fallback** | If a node cannot run on a preferred EP, ORT may place it on another EP in the priority list. |
| **TensorRT EP nuance** | TRT may handle a fused segment while **neighboring** ops still run on CUDA or CPU. |

## Companion materials

- Narrative walkthrough: [01 — ORT Architecture](../01_ORT_Architecture/README.md)
- EP catalog: [02 — Execution Providers](../02_Execution_Providers/README.md)
