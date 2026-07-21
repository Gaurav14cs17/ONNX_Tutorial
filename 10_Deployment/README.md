# Chapter 10: Deployment

This chapter connects your ONNX model to **real-world surfaces**: **edge devices**, **cloud platforms**, **mobile apps**, and the **browser**. The same `.onnx` artifact can travel across these targets when you pair it with the right **runtime**, **hardware execution providers**, **packaging** (containers, app bundles), and **operational practices** (scaling, observability, cost).

---

## Learning Order

| # | Topic | Focus | Time | Prerequisites |
|:-:|-------|-------|:----:|---------------|
| 1 | [01 — Edge Deployment](./01_Edge_Deployment/) | IoT/edge, Raspberry Pi, Jetson, Intel NCS, optimization | ~35 min | Chapters 6–9 |
| 2 | [02 — Cloud Deployment](./02_Cloud_Deployment/) | Docker, Kubernetes, Azure ML / SageMaker / Vertex AI, REST & gRPC | ~35 min | 01_Edge_Deployment |
| 3 | [03 — Mobile Deployment](./03_Mobile_Deployment/) | ORT Mobile, Android NNAPI, iOS CoreML, quantization, Flutter/RN | ~35 min | 02_Cloud_Deployment |
| 4 | [04 — Web Deployment (ONNX.js / ORT Web)](./04_Web_Deployment_ONNX_JS/) | WASM/WebGL/WebGPU, browser inference, JS/TS examples | ~35 min | 03_Mobile_Deployment |

---

## Chapter at a Glance

| Section | Title | Focus |
|---------|-------|-------|
| [01 — Edge Deployment](./01_Edge_Deployment/README.md) | IoT/edge concepts, ORT on constrained hardware, Raspberry Pi, Jetson, Intel NCS, optimization, step-by-step guide |
| [02 — Cloud Deployment](./02_Cloud_Deployment/README.md) | Docker/Kubernetes, Azure ML / SageMaker / Vertex AI, REST & gRPC serving, scaling, platform comparison |
| [03 — Mobile Deployment](./03_Mobile_Deployment/README.md) | ORT Mobile, Android NNAPI, iOS CoreML EP, quantization/pruning, Flutter/React Native, battery/memory |
| [04 — Web Deployment (ONNX.js / ORT Web)](./04_Web_Deployment_ONNX_JS/README.md) | WASM/WebGL/WebGPU, browser architecture, JS/TS examples, image classification demo |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Contrast** deployment targets (edge, cloud, mobile, web) by **latency budgets**, **throughput**, **privacy**, **connectivity**, and **operational complexity**.
2. **Explain** how **ONNX Runtime** selects **Execution Providers** and applies **graph optimizations** differently on **ARM SoCs**, **Jetson**, **Intel accelerators**, **mobile NPUs**, and **browser/WebGPU** stacks.
3. **Design** an **edge deployment**: choose a **model size** and **precision** (FP32/FP16/INT8), a **build** of ORT for your ABI/architecture, and a **serving pattern** (embedded loop vs local microservice).
4. **Package** ONNX inference in **Docker**, deploy to **Kubernetes**, and expose **REST** or **gRPC** APIs with sane **timeouts**, **payload limits**, and **health checks**.
5. **Navigate** major **cloud ML platforms** (Azure ML, SageMaker, Vertex AI) for **ONNX model registration**, **managed endpoints**, and **MLOps** concerns (versions, staging, monitoring).
6. **Integrate** **ONNX Runtime Mobile** into **Android** and **iOS** apps, understanding **NNAPI** and **CoreML** EP trade-offs and **memory** constraints.
7. **Run** inference in the **browser** with **ONNX Runtime Web**, choosing among **WASM**, **WebGL**, and **WebGPU** backends based on **compatibility** and **performance**.
8. **Apply** a repeatable deployment checklist: **correct input/output tensor contracts**, **pre/post-processing parity** with training, **benchmarks**, and **failure modes** (OOM, thermal throttling, cold starts).

---

## Prerequisites

### Knowledge

- Chapters on **exporting to ONNX**, **ONNX Runtime sessions**, and **quantization** (strongly recommended).
- Basic familiarity with **HTTP APIs**, **containers**, and (for mobile) **native app lifecycles**.

### Software (varies by lesson)

```bash
# Typical Python baseline used across cloud/edge examples
pip install onnx onnxruntime numpy

# API lesson
pip install fastapi uvicorn python-multipart

# Optional GPU (match your CUDA stack)
# pip install onnxruntime-gpu
```

---

## Diagrams in This Chapter

| Path | Description |
|------|-------------|
| [diagrams/deployment_overview.md](./diagrams/deployment_overview.md) | **Mermaid**: deployment targets overview |

---

## Mental Model: One Model, Many Runtimes

```
  Train / export              Same ONNX                 Different surfaces
 +------------------+    +------------------+    +------------------------------+
 | Framework export | -> | model.onnx       | -> | Edge: ORT + ARM CPU / NPU   |
 | (torch.onnx etc.)|    | + metadata       |    | Cloud: ORT in container/K8s |
 +------------------+    +------------------+    | Mobile: ORT Mobile + EPs    |
                                                 | Web: ORT Web / WASM / WebGPU|
                                                 +------------------------------+
```

---

## How to Use This Chapter

- **Pick a target** early: the "best" deployment is the one that meets **latency**, **cost**, **privacy**, and **reliability** constraints—not the newest accelerator.
- **Lock I/O contracts**: document **input tensor names/shapes/dtypes** and **normalization**; most production incidents come from **training/serving skew**.
- **Measure on hardware**: micro-benchmarks on a laptop rarely predict **Jetson thermals**, **browser WASM variance**, or **API gateway cold starts**.

---

## Next Chapter

Continue to [Chapter 11: Advanced Topics and Real-World Projects](../11_Advanced_Topics_and_Projects/README.md) for **NLP**, **computer vision**, **generative AI** patterns, and a **full end-to-end project**.

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
