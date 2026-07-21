# Mobile Deployment with ONNX Runtime Mobile

> **Interview Relevance:** HIGH — On-device ML for mobile is a rapidly growing field; expect questions on NNAPI/CoreML Execution Providers, quantization for mobile, and battery/memory trade-offs.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

```
 +-----------------------------+        +-----------------------------+
 | Android / iOS app           |        | Shared native module        |
 |  UI thread (60/120 fps)     |        |  ORT Mobile session         |
 |  Camera / mic / sensors     | <----> |   + CPU EP (always avail.)  |
 +--------------+--------------+        |   + NNAPI / CoreML EP       |
                |                       |   + XNNPACK (case-dependent)|
                |                       +--------------+--------------+
                |                                      |
                v                                      v
        +---------------+                    +-------------------+
        | Preprocess    |                    | Postprocess       |
        | resize, norm  |                    | NMS, thresholds   |
        +---------------+                    +-------------------+
                \______________________________________/
                                  |
                                  v
                         +---------------------+
                         | ONNX model (.onnx)   |
                         +---------------------+
```

## Why (Motivation)

Mobile deployment optimizes for **battery life**, **thermal headroom**, **privacy** (on-device processing), and **offline resilience**. ONNX serves as a portable graph that can be accelerated by mobile NPUs and DSPs through Execution Providers like Android NNAPI and Apple CoreML, enabling the same model to run across both platforms.

## When (Use Cases)

- **Camera-based apps** with real-time object detection or face recognition
- **Voice assistants** running speech-to-text on-device for privacy
- **Health/fitness apps** processing sensor data without cloud dependency
- **Augmented reality** requiring low-latency pose estimation
- **Offline-capable apps** in areas with poor connectivity

## How (Mechanism)

ONNX Runtime Mobile is a footprint-conscious ORT variant targeting Android and iOS. On Android, the NNAPI EP delegates supported subgraphs to GPU/DSP/NPU hardware; on iOS, the CoreML EP targets CPU/GPU/Neural Engine. Unsupported ops fall back to CPU EP. The model ships as a bundled asset, and inference runs on background threads to avoid UI jank.

---

## Prerequisites
- [02_Cloud_Deployment](../02_Cloud_Deployment/) — Compare cloud vs on-device trade-offs
- [08_Model_Optimization_and_Quantization](../../08_Model_Optimization_and_Quantization/) — Quantization for mobile

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| ORT Mobile architecture and packaging | Foundation for mobile ML integration | ⭐⭐⭐ |
| Android NNAPI Execution Provider | Hardware acceleration on Android devices | ⭐⭐⭐ |
| iOS CoreML Execution Provider | Apple Neural Engine acceleration | ⭐⭐⭐ |
| Quantization for mobile (INT8/FP16) | Model size and battery impact | ⭐⭐⭐ |
| Flutter/React Native integration patterns | Cross-platform mobile ML | ⭐⭐ |
| Battery and memory management | Production mobile ML reliability | ⭐⭐ |
| UI thread architecture for inference | Preventing jank in mobile apps | ⭐⭐ |

## Key Interview Questions Answered Here

1. **How does NNAPI accelerate ONNX inference on Android?** → NNAPI delegates supported subgraphs to vendor-specific GPU/DSP/NPU implementations; unsupported ops fall back to CPU EP, resulting in partitioned graphs.
2. **What's the biggest challenge with mobile ML deployment?** → Device variance: the same APK behaves differently across OEMs due to different NNAPI implementations, thermal limits, and memory constraints.
3. **How do you prevent UI jank during inference?** → Run inference on background threads/queues with backpressure; never block the UI thread (60/120 fps) with model execution.
4. **When should you use FP16 vs INT8 on mobile?** → FP16 aligns well with mobile GPU paths (Apple/Android); INT8 is better for NPU/DSP targets but requires accuracy validation with calibration data.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Mobile_Deployment_Deep_Dive.ipynb](01_Mobile_Deployment_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Mobile_Deployment_Apply.ipynb](02_Mobile_Deployment_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions

- **Testing only on emulators** — NNAPI/CoreML behavior differs dramatically on real devices vs emulators.
- **Ignoring device variance** — Same model + same APK can have wildly different performance across OEM NNAPI implementations.
- **Warming up on battery** — Running warm-up inferences at launch wastes battery if the feature is rarely used.
- **Sending tensors over JS bridge repeatedly** — In React Native/Flutter, keep tensor data native-side to avoid bridge overhead.
- **Not matching preprocessing** — RGB/BGR order and mean/std must exactly match training pipeline.

---

## Next Steps
→ [04_Web_Deployment_ONNX_JS](../04_Web_Deployment_ONNX_JS/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
