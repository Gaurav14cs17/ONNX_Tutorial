# ONNX for Computer Vision: Classification, Detection, and Segmentation

> **Interview Relevance:** HIGH — CV model deployment is the most mature ONNX use case; expect questions on preprocessing parity, NMS implementation, and NCHW vs NHWC tensor formats.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

```
+-------------------+       +---------------------------+
| Input image (RGB) |  -->  | Preprocess (CPU)          |
| PIL / OpenCV      |       | resize, pad, normalize    |
+---------+---------+       +--------------+------------+
          |                                |
          v                                v
+-------------------+       +---------------------------+
| ONNX (CV model)   |  <--  | Tensor [N,C,H,W] float32 |
| ORT session       |       +---------------------------+
+---------+---------+
          |
          v
+---------------------------+        +--------------------+
| Raw outputs               |  -->   | Postprocess        |
| boxes / logits / masks    |        | softmax,NMS,argmax |
+---------------------------+        +---------+----------+
                                               |
                                               v
                                     +-------------------+
                                     | Visualize / API   |
                                     +-------------------+
```

## Why (Motivation)

Computer vision deployments hinge on **consistent pre/post-processing**. ONNX captures the network — not your OpenCV resize rules, letterboxing, or NMS thresholds — so production systems wrap ORT with explicit, testable vision pipelines. Understanding these boundaries is critical for reliable CV model deployment.

## When (Use Cases)

- **Image classification APIs** (ResNet, EfficientNet) for product categorization
- **Object detection systems** (YOLO, SSD) for surveillance and quality inspection
- **Semantic segmentation** for autonomous driving and medical imaging
- **Instance segmentation** for robotics and AR applications
- **Real-time video analytics** pipelines with camera feeds

## How (Mechanism)

Load the ONNX model with ORT, preprocess images into NCHW float32 tensors with exact training-time normalization, run inference, then postprocess outputs (softmax for classification, NMS + box decoding for detection, argmax for segmentation). Every preprocessing parameter (resize method, mean/std, channel order) must be frozen and shared between training and serving.

---

## Prerequisites
- [01_ONNX_for_NLP](../01_ONNX_for_NLP/) — Compare NLP vs CV ONNX patterns
- [06_Exporting_Models_to_ONNX](../../06_Exporting_Models_to_ONNX/) — Export mechanics

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Classification model I/O contracts | Foundation for all CV deployment | ⭐⭐⭐ |
| Object detection postprocessing (NMS) | Most complex CV deployment step | ⭐⭐⭐ |
| Preprocessing parity (mean/std/resize) | #1 source of CV serving bugs | ⭐⭐⭐ |
| NCHW vs NHWC tensor format handling | Critical for cross-framework deployment | ⭐⭐⭐ |
| Segmentation output processing | Growing area in production CV | ⭐⭐ |
| Visualization and debugging patterns | Essential for model validation | ⭐⭐ |

## Key Interview Questions Answered Here

1. **Why is NMS usually outside the ONNX graph?** → NMS is post-processing that varies by deployment context (confidence threshold, IoU threshold, max detections). Keeping it outside allows tuning without re-exporting.
2. **How do you ensure preprocessing parity?** → Freeze preprocessing parameters in a single module shared by train, eval, and serve. Store reference tensors for CI parity tests.
3. **NCHW vs NHWC — how do you know which to use?** → Inspect the model's input metadata (`sess.get_inputs()[0].shape`). PyTorch exports typically use NCHW; TensorFlow-derived graphs may use NHWC.
4. **What's the most common silent failure in CV deployment?** → Channel order mismatch (RGB vs BGR) causes the model to produce confident but wrong predictions.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [ONNX_for_CV_Deep_Dive.ipynb](01_ONNX_for_CV_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [ONNX_for_CV_Apply.ipynb](02_ONNX_for_CV_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions

- **RGB vs BGR channel order** — OpenCV reads BGR by default; most ONNX models expect RGB. Silent color flip causes confident wrong predictions.
- **Wrong resize interpolation** — Using nearest-neighbor instead of bilinear changes accuracy; match training exactly.
- **Forgetting letterbox inverse mapping** — Detection box coordinates must be mapped back to original image space after letterboxing.
- **Applying softmax twice** — Some exports include softmax in the graph; applying it again in postprocessing is wrong.
- **Aspect ratio distortion** — Using fixed-square resize when training used short-side resize causes systematic accuracy loss.

---

## Next Steps
→ [03_ONNX_for_Generative_AI](../03_ONNX_for_Generative_AI/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
