# Web Deployment: ONNX.js and ONNX Runtime Web

> **Interview Relevance:** MEDIUM — Browser-based ML is a growing niche; expect questions on WASM vs WebGPU trade-offs, preprocessing parity in JavaScript, and web worker architectures for smooth UI.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

```
 +--------------------+      +-------------------------+
 | Web app (React/Vue)|      | Preprocess (JS/WASM)    |
 |   UI + canvas      | ---> | resize tensors, norm    |
 +---------+----------+      +------------+------------+
           |                              |
           v                              v
 +--------------------+      +-------------------------+
 | onnxruntime-web    | <--> | Model assets (.onnx)    |
 | session + backends |      | fetched as static / CDN |
 +---------+----------+      +-------------------------+
           |
           +----> WASM backend (wide support)
           +----> WebGPU backend (when available)
           +----> WebGL backend (legacy/compatibility)
           v
 +-------------------------+
 | Postprocess (JS)        |
 | softmax, label mapping  |
 +-------------------------+
```

## Why (Motivation)

Browser-based inference enables **zero-install** experiences, **client-side privacy**, and **interactive** ML demos. Users get instant predictions without uploading data to a server, eliminating latency and privacy concerns. ONNX Runtime Web provides multiple backends (WASM, WebGL, WebGPU) to balance compatibility with performance.

## When (Use Cases)

- **Interactive demos** showcasing ML models without backend infrastructure
- **Privacy-sensitive applications** where data must never leave the client
- **Educational tools** for live ML experimentation in the browser
- **Progressive web apps** with offline ML capabilities
- **Client-side image/text processing** to reduce server costs

## How (Mechanism)

ONNX Runtime Web (`onnxruntime-web`) loads `.onnx` model files as static assets, creates an inference session with a selected backend (WASM for broad compatibility, WebGPU for GPU acceleration), and runs inference in JavaScript/TypeScript. Web Workers offload computation from the main thread to prevent UI blocking.

---

## Prerequisites
- [03_Mobile_Deployment](../03_Mobile_Deployment/) — Compare on-device vs browser trade-offs
- [07_ONNX_Runtime](../../07_ONNX_Runtime/) — ORT session fundamentals

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| ONNX.js vs ONNX Runtime Web evolution | Understanding the ecosystem | ⭐⭐ |
| WASM backend characteristics | Most compatible browser ML path | ⭐⭐⭐ |
| WebGPU backend and future direction | Modern GPU compute in browser | ⭐⭐ |
| JS/TS session creation and tensor handling | Core browser ML programming | ⭐⭐⭐ |
| Web Worker architecture for inference | Preventing UI jank | ⭐⭐⭐ |
| Image classification in-browser pipeline | End-to-end browser ML demo | ⭐⭐ |
| Model caching and versioning strategies | Production browser ML delivery | ⭐⭐ |

## Key Interview Questions Answered Here

1. **WASM vs WebGPU — when do you use each?** → WASM for broad compatibility and small models; WebGPU for larger models where GPU dramatically beats WASM. Always keep WASM as a fallback.
2. **How do you prevent the browser from freezing during inference?** → Use Web Workers: main thread handles UI, worker thread runs ORT session. Transfer preprocessed Float32Arrays via transferable objects.
3. **What's the biggest risk in browser ML preprocessing?** → Preprocessing parity: every resize/crop/normalization rule must match the training export exactly, or accuracy silently degrades.
4. **How do you handle large ONNX models in the browser?** → HTTP caching with versioned URLs, CDN delivery, progress indicators during fetch, and session caching to avoid recompilation.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Web_Deployment_Deep_Dive.ipynb](01_Web_Deployment_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Web_Deployment_Apply.ipynb](02_Web_Deployment_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions

- **Assuming WebGPU is always faster** — For tiny models, WASM can be faster due to GPU dispatch overhead.
- **Blocking the main thread** — Running inference without Web Workers causes visible UI freezes.
- **Wrong mean/std in JS preprocessing** — Produces all-NaN outputs; the most common browser ML debugging issue.
- **Not showing loading progress** — Model fetch + session compile causes visible latency; users need feedback.
- **Ignoring WASM threading headers** — WASM multi-threading requires `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers.

---

## Next Steps
→ [11_Advanced_Topics_and_Projects/01_ONNX_for_NLP](../../11_Advanced_Topics_and_Projects/01_ONNX_for_NLP/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
