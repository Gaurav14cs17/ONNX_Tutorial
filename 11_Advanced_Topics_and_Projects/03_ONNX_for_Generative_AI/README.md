# ONNX for Generative AI: Diffusion, LLMs, and Audio

> **Interview Relevance:** HIGH — Generative AI optimization is the hottest topic in ML engineering; expect questions on multi-model pipelines, KV-cache management, and memory optimization strategies.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

```
+----------------------+        +--------------------------+
| Text prompt          | -----> | Text encoder (ONNX opt.) |
| "A fox in ONNX..."  |        | -> text embeddings       |
+----------------------+        +-------------+------------+
                                              |
                                              v
+----------------------+        +--------------------------+
| Noise z ~ N(0, I)   | -----> | Scheduler loop (CPU)     |
| timestep schedule    |        | repeat T steps:          |
+----------------------+        |   UNet(x_t, t, text)     |
                                              |
                                              v
                                +--------------------------+
                                | UNet denoiser (ONNX)     |
                                +-------------+------------+
                                              |
                                              v
                                +--------------------------+
                                | VAE decode (ONNX)        |
                                | latents -> RGB image     |
                                +--------------------------+
```

## Why (Motivation)

Generative AI workloads push ONNX from "run a CNN once" into **multi-stage pipelines**, **autoregressive loops**, and **multi-GB weights**. ONNX remains valuable as a portable interchange and as a subgraph accelerator, but engineering focus shifts to memory management, compilation efficiency, caching strategies, and operator coverage across execution providers.

## When (Use Cases)

- **Text-to-image** (Stable Diffusion) with ONNX-accelerated UNet and VAE
- **LLM inference** with ONNX Runtime for transformer subgraphs
- **Speech-to-text** (Whisper) with ONNX-optimized encoder/decoder
- **Text-to-speech** with streaming audio generation
- **Multi-modal pipelines** combining vision and language models

## How (Mechanism)

Generative models are typically split into multiple ONNX subgraphs (e.g., text encoder, UNet, VAE for diffusion). A Python/C++ scheduler orchestrates the iterative pipeline (sampling loop for diffusion, autoregressive decoding for LLMs) while ORT accelerates the heavy neural kernels. Memory management is critical: precision reduction (FP16/INT8/INT4), attention chunking, and buffer reuse prevent OOM on large models.

---

## Prerequisites
- [02_ONNX_for_Computer_Vision](../02_ONNX_for_Computer_Vision/) — CV pipeline patterns that generative AI builds upon
- [08_Model_Optimization_and_Quantization](../../08_Model_Optimization_and_Quantization/) — Quantization for large models

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Stable Diffusion ONNX pipeline architecture | Understand multi-model orchestration | ⭐⭐⭐ |
| LLM ONNX export and KV-cache management | Foundation for LLM serving optimization | ⭐⭐⭐ |
| Memory management for generative models | Prevents OOM and enables larger models | ⭐⭐⭐ |
| Voice/audio model ONNX patterns | Growing area for on-device AI | ⭐⭐ |
| INT8/INT4 weight quantization for LLMs | Key to making LLMs fit in memory | ⭐⭐⭐ |
| Where ONNX stops and systems engineering starts | Realistic production perspective | ⭐⭐ |

## Key Interview Questions Answered Here

1. **Why is diffusion typically multi-model?** → Text encoder, UNet, and VAE have different compute profiles and can be individually optimized/quantized. The scheduler loop runs imperatively outside ONNX.
2. **What is KV-cache and why does it matter for LLM serving?** → KV-cache stores computed key/value attention tensors to avoid recomputation during autoregressive generation. Whether it lives inside or outside the ONNX graph is an architecture decision.
3. **What are the three main memory levers for generative AI?** → Precision (FP16/INT8/INT4), resolution/sequence length, and batch size — each trades quality or throughput for memory.
4. **Is ONNX enough for a production generative AI system?** → No. ONNX accelerates neural kernels, but production systems also need model governance, safety filters, fallback strategies, and observability beyond latency.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Generative_AI_Deep_Dive.ipynb](01_Generative_AI_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Generative_AI_Apply.ipynb](02_Generative_AI_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions

- **Treating generative AI as "one ONNX file"** — It's almost always multiple models orchestrated by external scheduling code.
- **Ignoring peak memory** — Peak memory is the product of resolution × batch × precision × attention patterns; it's easy to OOM.
- **Applying cloud scaling naively** — Generative models have fundamentally different latency profiles (many sequential steps) vs single-inference models.
- **Confusing export format with runtime** — ONNX is often a compiler input to vendor runtimes (TensorRT, CoreML), not the final execution artifact.
- **Overlooking STFT/mel alignment** — For audio models, spectral preprocessing mismatches cause silent quality degradation.

---

## Next Steps
→ [04_End_to_End_Project](../04_End_to_End_Project/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
