# Chapter 11: Advanced Topics and Real-World Projects

This chapter combines **advanced ONNX usage patterns** across **natural language processing**, **computer vision**, and **generative AI**, then culminates in an **end-to-end** project that mirrors how teams ship: **train → export → optimize → serve → observe**.

---

## Learning Order

| # | Topic | Focus | Time | Prerequisites |
|:-:|-------|-------|:----:|---------------|
| 1 | [01 — ONNX for NLP](./01_ONNX_for_NLP/) | Transformer export, tokenizers, dynamic shapes, Optimum | ~35 min | Chapter 10 |
| 2 | [02 — ONNX for Computer Vision](./02_ONNX_for_Computer_Vision/) | Classification, detection (YOLO/SSD), segmentation, pre/post-processing | ~35 min | 01_ONNX_for_NLP |
| 3 | [03 — ONNX for Generative AI](./03_ONNX_for_Generative_AI/) | Stable Diffusion, LLMs, audio models, memory strategies | ~35 min | 02_ONNX_for_CV |
| 4 | [04 — End-to-End Project](./04_End_to_End_Project/) | Image classification API: train, export, optimize, deploy, benchmark | ~35 min | All previous |

---

## Chapter at a Glance

| Section | Title | Focus |
|---------|-------|-------|
| [01 — ONNX for NLP](./01_ONNX_for_NLP/README.md) | Transformer export (BERT/GPT-2), tokenizers, dynamic shapes, Optimum, full inference example |
| [02 — ONNX for Computer Vision](./02_ONNX_for_Computer_Vision/README.md) | Classification, detection (YOLO/SSD), segmentation, pre/post-processing, visualization example |
| [03 — ONNX for Generative AI](./03_ONNX_for_Generative_AI/README.md) | Stable Diffusion ONNX, LLMs, audio models, optimization & memory strategies |
| [04 — End-to-End Project](./04_End_to_End_Project/README.md) | Image classification API: train, export, optimize, deploy, benchmark |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Export** transformer-based NLP models to ONNX with correct **input/output tensor bundles** (`input_ids`, `attention_mask`, optional `token_type_ids`) and understand how **dynamic axes** map to variable **sequence lengths**.
2. **Pair** ONNX models with **Hugging Face tokenizers** (or equivalent) without training/serving skew—**normalization**, **special tokens**, and **padding/truncation** parity.
3. **Run** **computer vision** models in ONNX for **classification**, **detection**, and **segmentation**, implementing **preprocessing** (resize/crop/normalize) and **postprocessing** (NMS, masks, thresholding) as explicit, testable code.
4. **Describe** how **generative** workloads (diffusion, LLMs) stress **memory** and **attention** patterns, and how ONNX-based toolchains address **splitting**, **quantization**, and **caching** strategies—at a systems level.
5. **Execute** a complete **PyTorch → ONNX → ORT → FastAPI** deployment, including **quantization** experiments and **basic monitoring hooks**.
6. **Validate** models using **golden vectors**, **property tests** (shapes/dtypes), and **domain metrics** (accuracy, mAP, perplexity proxies)—not only unit tests of plumbing.

---

## Prerequisites

- Chapters on **export**, **ONNX Runtime**, and **quantization** (strongly recommended).
- Python 3.9+ and comfort with **PyTorch** basics.

### Baseline Installs (varies by section)

```bash
pip install onnx onnxruntime numpy torch torchvision pillow requests matplotlib

# NLP lesson (optional but recommended)
pip install transformers optimum[onnxruntime] sentencepiece

# API lesson
pip install fastapi uvicorn python-multipart
```

---

## Diagrams in This Chapter

| Path | Description |
|------|-------------|
| [diagrams/project_pipeline.md](./diagrams/project_pipeline.md) | **Mermaid**: end-to-end project pipeline |

---

## How to Use This Chapter

- Treat each section as a **pattern library**—reuse the checklists in design reviews.
- For generative AI, prioritize **memory and KV-cache realities** over benchmark bragging rights.
- The **end-to-end project** is intentionally "small but complete" so you can execute it on a laptop CPU.

---

## Next Steps After This Book

- Build **continuous evaluation** pipelines (offline + shadow traffic).
- Standardize **Model Cards** and **Responsible AI** review gates alongside technical gates.

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
