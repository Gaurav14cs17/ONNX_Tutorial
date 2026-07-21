# End-to-End Project: Image Classification API with ONNX

> **Interview Relevance:** HIGH — End-to-end ML system design is the most important interview topic for ML engineers; this project demonstrates the complete train → export → optimize → deploy → monitor pipeline.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

```
+------------------+     +-------------------+      +---------------------+
| CIFAR-10 dataset | --> | train_model.py    | -->  | checkpoint.pt       |
+------------------+     | (PyTorch training)|      +----------+----------+
                         +-------------------+                 |
                                                               v
                                                  +-----------------------+
                                                  | export_and_optimize.py|
                                                  | torch.onnx.export     |
                                                  +-----------+-----------+
                                                              |
                                                              v
                                                  +-----------------------+
                                                  | model.onnx            |
                                                  | model.int8.onnx (opt) |
                                                  +-----------+-----------+
                                                              |
                                                              v
                                                  +-----------------------+
                                                  | deploy_api.py         |
                                                  | FastAPI + ORT session |
                                                  +-----------+-----------+
                                                              |
                                                  +-----------+-----------+
                                                  | Benchmark / monitor   |
                                                  | hey/k6 + logs         |
                                                  +-----------------------+
```

## Why (Motivation)

This project ties together every theme in the tutorial: training a model in PyTorch, exporting it to ONNX with verifiable I/O metadata, optimizing via graph settings and dynamic quantization, deploying behind FastAPI with a stable inference contract, and benchmarking with health/metadata endpoints. It mirrors how real teams ship ML models to production.

## When (Use Cases)

- **Learning exercise** to practice the complete ML deployment lifecycle
- **Template** for real-world image classification APIs
- **Interview preparation** for ML system design questions
- **Starting point** for custom ONNX deployment pipelines

## How (Mechanism)

1. `train_model.py` trains a small CNN on CIFAR-10 and saves a checkpoint
2. `export_and_optimize.py` exports to ONNX, validates with `onnx.checker`, and optionally applies INT8 dynamic quantization
3. `deploy_api.py` serves predictions via FastAPI with `/health`, `/predict` (JSON tensor), and `/predict_file` (image upload) endpoints
4. Benchmarking tools (hey/k6) measure latency and throughput

---

## Prerequisites
- [03_ONNX_for_Generative_AI](../03_ONNX_for_Generative_AI/) — Advanced ONNX patterns
- [10_Deployment/02_Cloud_Deployment](../../10_Deployment/02_Cloud_Deployment/) — Cloud serving fundamentals

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| PyTorch training → ONNX export pipeline | Complete ML lifecycle | ⭐⭐⭐ |
| ONNX checker validation | Export correctness verification | ⭐⭐ |
| Dynamic INT8 quantization | CPU throughput optimization | ⭐⭐⭐ |
| FastAPI + ORT serving | Production serving pattern | ⭐⭐⭐ |
| PyTorch vs ORT parity testing | Model validation best practice | ⭐⭐⭐ |
| Benchmarking and monitoring endpoints | Production readiness | ⭐⭐ |
| I/O contract documentation | Preventing training/serving skew | ⭐⭐⭐ |

## Key Interview Questions Answered Here

1. **Walk me through deploying an ML model end-to-end.** → Train (CIFAR-10 CNN) → Export (torch.onnx.export with shape validation) → Optimize (INT8 quantization) → Serve (FastAPI + ORT) → Monitor (/health + benchmarks).
2. **How do you validate an ONNX export?** → Run `onnx.checker.check_model`, then compare PyTorch vs ORT outputs on identical inputs with `np.allclose`.
3. **What should a /health endpoint return?** → Model path, loaded status, providers, input/output tensor metadata (names, shapes, types).
4. **How do you handle both JSON tensor and image file inputs?** → Two endpoints: `/predict` for raw tensor JSON, `/predict_file` for multipart image upload with server-side preprocessing.

## Project Scripts

| Step | Script | Purpose |
|------|--------|---------|
| Train | [`train_model.py`](./train_model.py) | Train CNN & save checkpoint |
| Export | [`export_and_optimize.py`](./export_and_optimize.py) | ONNX export + optional INT8 |
| Serve | [`deploy_api.py`](./deploy_api.py) | REST API with ORT |

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [End_to_End_Project_Deep_Dive.ipynb](01_End_to_End_Project_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [End_to_End_Project_Apply.ipynb](02_End_to_End_Project_Apply.ipynb) | Hands-on practice | ~15 min |

## Quick Start

```bash
# Step 1: Train
python train_model.py --epochs 5 --batch-size 128 --out artifacts/last.pt

# Step 2: Export + Quantize
python export_and_optimize.py \
  --checkpoint artifacts/last.pt \
  --onnx-out artifacts/model.onnx \
  --quantized-out artifacts/model.int8.onnx

# Step 3: Serve
export MODEL_PATH=artifacts/model.onnx
uvicorn deploy_api:app --host 0.0.0.0 --port 9000

# Step 4: Benchmark
hey -n 200 -c 10 http://127.0.0.1:9000/health
```

## Common Mistakes & Confusions

- **Not freezing normalization constants** — CIFAR mean/std must match between `train_model.py` and `deploy_api.py` exactly.
- **Skipping the parity test** — `export_and_optimize.py` includes a PyTorch vs ORT comparison; never skip this step.
- **Using training transforms at serving time** — Random augmentations (flip, crop) are for training only; serving uses deterministic preprocessing.
- **Forgetting to validate INT8 accuracy** — Quantized models can lose accuracy on specific classes; always compare per-class metrics.
- **Not versioning the ONNX artifact** — Pin model checksum in `/health` metadata for production traceability.

---

## Stretch Goals

- Swap CPU for CUDA EP in Docker with GPU base images
- Add Prometheus metrics (`/metrics`) with `prometheus_client`
- Add OpenTelemetry tracing for request spans
- Implement A/B testing between FP32 and INT8 models

---

## Next Steps
→ Continue to build continuous evaluation pipelines and Model Cards for responsible AI review.

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
