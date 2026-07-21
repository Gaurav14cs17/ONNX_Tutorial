# Cloud Deployment with ONNX Runtime

> **Interview Relevance:** HIGH — Cloud ML deployment is the most common production pattern; expect questions on containerization, REST/gRPC serving, Kubernetes scaling, and managed ML platform trade-offs.

---

## Visual Overview
![ONNX Deployment Targets — Cloud, edge, mobile, and web deployment paths](assets/onnx_deployment_targets.png)

```
  +-------------+      +----------------+      +-------------------------+
  | Data / apps | --> | Ingress + WAF    | --> | AuthN/Z + rate limiting  |
  +-------------+      +--------+-------+      +------------+------------+
                                |                            |
                                v                            v
                     +---------------------+        +---------------------+
                     | API Gateway / LB    | -----> | Service mesh / mTLS |
                     +----------+----------+        +----------+----------+
                                |
                                v
          +--------------------------------------------+
          |  Kubernetes Deployment (HPA / KEDA autoscale)|
          |   Pod: container( FastAPI + ORT )            |
          |        - readiness: model loaded            |
          |        - liveness: process healthy          |
          +--------------------+-------------------------+
                               |
                               v
                  +------------+-------------+
                  | Observability stack       |
                  | traces + metrics + logs   |
                  +---------------------------+
```

## Why (Motivation)

Cloud deployment optimizes for **elastic scale**, **managed operations**, and **multi-tenant APIs**. ONNX models integrate cleanly because container images can ship a pinned ONNX Runtime build and a stable server process. The same `.onnx` artifact can be served across Azure ML, AWS SageMaker, Google Vertex AI, or custom Kubernetes clusters.

## When (Use Cases)

- **High-throughput prediction APIs** serving thousands of requests per second
- **Multi-model serving** behind a unified gateway for A/B testing and canary deployments
- **Managed ML platforms** for teams needing integrated monitoring, versioning, and rollback
- **GPU-accelerated batch inference** where dynamic batching maximizes utilization
- **Globally distributed endpoints** requiring load balancing and regional failover

## How (Mechanism)

Package ORT + your model in a Docker container, expose REST or gRPC endpoints (e.g., FastAPI), deploy to Kubernetes with readiness/liveness probes, and scale via HPA or custom metrics. Cloud ML platforms (Azure ML, SageMaker, Vertex AI) abstract this further with managed endpoints, model registries, and built-in observability.

---

## Prerequisites
- [01_Edge_Deployment](../01_Edge_Deployment/) — Contrast edge vs cloud trade-offs
- [07_ONNX_Runtime](../../07_ONNX_Runtime/) — Session creation and inference basics

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| Docker containerization for ORT | Foundation for reproducible ML serving | ⭐⭐⭐ |
| Kubernetes deployment with health probes | Industry-standard orchestration | ⭐⭐⭐ |
| REST API serving with FastAPI | Most common serving pattern | ⭐⭐⭐ |
| gRPC serving trade-offs | High-throughput internal services | ⭐⭐ |
| Azure ML / SageMaker / Vertex AI comparison | Platform selection is a design interview staple | ⭐⭐⭐ |
| Scaling and load balancing strategies | Critical for production reliability | ⭐⭐⭐ |
| Dynamic batching patterns | GPU utilization optimization | ⭐⭐ |

## Key Interview Questions Answered Here

1. **How do you containerize an ONNX model for production?** → Build a Docker image with pinned ORT + dependencies, mount model via volume, expose health/inference endpoints, run as non-root.
2. **What's the difference between readiness and liveness probes?** → Readiness waits for model load + GPU init; liveness confirms the process is healthy. Both prevent serving broken replicas.
3. **How do you choose between Azure ML, SageMaker, and Vertex AI?** → Optimize for your org's IAM model, data residency, existing CI/CD, and GPU quota — not the vendor logo.
4. **When would you use gRPC over REST for model serving?** → Internal east-west traffic with large tensor payloads or streaming; REST is simpler for external-facing APIs.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [Cloud_Deployment_Deep_Dive.ipynb](01_Cloud_Deployment_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [Cloud_Deployment_Apply.ipynb](02_Cloud_Deployment_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions

- **Loading the model per request** — Load once at startup; per-request loading causes catastrophic latency.
- **Missing readiness probes** — Kubernetes sends traffic before model is loaded, causing 503 errors.
- **Ignoring cold start** — First inference after model load is significantly slower; warm replicas before traffic switches.
- **Setting timeouts at p50** — Set client/server timeouts above p99 inference latency, not p50.
- **Baking secrets into Docker layers** — Mount models and credentials via volumes or config maps.

---

## Next Steps
→ [03_Mobile_Deployment](../03_Mobile_Deployment/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
