# End-to-end image classification pipeline

This diagram summarizes the **Chapter 10** project flow: train in PyTorch, export to ONNX, optimize/quantize for deployment constraints, serve over HTTP, then benchmark and monitor.

```mermaid
flowchart LR
  subgraph dev["Development"]
    TRAIN[train_model.py\nPyTorch training]
    DATA[(CIFAR-10)]
    DATA --> TRAIN
    TRAIN --> CKPT[checkpoint.pt]
  end

  subgraph export["ONNX & optimization"]
    EX[export_and_optimize.py\nexport + shape check]
    Q[dynamic quantization\noptional]
    CKPT --> EX
    EX --> ONNX[model.onnx]
    EX --> Q
    Q --> ONNXQ[model.int8.onnx]
  end

  subgraph serve["Serving"]
    API[deploy_api.py\nFastAPI + ORT]
    ONNX --> API
    ONNXQ --> API
  end

  subgraph observe["Benchmark & monitor"]
    BENCH[Latency throughput\nscripts / hey / k6]
    LOGS[/health + metrics hooks/]
    API --> BENCH
    API --> LOGS
  end
```

**Reading notes**

- The arrow from **checkpoint** to **export** implies you must **freeze architecture** and **weights** reproducibly; avoid “mystery” checkpoints without commit hashes.
- **Quantization** produces a separate artifact—version both and evaluate accuracy deltas.
