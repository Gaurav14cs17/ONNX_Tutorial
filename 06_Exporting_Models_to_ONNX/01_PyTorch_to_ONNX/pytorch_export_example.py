#!/usr/bin/env python3
"""
Chapter 5 — PyTorch → ONNX minimal end-to-end example.

Steps:
  1. Define a small CNN for MNIST-style inputs (1x28x28).
  2. Export to ONNX with dynamic batch dimension.
  3. Validate with onnx.checker + optional shape inference.
  4. Compare PyTorch vs ONNX Runtime outputs (L∞ / max abs diff).

Requirements:
  pip install torch onnx onnxruntime numpy
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import checker


class SmallCNN(nn.Module):
    """Tiny CNN: 1x28x28 -> logits (10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1, 28, 28]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def export_model(
    model: nn.Module,
    onnx_path: str,
    *,
    opset: int = 17,
) -> None:
    model.eval()
    dummy = torch.randn(2, 1, 28, 28, dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch"},
        "logits": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )


def validate_onnx(path: str) -> onnx.ModelProto:
    model = onnx.load(path)
    checker.check_model(model)

    try:
        from onnx import shape_inference

        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] shape_inference skipped: {exc}")

    return model


def summarize_model(m: onnx.ModelProto, max_initializers: int = 5) -> None:
    g = m.graph
    print("\n--- ONNX graph summary ---")
    print(f"IR version: {m.ir_version}, producer: {m.producer_name} {m.producer_version}")
    print("Inputs:")
    for i in g.input:
        t = i.type.tensor_type
        shape = [d.dim_value if d.HasField("dim_value") else "?" for d in t.shape.dim]
        print(f"  {i.name}: elem_type={t.elem_type} shape={shape}")
    print("Outputs:")
    for o in g.output:
        t = o.type.tensor_type
        shape = [d.dim_value if d.HasField("dim_value") else "?" for d in t.shape.dim]
        print(f"  {o.name}: elem_type={t.elem_type} shape={shape}")

    inits = [init.name for init in g.initializer]
    print(f"Initializers: {len(inits)} total; first {max_initializers}: {inits[:max_initializers]}")


def run_ort(path: str, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x.astype(np.float32)})
    return out[0]


def compare_outputs(pt: torch.Tensor, onnx_out: np.ndarray) -> None:
    pt_np = pt.detach().cpu().numpy()
    diff = np.abs(pt_np - onnx_out)
    print("\n--- Numeric comparison (PyTorch vs ONNX Runtime) ---")
    print(f"max abs diff: {diff.max():.6e}")
    print(f"mean abs diff: {diff.mean():.6e}")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = SmallCNN(num_classes=10)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "small_cnn.onnx")
        export_model(model, onnx_path)

        print(f"Exported ONNX to temporary path:\n  {onnx_path}")
        m = validate_onnx(onnx_path)
        summarize_model(m)

        batch = 4
        x_torch = torch.randn(batch, 1, 28, 28, dtype=torch.float32)
        with torch.no_grad():
            y_pt = model(x_torch)

        y_ort = run_ort(onnx_path, x_torch.numpy())
        compare_outputs(y_pt, y_ort)

        # Simple self-check: ONNX graph contains conv + gemm-style nodes
        ops = {n.op_type for n in m.graph.node}
        print("\nUnique ops in exported graph:", sorted(ops))


if __name__ == "__main__":
    main()
