#!/usr/bin/env python3
"""
Export the trained CIFAR CNN to ONNX and optionally apply dynamic INT8 quantization.

Example:
  python export_and_optimize.py --checkpoint artifacts/last.pt --onnx-out artifacts/model.onnx

Notes
- Uses CPU export for maximum portability in tutorials.
- Validates ONNX with onnx.checker when available.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnx
    from onnx import checker
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install onnx") from e


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class SmallCnn(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_checkpoint(path: str) -> Dict[str, object]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with 'state_dict' (see train_model.py).")
    return ckpt


def export_onnx(
    model: nn.Module,
    onnx_path: str,
    opset: int,
) -> None:
    model.eval()
    dummy = torch.randn(1, 3, 32, 32, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    m = onnx.load(onnx_path)
    checker.check_model(m)
    print(f"ONNX export OK (checked): {onnx_path}")


def maybe_quantize(src_onnx: str, dst_onnx: str) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=src_onnx,
        model_output=dst_onnx,
        weight_type=QuantType.QUInt8,
    )
    print(f"Wrote dynamically quantized model: {dst_onnx}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--onnx-out", type=str, required=True)
    p.add_argument("--quantized-out", type=str, default=None)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    meta = ckpt.get("meta", {})
    print("meta:", json.dumps(meta, indent=2))

    model = SmallCnn(num_classes=int(meta.get("num_classes", 10)))
    model.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]
    model.eval()

    os.makedirs(os.path.dirname(args.onnx_out) or ".", exist_ok=True)
    export_onnx(model, args.onnx_out, opset=args.opset)

    if args.quantized_out:
        os.makedirs(os.path.dirname(args.quantized_out) or ".", exist_ok=True)
        maybe_quantize(args.onnx_out, args.quantized_out)

    # Quick parity: PyTorch vs ORT on a random tensor
    try:
        import onnxruntime as ort
    except ImportError:
        print("Install onnxruntime to run parity check: pip install onnxruntime")
        return

    x = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    with torch.no_grad():
        y_ref = model(x).numpy()

    sess = ort.InferenceSession(
        args.onnx_out,
        providers=["CPUExecutionProvider"],
    )
    y_ort = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.max(np.abs(y_ref - y_ort)))
    print(f"Parity max_abs_diff (random tensor): {max_diff:.6e}")


if __name__ == "__main__":
    main()
