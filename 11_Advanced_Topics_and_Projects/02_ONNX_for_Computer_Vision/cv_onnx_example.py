#!/usr/bin/env python3
"""
Computer Vision ONNX inference + optional visualization.

Features
- Inspect ONNX I/O metadata
- Build a sample ResNet-18 ONNX (random weights) with PyTorch export
- Run ORT with a random tensor or an image file path
- Optional matplotlib visualization (top-k bar chart)

Dependencies
  pip install onnx onnxruntime numpy pillow matplotlib torch torchvision
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install onnxruntime") from e


def build_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


def summarize_session(sess: ort.InferenceSession) -> Dict[str, object]:
    def dump(ts) -> List[Dict[str, object]]:
        return [{"name": t.name, "type": str(t.type), "shape": list(t.shape)} for t in ts]

    return {"inputs": dump(sess.get_inputs()), "outputs": dump(sess.get_outputs()), "providers": sess.get_providers()}


def create_resnet18_onnx(out_path: str, opset: int = 17) -> None:
    import torch
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Wrote sample ResNet-18 ONNX to: {out_path}")


def imagenet_normalize(chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    return (chw - mean) / std


def load_image_nchw(path: str, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,C
    chw = np.transpose(arr, (2, 0, 1))[None, ...]  # 1,C,H,W
    return imagenet_normalize(chw)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return (e / np.sum(e)).astype(np.float32)


def topk(p: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-p)[:k]
    return idx, p[idx]


def maybe_plot(image_path: Optional[str], probs: np.ndarray, out_png: Optional[str]) -> None:
    import matplotlib.pyplot as plt

    k = min(10, probs.size)
    idx, vals = topk(probs, k)

    fig, axes = plt.subplots(1, 2 if image_path else 1, figsize=(10, 4))
    if image_path:
        from PIL import Image

        ax_img, ax_bar = axes
        ax_img.imshow(Image.open(image_path).convert("RGB"))
        ax_img.set_title("Input")
        ax_img.axis("off")
    else:
        ax_bar = axes

    ax_bar.bar(np.arange(k), vals)
    ax_bar.set_xticks(np.arange(k))
    ax_bar.set_xticklabels([str(i) for i in idx], rotation=45, ha="right")
    ax_bar.set_title("Top-k softmax probabilities")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
        print(f"Saved figure: {out_png}")
    else:
        plt.show()


def run_model(sess: ort.InferenceSession, tensor: np.ndarray, input_name: Optional[str]) -> np.ndarray:
    if not input_name:
        input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    out = sess.run([out_name], {input_name: tensor})[0]
    return np.asarray(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, help="Path to ONNX model")
    p.add_argument("--create-sample", type=str, help="Write a sample ResNet-18 ONNX to this path and exit")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--random-input", action="store_true")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--input-name", type=str, default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--plot", type=str, default=None, help="Optional path to save matplotlib figure PNG")
    args = p.parse_args()

    if args.create_sample:
        create_resnet18_onnx(args.create_sample, opset=args.opset)
        return

    if not args.model:
        raise SystemExit("Provide --model or use --create-sample to generate a demo ONNX model.")

    sess = build_session(args.model)
    print(json.dumps(summarize_session(sess), indent=2))

    if args.image:
        x = load_image_nchw(args.image, args.size)
    elif args.random_input:
        x = np.random.randn(1, 3, args.size, args.size).astype(np.float32)
    else:
        raise SystemExit("Provide --image PATH or --random-input")

    logits = run_model(sess, x, args.input_name).reshape(-1)
    probs = softmax(logits)
    idx, vals = topk(probs, args.topk)
    print("top indices:", idx.tolist())
    print("top probs: ", [float(v) for v in vals])

    if args.plot or args.image:
        maybe_plot(args.image, probs, args.plot)


if __name__ == "__main__":
    main()
