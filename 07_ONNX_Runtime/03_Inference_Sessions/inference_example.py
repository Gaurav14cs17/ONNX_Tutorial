"""
Complete, self-contained ONNX Runtime inference demonstration.

What this script demonstrates
-----------------------------
1. Build a tiny ONNX model at runtime (so the repo doesn't need a bundled .onnx file).
2. Create an InferenceSession with explicit SessionOptions.
3. Run inference with named input/output tensors.
4. Demonstrate a "dynamic batch" sweep on the batch dimension.

Requirements
------------
    pip install onnx onnxruntime numpy

Optional GPU notes
----------------
This script defaults to CPUExecutionProvider for maximum portability. If you have a
GPU-enabled ORT build, you can prioritize CUDA or other EPs via CLI flags.

"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TOY_MODEL = os.path.join(_SCRIPT_DIR, "toy_linear_relu.onnx")


def _require(pkgs: List[Tuple[str, str]]) -> None:
    """Import dependencies with actionable error messages."""
    for import_name, pip_name in pkgs:
        try:
            __import__(import_name)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(f"Missing dependency '{pip_name}'. Install with: pip install {pip_name}") from e


def build_toy_onnx_model(path: str, opset: int = 17) -> None:
    """
    Creates a minimal ONNX graph: Y = Relu(X @ W + B)

    - X: [N, in_features]
    - W: [in_features, out_features] (initializer)
    - B: [out_features] (initializer)
    """
    _require([("onnx", "onnx")])

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    in_features = 8
    out_features = 4

    # Deterministic, reproducible weights
    rng = np.random.default_rng(0)
    W = rng.standard_normal((in_features, out_features)).astype(np.float32)
    B = rng.standard_normal((out_features,)).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", in_features])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", out_features])

    initializers = [
        numpy_helper.from_array(W, name="W"),
        numpy_helper.from_array(B, name="B"),
    ]

    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["XW"]),
        helper.make_node("Add", ["XW", "B"], ["S"]),
        helper.make_node("Relu", ["S"], ["Y"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="ToyLinearRelu",
        inputs=[X],
        outputs=[Y],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_session(model_path: str, intra_threads: int, inter_threads: int, prefer_cuda: bool):
    import onnxruntime as ort

    available = ort.get_available_providers()

    providers: List[str | Tuple[str, dict]] = []
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": "0"}))
    providers.append("CPUExecutionProvider")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads

    # Note: execution_mode may help for wide DAGs; chain graphs often don't benefit.
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    return sess


def warmup(sess, feed: dict, repeats: int = 20) -> None:
    for _ in range(repeats):
        sess.run(None, feed)


def benchmark(sess, feed: dict, repeats: int = 100) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        sess.run(None, feed)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ONNX Runtime inference session demo.")
    parser.add_argument("--model", default="", help="Path to ONNX model. If omitted, a toy model is generated.")
    parser.add_argument(
        "--out",
        default=_DEFAULT_TOY_MODEL,
        help="Where to write the generated toy model (defaults next to this script).",
    )
    parser.add_argument("--intra", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--inter", type=int, default=1)
    parser.add_argument("--prefer-cuda", action="store_true", help="Prefer CUDA EP if available.")
    parser.add_argument("--warmup", type=int, default=30, help="Warm-up runs.")
    parser.add_argument("--bench", type=int, default=200, help="Benchmark runs for timing.")

    args = parser.parse_args(argv)

    _require([("onnxruntime", "onnxruntime"), ("numpy", "numpy")])

    import onnxruntime as ort

    model_path = args.model
    if not model_path:
        build_toy_onnx_model(args.out)
        model_path = args.out
        print(f"[info] wrote toy model to {model_path}")
    elif not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    sess = make_session(model_path, intra_threads=args.intra, inter_threads=args.inter, prefer_cuda=args.prefer_cuda)

    print("[info] ORT version:", ort.__version__)
    print("[info] Available providers:", ort.get_available_providers())
    print("[info] Session providers:", sess.get_providers())

    xinfo = sess.get_inputs()[0]
    print("[info] Input:", xinfo.name, "shape=", xinfo.shape, "type=", xinfo.type)

    # Dynamic batch sweep: N is free per run.
    for n in (1, 4, 16):
        x = np.random.randn(n, 8).astype(np.float32)
        feed = {xinfo.name: x}

        warmup(sess, feed, repeats=args.warmup)
        ms = benchmark(sess, feed, repeats=args.bench) * 1000.0

        y = sess.run(None, feed)[0]
        print(f"[result] N={n:3d}  y.shape={y.shape}  mean={y.mean():+.6f}  p50~ {ms:0.3f} ms/run (local timing)")

    print("\n[done] This script is a teaching baseline—use ORT profiling for serious performance work (Chapter 04).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
