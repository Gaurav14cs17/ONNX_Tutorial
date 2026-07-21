#!/usr/bin/env python3
"""
Benchmark tutorial script (Chapter 7).

Benchmarks ONNX Runtime inference with:
  - explicit providers (default: CPU)
  - warmup iterations
  - timed iterations
  - percentile statistics

Requirements:
  pip install numpy onnx onnxruntime

Usage examples:
  python benchmark_example.py --model model.onnx --repeats 200
  python benchmark_example.py  # builds a tiny ONNX model and benchmarks it
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Iterable, List, Tuple

import numpy as np

import onnx
from onnx import TensorProto, helper


def _build_tiny_model(path: str) -> None:
    """
    Build y = Relu(x - 1) with unknown batch dimension for X: [N, 4]
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 4])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1, 4], np.ones((1, 4), dtype=np.float32))

    c1 = helper.make_node("Constant", [], ["one"], value=one)
    sub = helper.make_node("Sub", ["X", "one"], ["t"])
    relu = helper.make_node("Relu", ["t"], ["Y"])
    graph = helper.make_graph([c1, sub, relu], "relu_shift", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[ok] wrote toy model to {path}")


def _random_feed_shapes(model: onnx.ModelProto) -> List[Tuple[str, Tuple[int, ...], np.dtype]]:
    """
    Produce random input tensors matching model graph inputs (static dims only).
    """
    feeds: List[Tuple[str, Tuple[int, ...], np.dtype]] = []
    for inp in model.graph.input:
        name = inp.name
        dims: List[int] = []
        ok = True
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                # Bind a simple default for dynamic batch in our toy model
                dims.append(1 if d.dim_param == "N" else 1)
            else:
                ok = False
                break
        if not ok or not dims:
            raise RuntimeError(f"Cannot synthesize random input for {name} (dynamic/unknown shape).")
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type).lower()
        np_dtype = {
            "float": np.float32,
            "double": np.float64,
            "int64": np.int64,
            "int32": np.int32,
        }.get(dtype, np.float32)
        feeds.append((name, tuple(dims), np_dtype))
    return feeds


def _make_random_inputs(
    spec: List[Tuple[str, Tuple[int, ...], np.dtype]], rng: np.random.Generator
) -> dict:
    out = {}
    for name, shape, np_dtype in spec:
        if np_dtype in (np.float32, np.float64):
            out[name] = rng.standard_normal(shape, dtype=np_dtype)
        else:
            out[name] = rng.integers(-3, 4, size=shape, dtype=np_dtype)
    return out


def _percentiles(samples: List[float], ps: Iterable[float]) -> dict[float, float]:
    arr = np.array(samples, dtype=np.float64)
    return {float(p): float(np.percentile(arr, p)) for p in ps}


def benchmark_session(
    model_path: str,
    *,
    providers: List[str],
    warmup: int,
    repeats: int,
    intra_op_threads: int | None,
    inter_op_threads: int | None,
) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if intra_op_threads is not None:
        so.intra_op_num_threads = int(intra_op_threads)
    if inter_op_threads is not None:
        so.inter_op_num_threads = int(inter_op_threads)

    sess = ort.InferenceSession(model_path, so, providers=providers)
    inputs = sess.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs.")

    proto = onnx.load(model_path)
    feed_spec = _random_feed_shapes(proto)
    rng = np.random.default_rng(0)

    # Use one fixed input sample for stable timing (cache-friendly); optionally vary if needed
    feed = _make_random_inputs(feed_spec, rng)

    outputs = [o.name for o in sess.get_outputs()]

    def one_run() -> float:
        t0 = time.perf_counter()
        sess.run(outputs, feed)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0  # ms

    for _ in range(warmup):
        one_run()

    times: List[float] = [one_run() for _ in range(repeats)]

    p = _percentiles(times, [50, 95, 99])
    return {
        "n": len(times),
        "mean_ms": float(statistics.mean(times)),
        "stdev_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "p50_ms": p[50.0],
        "p95_ms": p[95.0],
        "p99_ms": p[99.0],
        "throughput_rps": 1000.0 / float(statistics.mean(times)) if times else 0.0,
        "providers": providers,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="benchmark_toy.onnx")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--provider", action="append", default=None, help="Repeatable, e.g. --provider CPUExecutionProvider")
    p.add_argument("--intra-op", dest="intra_op", type=int, default=None)
    p.add_argument("--inter-op", dest="inter_op", type=int, default=None)
    args = p.parse_args(argv)

    if not os.path.exists(args.model):
        _build_tiny_model(args.model)

    providers = args.provider or ["CPUExecutionProvider"]
    stats = benchmark_session(
        args.model,
        providers=providers,
        warmup=args.warmup,
        repeats=args.repeats,
        intra_op_threads=args.intra_op,
        inter_op_threads=args.inter_op,
    )

    print("Benchmark results")
    print("-----------------")
    for k in ("providers", "n", "mean_ms", "stdev_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_rps"):
        print(f"{k:18s}: {stats[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
