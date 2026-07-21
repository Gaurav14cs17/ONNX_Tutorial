#!/usr/bin/env python3
"""
Quantization tutorial script (Chapter 7).

Demonstrates:
  1) Building a tiny ONNX MatMul model (synthetic).
  2) Dynamic quantization via onnxruntime.quantization.quantize_dynamic.
  3) Static quantization via onnxruntime.quantization.quantize_static + calibration.

Requirements:
  pip install numpy onnx onnxruntime

Notes:
  - API details (argument names, enums) can vary slightly by onnxruntime version.
    If something fails, consult `help(quantize_static)` in your environment.
  - For real models, replace the synthetic generator with loading numpy arrays
    or preprocessed images from disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader as _ORTCalibrationDataReader


def build_matmul_rel_onnx(path: str, opset: int = 17) -> None:
    """
    y = ReLU( X @ W + b )
    Shapes: X [N, K], W [K, M] initializer, b [M] initializer
    """
    n, k, m = 8, 32, 16

    np.random.seed(0)
    w = np.random.randn(k, m).astype(np.float32) * 0.05
    b = np.random.randn(m).astype(np.float32) * 0.01

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, k])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])

    w_init = numpy_helper.from_array(w, name="W")
    b_init = numpy_helper.from_array(b, name="b")

    w_as_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["W"],
        value=w_init,
    )
    matmul = helper.make_node("MatMul", ["X", "W"], ["tmp"])
    bias = helper.make_node("Add", ["tmp", "b"], ["prelu"])
    relu = helper.make_node("Relu", ["prelu"], ["Y"])

    graph = helper.make_graph(
        nodes=[w_as_const, matmul, bias, relu],
        name="tiny_linear_block",
        inputs=[x],
        outputs=[y],
        initializer=[b_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[ok] wrote synthetic FP32 model to {path}")


class _SyntheticCalibrationDataReader(_ORTCalibrationDataReader):
    """
    Minimal CalibrationDataReader for ORT static quantization.

    Implement `get_next()` until exhausted, then return None.
    """

    def __init__(self, input_name: str, batches: List[np.ndarray]) -> None:
        self._input_name = input_name
        self._batches = batches
        self._idx = 0

    def get_next(self) -> dict | None:  # type: ignore[override]
        if self._idx >= len(self._batches):
            return None
        x = self._batches[self._idx]
        self._idx += 1
        return {self._input_name: x}


def _try_dynamic_quant(fp32_path: str, out_path: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=fp32_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )
    print(f"[ok] dynamic quantization wrote {out_path}")


def _try_static_quant(fp32_path: str, out_path: str) -> None:
    # These imports are version-sensitive but widely available in ORT 1.16+ packages.
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    # Discover input name/shape from the ONNX model
    m = onnx.load(fp32_path)
    in_name = m.graph.input[0].name

    # Resolve shape (dims may be None/dynamic in general models; this toy is static)
    dims = []
    for d in m.graph.input[0].type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            # Synthetic toy always static; real code should handle symbols.
            raise RuntimeError(
                f"Input {in_name} has symbolic dim {d.dim_param}. "
                "Provide a concrete calibration batch for your model."
            )
        else:
            raise RuntimeError(f"Unknown dimension in input {in_name}.")

    def rand_batches(seed: int = 1) -> List[np.ndarray]:
        rng = np.random.default_rng(seed)
        batches: List[np.ndarray] = []
        for _ in range(8):
            batches.append(rng.standard_normal(dims, dtype=np.float32) * 0.25)
        return batches

    calib_reader = _SyntheticCalibrationDataReader(in_name, rand_batches())

    quantize_static(
        model_input=fp32_path,
        model_output=out_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,  # widely interoperable pattern
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"[ok] static quantization wrote {out_path}")


def _sizes(paths: list[str]) -> None:
    for p in paths:
        if os.path.exists(p):
            print(f"{p:40s} {os.path.getsize(p):8d} bytes")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Chapter 7 quantization example")
    p.add_argument("--fp32", default="toy_fp32.onnx", help="FP32 model path")
    p.add_argument("--dynamic-out", default="toy_dynamic_int8.onnx")
    p.add_argument("--static-out", default="toy_static_int8.onnx")
    p.add_argument("--skip-static", action="store_true")
    args = p.parse_args(argv)

    if not os.path.exists(args.fp32):
        build_matmul_rel_onnx(args.fp32)

    _try_dynamic_quant(args.fp32, args.dynamic_out)

    if not args.skip_static:
        try:
            _try_static_quant(args.fp32, args.static_out)
        except Exception as e:  # noqa: BLE001 - tutorial: show actionable message
            print(
                "[warn] static quantization failed; your ORT build may differ.\n",
                f"       exception: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    print("\nFile sizes:")
    paths = [args.fp32, args.dynamic_out]
    if os.path.exists(args.static_out):
        paths.append(args.static_out)
    _sizes(paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
