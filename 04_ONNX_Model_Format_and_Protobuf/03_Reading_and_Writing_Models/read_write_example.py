#!/usr/bin/env python3
"""
Demonstration: round-trip ONNX save/load, optional external data, and
load_external_data toggles. Safe to run anywhere; uses a temporary directory.

Requires: pip install onnx numpy
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, checker, helper
from onnx.external_data_helper import load_external_data_for_model


def build_toy_matmul_model() -> onnx.ModelProto:
    """Single MatMul: Y = X @ W with static weight W (initializer)."""
    batch, in_feat, out_feat = 1, 3, 2

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch, in_feat])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, out_feat])

    w_arr = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, list(w_arr.shape), w_arr.tobytes(), raw=True)

    node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"], name="matmul")

    graph = helper.make_graph(
        [node],
        name="tiny_matmul",
        inputs=[x],
        outputs=[y],
        initializer=[w_init],
    )

    model = helper.make_model(
        graph,
        producer_name="read_write_example",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    checker.check_model(model, full_check=True)
    return model


def demo_basic_roundtrip() -> None:
    model = build_toy_matmul_model()
    raw = model.SerializeToString()
    roundtrip = onnx.load_model_from_string(raw)
    checker.check_model(roundtrip, full_check=True)
    print("[basic] SerializeToString round-trip OK, bytes:", len(raw))


def demo_file_roundtrip(directory: str) -> None:
    path = os.path.join(directory, "model.onnx")
    model = build_toy_matmul_model()
    onnx.save_model(model, path)

    loaded = onnx.load_model(path)
    checker.check_model(loaded, full_check=True)

    copy_path = os.path.join(directory, "model_copy.onnx")
    onnx.save_model(loaded, copy_path)
    assert os.path.getsize(copy_path) > 0
    print("[file]  Saved and reloaded from", path)


def demo_external_data(directory: str) -> None:
    path = os.path.join(directory, "external.onnx")
    ext_name = "params.bin"

    model = build_toy_matmul_model()
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=ext_name,
        size_threshold=0,
        convert_attribute=False,
    )

    # Initializer bytes are no longer intended to sit entirely inline in protobuf.
    print("[external] Files:", sorted(os.listdir(directory)))

    without_bytes = onnx.load_model(path, load_external_data=False)
    w0 = without_bytes.graph.initializer[0]
    print(
        "[external] Loaded skeleton; weight tensor data_location =",
        w0.data_location,
        "external_data entries =",
        len(w0.external_data),
    )

    base = directory
    load_external_data_for_model(without_bytes, base)
    w0b = without_bytes.graph.initializer[0]
    print(
        "[external] After load_external_data_for_model: raw_data bytes =",
        len(w0b.raw_data) if w0b.raw_data else 0,
    )

    # Full loader path in one call
    full = onnx.load_model(path, load_external_data=True)
    checker.check_model(full, full_check=True)
    print("[external] Full load_model(load_external_data=True) OK")


def main() -> None:
    demo_basic_roundtrip()
    with tempfile.TemporaryDirectory(prefix="onnx_rw_") as tmp:
        demo_file_roundtrip(tmp)
        demo_external_data(tmp)
    print("Done.")


if __name__ == "__main__":
    main()
