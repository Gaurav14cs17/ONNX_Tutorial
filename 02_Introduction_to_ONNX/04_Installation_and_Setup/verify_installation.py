#!/usr/bin/env python3
"""Verify a local ONNX + ONNX Runtime installation.

Creates a minimal ONNX model (Y = X + B) in memory, serializes it, loads it in
ONNX Runtime, runs inference, and checks the output against NumPy reference.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def build_minimal_model() -> onnx.ModelProto:
    """Return a valid ONNX ModelProto computing Y = X + B with B constant."""
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    b_array = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b_init = numpy_helper.from_array(b_array, name="B")

    add_node = helper.make_node("Add", inputs=["X", "B"], outputs=["Y"])
    graph = helper.make_graph(
        nodes=[add_node],
        name="add_graph",
        inputs=[x_info],
        outputs=[y_info],
        initializer=[b_init],
    )
    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset], ir_version=8)
    onnx.checker.check_model(model)
    return model


def main() -> None:
    print(f"onnx version: {onnx.__version__}")
    print(f"onnxruntime version: {ort.__version__}")

    model = build_minimal_model()

    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    x = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    y = sess.run(None, {"X": x})[0]
    expected = x + np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    assert np.allclose(y, expected), f"Mismatch: got {y}, expected {expected}"
    print("inference output:", y)
    print("SUCCESS: ONNX installation verified (model build + ORT inference OK).")


if __name__ == "__main__":
    main()
