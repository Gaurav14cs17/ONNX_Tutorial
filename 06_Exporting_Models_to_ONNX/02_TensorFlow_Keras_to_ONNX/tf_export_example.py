#!/usr/bin/env python3
"""
Chapter 5 — TensorFlow / Keras → ONNX using tf2onnx.

Flow:
  1. Build a small Keras CNN for MNIST-style images [N, 28, 28, 1] (channels-last).
  2. Convert with tf2onnx.convert.from_keras (SavedModel path optional).
  3. Validate with onnx.checker.
  4. Compare TensorFlow inference vs ONNX Runtime (max absolute difference).

Requirements:
  pip install "tensorflow>=2.12" tf2onnx onnx onnxruntime numpy
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

import onnx
from onnx import checker

import tensorflow as tf

import tf2onnx

import onnxruntime as ort


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="small_cnn_tf")


def export_via_keras_api(model: tf.keras.Model, onnx_path: str, opset: int = 17) -> onnx.ModelProto:
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
    onnx.save(model_proto, onnx_path)
    checker.check_model(model_proto)
    return model_proto


def export_via_savedmodel_fallback(model: tf.keras.Model, onnx_path: str, opset: int = 17) -> onnx.ModelProto:
    """Alternative path: SavedModel on disk → tf2onnx (useful in CI that forbids in-memory only)."""
    with tempfile.TemporaryDirectory() as td:
        sm = os.path.join(td, "sm")
        model.save(sm, save_format="tf")
        model_proto, _ = tf2onnx.convert.from_saved_model(
            sm,
            input_signature=(tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),),
            opset=opset,
        )
        onnx.save(model_proto, onnx_path)
        checker.check_model(model_proto)
        return model_proto


def compare_tf_vs_ort(keras_model: tf.keras.Model, onnx_path: str, batch: int = 5) -> None:
    x = np.random.randn(batch, 28, 28, 1).astype(np.float32)
    y_tf = keras_model(x, training=False).numpy()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    y_ort = sess.run(None, {in_name: x})[0]

    diff = np.abs(y_tf - y_ort)
    print("\n--- Numeric comparison (TensorFlow vs ONNX Runtime) ---")
    print(f"max abs diff: {diff.max():.6e}")
    print(f"mean abs diff: {diff.mean():.6e}")


def main() -> int:
    tf.keras.utils.set_random_seed(0)

    model = build_model()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "small_cnn_tf.onnx")

        try:
            export_via_keras_api(model, onnx_path)
            print(f"Exported with tf2onnx (from_keras) to:\n  {onnx_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] from_keras failed ({exc}); retrying SavedModel path.", file=sys.stderr)
            export_via_savedmodel_fallback(model, onnx_path)
            print(f"Exported with tf2onnx (from_saved_model) to:\n  {onnx_path}")

        m = onnx.load(onnx_path)
        checker.check_model(m)
        print("onnx.checker: OK")

        compare_tf_vs_ort(model, onnx_path)

        ops = {n.op_type for n in m.graph.node}
        print("\nUnique ops in exported graph:", sorted(ops))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
