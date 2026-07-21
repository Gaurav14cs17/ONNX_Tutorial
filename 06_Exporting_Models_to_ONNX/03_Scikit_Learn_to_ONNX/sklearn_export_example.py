#!/usr/bin/env python3
"""
Chapter 5 — scikit-learn → ONNX using skl2onnx.

Demonstrates:
  1) LogisticRegression on synthetic tabular data.
  2) Pipeline: StandardScaler + RandomForestClassifier.
  3) onnx.checker validation + ONNX Runtime parity vs sklearn.

Requirements:
  pip install scikit-learn skl2onnx onnx onnxruntime numpy
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import onnx
from onnx import checker

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import onnxruntime as ort


def _ort_probs(sess: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """Return the most likely `predict_proba`-like matrix from an skl2onnx classifier."""
    feeds = {sess.get_inputs()[0].name: X.astype(np.float32)}
    arrs = sess.run(None, feeds)
    candidates = [a for a in arrs if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[0] == X.shape[0]]
    if not candidates:
        raise RuntimeError(f"Could not infer probability matrix from ORT outputs: {[a.shape for a in arrs]}")
    # Prefer the widest matrix (multi-class probabilities); fall back to first 2D.
    return max(candidates, key=lambda a: a.shape[1])


def example_logistic() -> None:
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        random_state=0,
    )
    X = X.astype(np.float32)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    initial_type = [("X", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)
    checker.check_model(onnx_model)

    onnx_bytes = onnx_model.SerializeToString()
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    sk_pred = model.predict_proba(X[:16])
    ort_pred = _ort_probs(sess, X[:16])

    diff = np.abs(sk_pred - ort_pred).max()
    print("\n=== LogisticRegression ===")
    print(f"max abs diff (predict_proba): {diff:.6e}")


def example_pipeline_rf() -> None:
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=10,
        random_state=1,
    )
    X = X.astype(np.float32)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=30, random_state=0)),
        ]
    )
    pipe.fit(X, y)

    initial_type = [("X", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=17)
    checker.check_model(onnx_model)

    X_test = X[:128]
    onnx_bytes = onnx_model.SerializeToString()
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    sk_proba = pipe.predict_proba(X_test)
    ort_proba = _ort_probs(sess, X_test)

    sk_cls = pipe.predict(X_test)
    ort_cls = ort_proba.argmax(axis=1)

    print("\n=== Pipeline: StandardScaler + RandomForestClassifier ===")
    print(f"max abs diff (predict_proba): {np.abs(sk_proba - ort_proba).max():.6e}")
    print(f"accuracy match rate (argmax vs predict): {np.mean(sk_cls == ort_cls):.4f}")
    print(f"sklearn accuracy on sample: {accuracy_score(y[:128], sk_cls):.4f}")


def main() -> None:
    example_logistic()
    example_pipeline_rf()
    print("\nDone. Models were checked with onnx.checker and compared against ORT in-memory.")


if __name__ == "__main__":
    main()
