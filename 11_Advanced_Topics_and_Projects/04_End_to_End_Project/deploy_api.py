#!/usr/bin/env python3
"""
FastAPI deployment for the CIFAR-10 ONNX classifier.

Endpoints
  GET  /health
  POST /predict        JSON: { "data": [...], "shape": [1,3,32,32] }
  POST /predict_file    multipart image (RGB) -> resized to 32x32 internally

Environment
  MODEL_PATH  Path to ONNX file (default: artifacts/model.onnx)

Run
  uvicorn deploy_api:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install onnxruntime") from e

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install pillow") from e

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.onnx")

# Must match train_model.py normalization
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)

CIFAR_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class PredictRequest(BaseModel):
    data: List[float] = Field(..., description="Row-major CHW tensor values for batch=1")
    shape: List[int] = Field(default=[1, 3, 32, 32])


def preprocess_pil_rgb32(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, :, :, :]
    return (chw - CIFAR_MEAN) / CIFAR_STD


def build_session(path: str) -> ort.InferenceSession:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


SESSION: Optional[ort.InferenceSession] = None

app = FastAPI(title="CIFAR-10 ONNX API", version="1.0.0")


@app.on_event("startup")
def _load() -> None:
    global SESSION
    SESSION = build_session(MODEL_PATH)


@app.get("/health")
def health() -> Dict[str, Any]:
    if SESSION is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    inp = SESSION.get_inputs()[0]
    out = SESSION.get_outputs()[0]
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "providers": SESSION.get_providers(),
        "input": {"name": inp.name, "type": str(inp.type), "shape": list(inp.shape)},
        "output": {"name": out.name, "type": str(out.type), "shape": list(out.shape)},
    }


def _predict_array(x: np.ndarray) -> Dict[str, Any]:
    if SESSION is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    if x.dtype != np.float32:
        x = x.astype(np.float32)

    inp_name = SESSION.get_inputs()[0].name
    out_name = SESSION.get_outputs()[0].name
    logits = SESSION.run([out_name], {inp_name: x})[0]
    logits = np.asarray(logits).reshape(-1)

    ex = float(np.max(logits))
    probs = np.exp(logits - ex)
    probs = probs / np.sum(probs)

    top_idx = int(np.argmax(probs))
    top5 = np.argsort(-probs)[:5]
    return {
        "top_label": CIFAR_LABELS[top_idx],
        "top_index": top_idx,
        "confidence": float(probs[top_idx]),
        "top5": [
            {"label": CIFAR_LABELS[int(i)], "prob": float(probs[int(i)])} for i in top5
        ],
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        arr = np.asarray(req.data, dtype=np.float32).reshape(tuple(req.shape))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad tensor: {e}") from e

    if arr.ndim != 4:
        raise HTTPException(status_code=400, detail="Expected 4D tensor [N,C,H,W]")

    return _predict_array(arr)


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    x = preprocess_pil_rgb32(img)
    return _predict_array(x)


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9000"))
    uvicorn.run("deploy_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
