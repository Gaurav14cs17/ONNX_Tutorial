"""
FastAPI server for ONNX model inference (educational + production-oriented skeleton).

Features
- Loads an ONNX model from MODEL_PATH (env var) at startup
- GET /health  -> readiness information
- GET /metadata -> input/output tensor metadata
- POST /infer  -> JSON payload with tensor data (flattened) + shape + dtype

Security note
- Parsing arbitrary client tensors is powerful and risky; behind an API gateway in prod.

Env vars
- MODEL_PATH: path to ONNX model (default: /models/model.onnx)
- INTRA_OP_THREADS: optional intra-op thread count for CPU EP
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import onnxruntime as ort

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
INTRA_OP_THREADS = os.environ.get("INTRA_OP_THREADS")


def _build_session(model_path: str) -> ort.InferenceSession:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if INTRA_OP_THREADS:
        so.intra_op_num_threads = int(INTRA_OP_THREADS)

    # Prefer CUDA if available; fall back to CPU for tutorials.
    providers = ort.get_available_providers()
    chosen: List[str] = []
    if "CUDAExecutionProvider" in providers:
        chosen.append("CUDAExecutionProvider")
    chosen.append("CPUExecutionProvider")

    return ort.InferenceSession(model_path, sess_options=so, providers=chosen)


SESSION: Optional[ort.InferenceSession] = None

app = FastAPI(title="ONNX Runtime FastAPI Server", version="1.0.0")


@app.on_event("startup")
def load_model() -> None:
    global SESSION
    SESSION = _build_session(MODEL_PATH)


class InferRequest(BaseModel):
    """
    Flattened numeric tensor with explicit shape and numpy dtype string.
    Example dtype: "float32", "int64"
    """

    input_name: Optional[str] = Field(
        None,
        description="If omitted, the first model input name is used.",
    )
    data: List[float] = Field(
        ...,
        description="Flattened tensor values (use /infer_bytes for large payloads in production).",
    )
    shape: List[int] = Field(..., description="Tensor shape, e.g. [1,3,224,224]")
    dtype: str = Field("float32", description='Numpy dtype string, e.g. "float32"')

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [0.0] * 6,
                    "shape": [1, 2, 3],
                    "dtype": "float32",
                }
            ]
        }
    }


def _describe_session(sess: ort.InferenceSession) -> Dict[str, Any]:
    def _typeinfo(t) -> Dict[str, Any]:
        return {"name": t.name, "type": str(t.type), "shape": list(t.shape)}

    return {
        "model_path": MODEL_PATH,
        "providers": sess.get_providers(),
        "inputs": [_typeinfo(i) for i in sess.get_inputs()],
        "outputs": [_typeinfo(o) for o in sess.get_outputs()],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    if SESSION is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", **_describe_session(SESSION)}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    if SESSION is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _describe_session(SESSION)


@app.post("/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    if SESSION is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sess = SESSION
    inputs = sess.get_inputs()
    if not inputs:
        raise HTTPException(status_code=500, detail="Model has no inputs")

    input_name = req.input_name or inputs[0].name
    try:
        dtype = np.dtype(req.dtype)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Bad dtype: {e}") from e

    try:
        arr = np.asarray(req.data, dtype=dtype)
        arr = arr.reshape(tuple(req.shape))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot build ndarray: {e}") from e

    feeds: Dict[str, Any] = {input_name: arr}
    output_names = [o.name for o in sess.get_outputs()]

    try:
        outputs = sess.run(output_names, feeds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}") from e

    result = []
    for name, out in zip(output_names, outputs):
        out_np = np.asarray(out)
        payload = {
            "name": name,
            "shape": list(out_np.shape),
            "dtype": str(out_np.dtype),
            "data": out_np.ravel().tolist(),
        }
        result.append(payload)

    return {"outputs": result}


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("serve_model:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
