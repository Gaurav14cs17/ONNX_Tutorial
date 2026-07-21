#!/usr/bin/env python3
"""
NLP ONNX inference with Hugging Face tokenizers + ONNX Runtime.

Typical workflow
1) Export a transformer ONNX model (e.g., with Optimum CLI) to MODEL.onnx
2) Point --model at that file and --tokenizer at the matching tokenizer id/dir

Dependencies
  pip install onnxruntime numpy transformers

Optional (for --export-torch-onnx demo)
  pip install torch optimum
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise SystemExit("Please install onnxruntime: pip install onnxruntime") from e


def build_session(model_path: str, intra_op_threads: Optional[int]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads:
        so.intra_op_num_threads = int(intra_op_threads)

    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def tokenize(text: str, tokenizer_id: str, max_length: int) -> Dict[str, np.ndarray]:
    from transformers import AutoTokenizer  # local import: clearer error if missing

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    enc = tok(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    feeds: Dict[str, np.ndarray] = {}
    for k, v in enc.items():
        arr = np.asarray(v)
        # ORT transformer graphs typically expect int64 inputs
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64)
        feeds[k] = arr
    return feeds


def pick_feeds(sess: ort.InferenceSession, candidate: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    input_names = {i.name for i in sess.get_inputs()}
    feeds: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for name in sorted(input_names):
        if name not in candidate:
            missing.append(name)
            continue
        feeds[name] = candidate[name]
    if missing:
        raise ValueError(
            "Model requires inputs not produced by tokenizer mapping: "
            + ", ".join(missing)
            + ". Provide manual feeds or a different tokenizer/export."
        )
    return feeds


def run_inference(sess: ort.InferenceSession, feeds: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    outputs = [(o.name, o) for o in sess.get_outputs()]
    out_names = [n for n, _ in outputs]
    outs = sess.run(out_names, feeds)
    return list(zip(out_names, outs))


def summarize_outputs(results: List[Tuple[str, np.ndarray]], topk: int = 5) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, tensor in results:
        t = np.asarray(tensor)
        entry: Dict[str, Any] = {"shape": list(t.shape), "dtype": str(t.dtype)}
        if t.size > 0 and t.ndim >= 2 and np.issubdtype(t.dtype, np.floating):
            flat = t.reshape(t.shape[0], -1)[0]
            if flat.size <= 64:
                entry["first_row_preview"] = flat.tolist()
            else:
                k = min(topk, flat.size)
                idx = np.argsort(-flat)[:k]
                entry["topk_indices"] = idx.tolist()
                entry["topk_values"] = flat[idx].tolist()
        else:
            entry["preview"] = str(t).replace("\n", " ")[:200]
        summary[name] = entry
    return summary


def export_demo_onnx(out_path: str, tokenizer_id: str, opset: int) -> None:
    """
    Demonstration exporter: DistilBERT feature-extraction traced to ONNX.
    Requires torch + transformers + (optionally) optimum installed.
    """
    import torch
    from transformers import DistilBertModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    model = DistilBertModel.from_pretrained(tokenizer_id)
    model.eval()

    text = "Export demonstration sentence for ONNX."
    enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    class Wrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state

    wrapped = Wrapper(model)
    wrapped.eval()

    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Wrote demo ONNX to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Runtime NLP inference demo")
    parser.add_argument("--model", type=str, help="Path to model.onnx")
    parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased")
    parser.add_argument("--text", type=str, default="ONNX Runtime executes transformers portably.")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--intra-op-threads", type=int, default=None)
    parser.add_argument("--export-torch-onnx", type=str, default=None, help="Path to write a demo DistilBERT ONNX via torch.export")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.export_torch_onnx:
        export_demo_onnx(args.export_torch_onnx, args.tokenizer, args.opset)
        args.model = args.model or args.export_torch_onnx

    if not args.model:
        raise SystemExit("Provide --model PATH or use --export-torch-onnx PATH to generate a demo model.")

    sess = build_session(args.model, args.intra_op_threads)
    feeds_token = tokenize(args.text, args.tokenizer, args.max_length)
    feeds = pick_feeds(sess, feeds_token)

    results = run_inference(sess, feeds)
    summary = summarize_outputs(results)
    print(json.dumps({"providers": sess.get_providers(), "outputs": summary}, indent=2))


if __name__ == "__main__":
    main()
