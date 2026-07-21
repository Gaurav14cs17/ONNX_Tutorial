#!/usr/bin/env python3
"""
Inspect ONNX models: IO, nodes, initializers, and operator histograms.

Usage:
  python inspect_model.py --model path/to/model.onnx
  python inspect_model.py --model path/to/model.onnx --limit-nodes 30

Requirements:
  pip install onnx numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def _elem_type_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"UNKNOWN({elem_type})"


def _fmt_attr_value(val: object) -> str:
    if isinstance(val, onnx.TensorProto):
        dt = TensorProto.DataType.Name(val.data_type) if val.data_type else "UNKNOWN"
        dims = list(val.dims)
        nelem = int(np.prod(dims)) if dims else 0
        return f"TensorProto(dtype={dt}, shape={dims}, elems={nelem})"

    s = repr(val)
    return s if len(s) <= 200 else s[:200] + "..."


def _shape_from_value_info(vi) -> str:
    tt = vi.type.tensor_type
    if not tt.shape.dim:
        return "[]"
    parts: List[str] = []
    for d in tt.shape.dim:
        if d.dim_value:
            parts.append(str(int(d.dim_value)))
        elif d.dim_param:
            parts.append(d.dim_param)
        else:
            parts.append("?")
    return "[" + ", ".join(parts) + "]"


def _initializer_summary(graph) -> Dict[str, object]:
    bytes_total = 0
    elem_total = 0
    dtypes = Counter()
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        nbytes = arr.nbytes
        bytes_total += nbytes
        elem_total += arr.size
        dtypes[arr.dtype.name] += 1
    return {
        "initializer_count": len(graph.initializer),
        "float_like_elements_est": int(elem_total),  # includes int initializers too
        "total_initializer_bytes": int(bytes_total),
        "initializer_dtype_counter": dict(dtypes),
    }


def summarize_model(model_path: str, *, limit_nodes: int | None) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = onnx.load(model_path, load_external_data=True)
    ir_vers = model.ir_version
    op_imports = [(o.domain, o.version) for o in model.opset_import]

    g = model.graph
    lines: List[str] = []
    lines.append(f"Model: {model_path}")
    lines.append(f"IR version: {ir_vers}")
    lines.append(f"Producer: {model.producer_name!r} {model.producer_version!r}")
    lines.append(f"Opset imports: {op_imports}")
    if model.doc_string:
        lines.append(f"doc_string: {model.doc_string[:200]!r}" + ("..." if len(model.doc_string) > 200 else ""))

    lines.append("")
    lines.append("Graph I/O")
    lines.append("---------")
    for vi in g.input:
        # Skip initializer-backed 'inputs' (weights appear as inputs in some ONNX versions/graphs)
        name = vi.name
        init_names = {i.name for i in g.initializer}
        if name in init_names:
            continue
        et = vi.type.tensor_type.elem_type
        lines.append(f"- input:  {name}: {_elem_type_name(et)} {_shape_from_value_info(vi)}")

    for vi in g.output:
        et = vi.type.tensor_type.elem_type
        lines.append(f"- output: {vi.name}: {_elem_type_name(et)} {_shape_from_value_info(vi)}")

    op_hist = Counter(n.op_type for n in g.node)
    lines.append("")
    lines.append("Node histogram (op_type -> count)")
    lines.append("-----------------------------------")
    for op, c in sorted(op_hist.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {op}: {c}")

    init_stats = _initializer_summary(g)
    lines.append("")
    lines.append("Initializers")
    lines.append("------------")
    for k, v in init_stats.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Nodes (detail)")
    lines.append("--------------")
    shown = 0
    for idx, node in enumerate(g.node):
        if limit_nodes is not None and shown >= limit_nodes:
            lines.append(f"... ({len(g.node) - idx} additional nodes omitted)")
            break
        attrs = ""
        if node.attribute:
            attrs = ", attrs=" + str(
                {a.name: _fmt_attr_value(onnx.helper.get_attribute_value(a)) for a in node.attribute}
            )
        lines.append(f"- [{idx}] {node.op_type} in={list(node.input)} out={list(node.output)}{attrs}")
        shown += 1

    return "\n".join(lines) + "\n"


def extract_simple_submodel_by_output_names(model: onnx.ModelProto, output_names: Iterable[str]) -> onnx.ModelProto:
    """
    Minimal educational helper: build a new model whose outputs are a subset of the original outputs.
    This does NOT automatically trim unreachable weights; it is useful when outputs are true graph outputs.
    """
    outs = set(output_names)
    missing = [o for o in outs if o not in {x.name for x in model.graph.output}]
    if missing:
        raise ValueError(f"Requested outputs not found in graph.output: {missing}")

    new_outputs = [o for o in model.graph.output if o.name in outs]
    new_graph = onnx.helper.make_graph(
        nodes=list(model.graph.node),
        name=model.graph.name + "_sub",
        inputs=list(model.graph.input),
        outputs=new_outputs,
        initializer=list(model.graph.initializer),
    )
    sub = onnx.helper.make_model(new_graph)
    sub.opset_import.extend(model.opset_import)
    sub.ir_version = model.ir_version
    onnx.checker.check_model(sub)
    return sub


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--limit-nodes", type=int, default=None)
    args = p.parse_args(argv)

    try:
        print(summarize_model(args.model, limit_nodes=args.limit_nodes))
    except Exception as e:  # noqa: BLE001 - CLI tool
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
