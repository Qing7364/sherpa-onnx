#!/usr/bin/env python3
"""Observe lm_head's existing dynamic activation scale without changing it."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from compare_decoder_logits import options

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("all", "last"), required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
model = onnx.load(root / "exports" / args.mode / "decoder.int8.onnx")
producer = {value: node for node in model.graph.node for value in node.output}
projection, = [node for node in model.graph.node if node.name == "/lm_head/MatMul_quant"]
quantizer = producer[projection.input[0]]
assert projection.op_type == "MatMulInteger" and quantizer.op_type == "DynamicQuantizeLinear"
scale, zero = quantizer.output[1:]
del model.graph.output[:]
model.graph.output.extend([helper.make_tensor_value_info(scale, TensorProto.FLOAT, []),
                           helper.make_tensor_value_info(zero, TensorProto.UINT8, [])])
session = ort.InferenceSession(model.SerializeToString(), sess_options=options(4), providers=["CPUExecutionProvider"])
records = []
for case in ("real-zh", "real-en", "synthetic-s448-past000", "synthetic-s511-past000"):
    with np.load(root / "decoder-inputs" / (case + ".npz"), allow_pickle=False) as data:
        feed = {meta.name: data[meta.name] for meta in session.get_inputs()}
    observed_scale, observed_zero = session.run(None, feed)
    assert np.isfinite(observed_scale) and observed_scale > 0
    records.append(dict(case=case, scale=float(observed_scale), zero_point=int(observed_zero)))
report = dict(mode=args.mode, activation_quantizer=quantizer.name, quantizer_input=list(quantizer.input), cases=records,
              note="Diagnostic output-only graph copy in memory; original weights/settings/files unchanged")
(root / "results" / f"int8-lm-head-scale-{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
