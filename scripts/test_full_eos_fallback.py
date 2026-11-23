#!/usr/bin/env python3
"""Exercise actual C++ first-EOS fallback with clearly synthetic decoder graphs.

Real frontend/tokenizer, synthetic logits and zero KV. This is control-flow
coverage only, never model correctness or recognition-quality evidence.
"""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

root = Path(__file__).resolve().parents[1]
directory = root / "synthetic-eos-models"
directory.mkdir(exist_ok=True)
reference = onnx.load(root / "exports/all/decoder.onnx", load_external_data=False)
checkpoint = root / "checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
vocab = json.loads((checkpoint / "vocab.json").read_text())
tokenizer_config = json.loads((checkpoint / "tokenizer_config.json").read_text())
eos, = [int(k) for k, v in tokenizer_config["added_tokens_decoder"].items()
        if v["content"] == tokenizer_config["eos_token"]]
expected, wrong = vocab["Hello"], vocab["Test"]
vocab_size = reference.graph.output[0].type.tensor_type.shape.dim[2].dim_value
kv_shape = [d.dim_value for d in reference.graph.input[4].type.tensor_type.shape.dim][2:]
good = np.full((1, 1, vocab_size), -100, dtype=np.float32)
bad = good.copy()
good[0, 0, eos], good[0, 0, expected] = 10, 9
bad[0, 0, wrong] = 10
rows = [json.loads(line) for line in (root / "eval-manifest.jsonl").read_text().splitlines()]
wav = root / rows[0]["wav"]
results = []
for mode in ("all", "all-extra", "last"):
    initializers = [numpy_helper.from_array(value, name) for name, value in (
        ("good", good), ("bad", bad), ("kv_tail", np.asarray(kv_shape, dtype=np.int64)),
        ("seq_index", np.asarray(1, dtype=np.int64)), ("zero", np.asarray(0, dtype=np.int64)),
        ("one", np.asarray(1, dtype=np.int64)), ("extra", np.asarray(2 if mode == "all-extra" else 0, dtype=np.int64)),
        ("axes", np.asarray([0, 2], dtype=np.int64)))]
    nodes = [helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node("Concat", ["ids_shape", "kv_tail"], ["kv_shape"], axis=0),
        helper.make_node("ConstantOfShape", ["kv_shape"], ["zero_kv"],
                         value=helper.make_tensor("zero_value", TensorProto.FLOAT, [1], [0]))]
    for output in reference.graph.output[1:]:
        nodes.append(helper.make_node("Identity", ["zero_kv"], [output.name]))
    if mode == "last":
        nodes.append(helper.make_node("Identity", ["good"], ["logits"]))
    else:
        nodes.extend([
            helper.make_node("Gather", ["ids_shape", "seq_index"], ["seq"]),
            # T > context_len tests the prefill compatibility contract only.
            # Existing single-token decoding still uses its final output row.
            helper.make_node("Greater", ["seq", "one"], ["is_prefill"]),
            helper.make_node("Cast", ["is_prefill"], ["prefill_int"], to=TensorProto.INT64),
            helper.make_node("Mul", ["prefill_int", "extra"], ["prefill_extra"]),
            helper.make_node("Add", ["seq", "prefill_extra"], ["time"]),
            helper.make_node("Sub", ["seq", "one"], ["correct_row"]),
            helper.make_node("Range", ["zero", "time", "one"], ["row_indices"]),
            helper.make_node("Equal", ["row_indices", "correct_row"], ["row_mask"]),
            helper.make_node("Unsqueeze", ["row_mask", "axes"], ["mask"]),
            helper.make_node("Where", ["mask", "good", "bad"], ["logits"])])
    outputs = copy.deepcopy(list(reference.graph.output))
    if mode == "last":
        outputs[0].type.tensor_type.shape.dim[1].dim_value = 1
    elif mode == "all-extra":
        outputs[0].type.tensor_type.shape.dim[1].dim_param = "seq_plus_two"
    graph = helper.make_graph(nodes, "SYNTHETIC_EOS_CONTROL_FLOW_ONLY", list(reference.graph.input), outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = directory / f"{mode}.onnx"
    onnx.checker.check_model(model)
    onnx.save_model(model, path)
    command = [str(root / "harness/candidate-sherpa-onnx-offline"),
        f"--qwen3-asr-conv-frontend={root / 'exports/all/conv_frontend.onnx'}",
        f"--qwen3-asr-encoder={root / 'exports/all/encoder.onnx'}",
        f"--qwen3-asr-decoder={path}",
        f"--qwen3-asr-tokenizer={root / 'checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0'}",
        "--qwen3-asr-max-total-len=512", "--qwen3-asr-max-new-tokens=8",
        "--qwen3-asr-temperature=1e-6", "--qwen3-asr-top-p=0.8", "--qwen3-asr-seed=42",
        "--num-threads=4", "--provider=cpu", "--debug=true", str(wav)]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    (directory / f"{mode}.stdout").write_text(run.stdout)
    (directory / f"{mode}.stderr").write_text(run.stderr)
    result = json.loads(run.stdout)
    assert "first token is EOS" in run.stderr, mode
    assert result["text"] == "Hello", (mode, result)
    results.append(dict(mode=mode, command=command, result=result, first_eos_observed=True))
    print(mode, "first EOS -> Hello -> EOS: passed", flush=True)
(directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
