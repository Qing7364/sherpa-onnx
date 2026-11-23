#!/usr/bin/env python3
"""Final structural checks of the exact paired FP32 and INT8 artifacts."""
import json
from pathlib import Path

import onnx

root = Path(__file__).resolve().parents[1]
records = []
for mode in ("all", "last"):
    for dtype, filename in (("fp32", "decoder.onnx"), ("int8", "decoder.int8.onnx")):
        path = root / "exports" / mode / filename
        onnx.checker.check_model(str(path))
        model = onnx.load(path, load_external_data=False)
        assert model.ir_version == 9 and model.opset_import[0].version == 17
        assert len(model.graph.input) == 60 and len(model.graph.output) == 57
        time = model.graph.output[0].type.tensor_type.shape.dim[1]
        assert time.dim_value == 1 if mode == "last" else time.dim_param == "seq"
        for output in model.graph.output[1:]:
            assert output.type.tensor_type.shape.dim[1].dim_param == "seq"
        producer = {value: node for node in model.graph.node for value in node.output}
        projection, = [node for node in model.graph.node
            if "/lm_head/" in node.name and node.op_type in ("MatMul", "Gemm", "MatMulInteger")]
        activation = producer[projection.input[0]]
        chain = [activation.name]
        while activation.op_type in ("Cast", "Identity", "DynamicQuantizeLinear"):
            activation = producer[activation.input[0]]
            chain.append(activation.name)
        assert (activation.op_type == "Slice") == (mode == "last")
        external = set()
        for tensor in model.graph.initializer:
            for entry in tensor.external_data:
                if entry.key == "location":
                    value = (path.parent / entry.value).resolve(strict=True)
                    assert value.is_relative_to(path.parent)
                    external.add(value.name)
        records.append(dict(mode=mode, dtype=dtype, ir_version=model.ir_version,
            opset=model.opset_import[0].version, projection=projection.name,
            activation_chain=chain, activation_op=activation.op_type,
            full_kv_outputs=56, external_files=sorted(external)))
        print(mode, dtype, "checker/layout/projection passed", flush=True)
(root / "results/final-model-structure.json").write_text(json.dumps(records, indent=2) + "\n")
