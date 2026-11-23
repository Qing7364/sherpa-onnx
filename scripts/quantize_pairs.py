#!/usr/bin/env python3
"""Reproduce the unchanged upstream quantization on new paired FP32 exports."""
import argparse
from pathlib import Path

from export_qwen3_asr_onnx import (monkeypatch_ort_get_opset_version,
    quantize_decoder_int8, quantize_encoder_int8)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
destinations = [root / "exports" / mode / "decoder.int8.onnx" for mode in ("all", "last")]
destinations.append(root / "exports/all/encoder.int8.onnx")
if any(path.exists() for path in destinations):
    raise ValueError("Refusing to replace a frozen quantized model")
monkeypatch_ort_get_opset_version()
for mode, destination in zip(("all", "last"), destinations):
    quantize_decoder_int8(str(root / "exports" / mode / "decoder.onnx"), str(destination))
quantize_encoder_int8(str(root / "exports/all/encoder.onnx"), str(destinations[-1]))
