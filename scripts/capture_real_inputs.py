#!/usr/bin/env python3
"""Capture real prefill/known continuation inputs from the original infer path."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

import infer_qwen3_asr as inference
from compare_decoder_logits import options


class CaptureDecoder:
    def __init__(self, session):
        self.session = session
        self.prefill = None
        self.tokens = []

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, outputs, feed):
        if self.prefill is None:
            self.prefill = {key: value.copy() for key, value in feed.items()}
        else:
            if feed["input_ids"].shape != (1, 1):
                raise ValueError("Expected single-token continuation")
            self.tokens.append(int(feed["input_ids"][0, 0]))
        return self.session.run(outputs, feed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint = root / "checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
    assert inference._register_qwen3_asr()
    torch.set_num_threads(4)
    tokenizer = inference._load_tokenizer(str(checkpoint))
    processor = inference._load_processor(str(checkpoint))
    _, audio_token = inference._resolve_audio_token_and_id(tokenizer, processor)
    sessions = {name: ort.InferenceSession(str(root / f"exports/all/{filename}"),
        sess_options=options(4), providers=["CPUExecutionProvider"])
        for name, filename in (("conv", "conv_frontend.onnx"), ("encoder", "encoder.onnx"), ("decoder", "decoder.onnx"))}
    manifest = [json.loads(line) for line in (root / "eval-manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for language in ("zh", "en"):
        samples = sorted((row for row in manifest if row["language"] == language), key=lambda row: row["duration_seconds"])
        sample = samples[len(samples) // 2]
        audio = root / sample["wav"]
        capture = CaptureDecoder(sessions["decoder"])
        config = SimpleNamespace(max_total_len=512, max_new_tokens=17, chunk_size=100, debug=False)
        inference._infer_one(config, tokenizer, processor, sessions["encoder"], capture,
            sessions["conv"], audio_token, [str(audio)], [inference._load_audio_any(str(audio))], [""])
        if capture.prefill is None or len(capture.tokens) < 2:
            raise ValueError("No valid real prefill with at least two continuation tokens")
        target = root / f"decoder-inputs/real-{language}.npz"
        if target.exists():
            raise ValueError("Refusing to replace frozen real inputs")
        np.savez_compressed(target, **capture.prefill, next_tokens=np.asarray(capture.tokens, dtype=np.int64))
        metadata = dict(source="Original infer_qwen3_asr._infer_one, shared all FP32 frontend",
            sample_id=sample["id"], wav_sha256=sample["wav_sha256"],
            seq=int(capture.prefill["input_ids"].shape[1]), past_len=0,
            continuation="Actual baseline greedy next-token inputs", next_tokens=capture.tokens,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            note="infer_qwen3_asr's printed RTF excludes prefill and is not a performance measurement here")
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(language, metadata["seq"], len(capture.tokens), flush=True)


if __name__ == "__main__":
    main()
