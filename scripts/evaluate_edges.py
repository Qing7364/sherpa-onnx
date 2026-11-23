#!/usr/bin/env python3
"""Run explicitly separated edge cases; synthetic audio has no CER denominator."""
import json
from pathlib import Path
import subprocess

from evaluate_asr import digest, parse_cli

root = Path(__file__).resolve().parents[1]
official = root / "models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
checkpoint = root / "checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
configurations = [
    ("A-compatibility", official, official, official / "tokenizer", "int8"),
    ("B-fp32", root / "exports/all", root / "exports/last", checkpoint, "fp32"),
    ("C-int8", root / "exports/all", root / "exports/last", checkpoint, "int8"),
]
rows = [json.loads(line) for line in (root / "edge-manifest.jsonl").read_text().splitlines()]
report = []
for comparison, baseline, candidate, tokenizer, dtype in configurations:
    output = root / "results/edges" / comparison
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".int8.onnx" if dtype == "int8" else ".onnx"
    for row in rows:
        wav = (root / row["wav"]).resolve(strict=True)
        assert wav.is_relative_to(root) and digest(wav) == row["wav_sha256"]
        record = dict(comparison=comparison, id=row["id"], options=row["options"],
                      duration_seconds=row["duration_seconds"])
        for mode, model in (("baseline", baseline), ("candidate", candidate)):
            binary_mode = "baseline" if comparison == "A-compatibility" and mode == "baseline" else "candidate"
            command = [str(root / "harness" / f"{binary_mode}-sherpa-onnx-offline"),
                f"--qwen3-asr-conv-frontend={baseline / 'conv_frontend.onnx'}",
                f"--qwen3-asr-encoder={baseline / ('encoder' + suffix)}",
                f"--qwen3-asr-decoder={model / ('decoder' + suffix)}",
                f"--qwen3-asr-tokenizer={tokenizer}", "--qwen3-asr-max-total-len=512",
                "--qwen3-asr-max-new-tokens=128", "--qwen3-asr-temperature=1e-6",
                "--qwen3-asr-top-p=0.8", "--qwen3-asr-seed=42", "--num-threads=4",
                "--provider=cpu", "--debug=true"]
            if "hotwords" in row["options"]:
                command.append("--qwen3-asr-hotwords=" + row["options"]["hotwords"])
            if "language_hint" in row["options"]:
                command.append("--bench-language=" + row["options"]["language_hint"])
            command.append(str(wav))
            prefix = output / (row["id"] + "-" + mode)
            prefix.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
            prefix.with_suffix(".stdout").write_text(result.stdout)
            prefix.with_suffix(".stderr").write_text(result.stderr)
            record[mode] = parse_cli(result.stdout, result.stderr)
            record[mode]["first_token_eos"] = "first token is EOS" in result.stderr
            record[mode]["capacity_warning"] = "Result is truncated." in result.stderr or "Truncating audio placeholders" in result.stderr
        record["identical"] = all(record["baseline"][key] == record["candidate"][key] for key in ("text", "tokens"))
        report.append(record)
        (root / "results/edges/summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(comparison, row["id"], "identical" if record["identical"] else "DIFFERENT", flush=True)
