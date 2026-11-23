#!/usr/bin/env python3
"""Isolated, alternating A/B CPU timings; require AC power for formal runs."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from evaluate_asr import digest, parse_cli


def power():
    status = subprocess.run(["pmset", "-g", "batt"], check=True, capture_output=True, text=True).stdout
    if "AC Power" not in status:
        raise RuntimeError("Formal performance measurements require AC power: " + status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    initial_power = power()
    output.mkdir(parents=True, exist_ok=False)
    (output / "power-start.txt").write_text(initial_power)
    (output / "power-settings.txt").write_text(subprocess.run(["pmset", "-g", "custom"], check=True, capture_output=True, text=True).stdout)
    (output / "vm-stat-start.txt").write_text(subprocess.run(["vm_stat"], check=True, capture_output=True, text=True).stdout)
    cases = output / "prefill-cases"
    cases.mkdir()
    for seq in (64, 128, 256, 448):
        source = root / f"decoder-inputs/synthetic-s{seq:03d}-past000.npz"
        (cases / source.name).hardlink_to(source)
    commands = []
    for dtype in ("fp32", "int8"):
        filename = "decoder.onnx" if dtype == "fp32" else "decoder.int8.onnx"
        for threads in (4, 1):
            for round_index, order in enumerate(("ab", "ba", "ab"), 1):
                name = f"prefill-{dtype}-t{threads}-r{round_index}"
                before = power()
                command = [sys.executable, str(root / "scripts/compare_decoder_logits.py"),
                    "--baseline", str(root / "exports/all" / filename),
                    "--candidate", str(root / "exports/last" / filename),
                    "--case-dir", str(cases), "--threads", str(threads),
                    "--benchmark", "--warmup", "3", "--runs", "20",
                    "--dtype", dtype, "--order", order, "--output", str(output / (name + ".json"))]
                print(name, flush=True)
                with (output / (name + ".log")).open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                commands.append(dict(name=name, command=command, power_before=before, power_after=power()))
                (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    rows = [json.loads(line) for line in (root / "eval-manifest.jsonl").read_text().splitlines()]
    selected = []
    for language in ("zh", "en"):
        subset = sorted((r for r in rows if r["language"] == language), key=lambda r: (r["duration_seconds"], r["id"]))
        for length, index in (("short", 0), ("medium", len(subset) // 2), ("long", -1)):
            selected.append(dict(subset[index], length_group=length))
    (output / "e2e-selection.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
    records = []
    # RSS is measured in a separate set of processes, never added to the
    # timing-only series. Each CLI process creates exactly one recognizer.
    for measurement in ("e2e", "rss"):
        samples = selected if measurement == "e2e" else [r for r in selected if r["language"] == "en" and r["length_group"] == "long"]
        for dtype in ("fp32", "int8"):
            suffix = ".onnx" if dtype == "fp32" else ".int8.onnx"
            for repeat in range(3):
                for row in samples:
                    wav = root / row["wav"]
                    assert digest(wav) == row["wav_sha256"]
                    order = ("all", "last") if repeat % 2 == 0 else ("last", "all")
                    for mode in order:
                        name = f"{measurement}-{dtype}-{row['language']}-{row['length_group']}-r{repeat + 1}-{mode}"
                        command = [str(root / "harness/candidate-sherpa-onnx-offline"),
                            f"--qwen3-asr-conv-frontend={root / 'exports/all/conv_frontend.onnx'}",
                            f"--qwen3-asr-encoder={root / 'exports/all' / ('encoder' + suffix)}",
                            f"--qwen3-asr-decoder={root / 'exports' / mode / ('decoder' + suffix)}",
                            f"--qwen3-asr-tokenizer={root / 'checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0'}",
                            "--qwen3-asr-max-total-len=512", "--qwen3-asr-max-new-tokens=128",
                            "--qwen3-asr-temperature=1e-6", "--qwen3-asr-top-p=0.8",
                            "--qwen3-asr-seed=42", "--num-threads=4", "--provider=cpu", str(wav)]
                        if measurement == "rss":
                            command = ["/usr/bin/time", "-l", *command]
                        before = power()
                        result = subprocess.run(command, check=True, capture_output=True, timeout=180)
                        (output / (name + ".stdout")).write_bytes(result.stdout)
                        (output / (name + ".stderr")).write_bytes(result.stderr)
                        values = parse_cli(result.stdout, result.stderr)
                        if values["generated_token_count"] is None:
                            raise ValueError("Missing instrumented generated-token count")
                        if measurement == "rss":
                            match = re.search(rb"([0-9]+)\s+maximum resident set size", result.stderr)
                            if match is None:
                                raise ValueError("macOS time did not report RSS")
                            values.update(peak_rss_bytes=int(match[1]), peak_rss_mib=int(match[1]) / 2**20)
                        values.update(measurement=measurement, dtype=dtype, mode=mode, repeat=repeat + 1,
                            language=row["language"], sample_id=row["id"], length_group=row["length_group"],
                            duration_seconds=row["duration_seconds"], rtf=values["elapsed_seconds"] / row["duration_seconds"],
                            command=command, power_before=before, power_after=power())
                        records.append(values)
                        (output / "cli-results.json").write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n")
                    print(measurement, dtype, row["language"], row["length_group"], repeat + 1, flush=True)
    # Separate profiler runs; their timings never enter the formal summaries.
    for dtype in ("fp32", "int8"):
        filename = "decoder.onnx" if dtype == "fp32" else "decoder.int8.onnx"
        for mode in ("all", "last"):
            destination = output / f"profile-{dtype}-{mode}"
            command = [sys.executable, str(root / "scripts/compare_decoder_logits.py"),
                "--worker-model", str(root / "exports" / mode / filename),
                "--case", str(root / "decoder-inputs/synthetic-s256-past000.npz"),
                "--threads", "4", "--benchmark", "--warmup", "3", "--runs", "5",
                "--profile", "--output", str(destination)]
            print("profiling", dtype, mode, flush=True)
            subprocess.run(command, check=True)
            commands.append(dict(name=destination.name, command=command))
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (output / "power-end.txt").write_text(power())
    (output / "vm-stat-end.txt").write_text(subprocess.run(["vm_stat"], check=True, capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
