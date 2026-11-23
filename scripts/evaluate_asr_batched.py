#!/usr/bin/env python3
"""Quality-only paired evaluation through the CLI's sequential multi-file loop.

Batching amortizes model loading; each stream still has batch size one and a
fresh KV cache. These elapsed times are never used as performance evidence.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from evaluate_asr import digest, edit_distance, normalize, parse_cli


def parse_batch(stdout, stderr, count):
    lines = stdout.splitlines()
    if len(lines) != count:
        raise ValueError("Batch output count differs from input count")
    results = [parse_cli(line, stderr) for line in lines]
    for result in results:
        result["batch_elapsed_seconds"] = result.pop("elapsed_seconds")
        result["elapsed_seconds"] = None
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "int8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    suffix = ".onnx" if args.dtype == "fp32" else ".int8.onnx"
    rows = [json.loads(line) for line in (root / "eval-manifest.jsonl").read_text().splitlines()]
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        if row["language"] not in ("zh", "en") or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["id"]):
            raise ValueError("Invalid sample identity")
        path = (root / row["wav"]).resolve(strict=True)
        assert path.is_relative_to(root) and digest(path) == row["wav_sha256"]
    configuration = dict(dtype=args.dtype, binary=str(binary), binary_sha256=digest(binary),
        manifest_sha256=digest(root / "eval-manifest.jsonl"), batch_size=10,
        model_manifest_sha256=digest(root / "manifest.json"),
        semantics="Existing C++ DecodeStreams loops Decode separately; fresh per-stream KV. Quality only, no per-sample timing claim.")
    (output / "run-config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    records = []
    for start in range(0, len(rows), 10):
        batch = rows[start:start + 10]
        modes = ("baseline", "candidate") if (start // 10) % 2 == 0 else ("candidate", "baseline")
        results = {}
        for mode in modes:
            model_mode = "all" if mode == "baseline" else "last"
            command = [str(binary),
                f"--qwen3-asr-conv-frontend={root / 'exports/all/conv_frontend.onnx'}",
                f"--qwen3-asr-encoder={root / 'exports/all' / ('encoder' + suffix)}",
                f"--qwen3-asr-decoder={root / 'exports' / model_mode / ('decoder' + suffix)}",
                f"--qwen3-asr-tokenizer={root / 'checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0'}",
                "--qwen3-asr-max-total-len=512", "--qwen3-asr-max-new-tokens=128",
                "--qwen3-asr-temperature=1e-6", "--qwen3-asr-top-p=0.8",
                "--qwen3-asr-seed=42", "--num-threads=4", "--provider=cpu"]
            wavs = [str(root / row["wav"]) for row in batch]
            prefix = output / f"batch-{start:03d}-{mode}"
            prefix.with_suffix(".command.json").write_text(json.dumps(command + wavs, indent=2) + "\n")
            run = subprocess.run(command + wavs, capture_output=True, check=True, timeout=600)
            prefix.with_suffix(".stdout").write_bytes(run.stdout)
            prefix.with_suffix(".stderr").write_bytes(run.stderr)
            parsed = parse_batch(run.stdout, run.stderr, len(batch))
            # Warnings are process-wide. Re-run affected batches individually
            # to attribute truncation/repetition to the correct sample.
            if any(r["truncated"] or r["potential_repetition"] for r in parsed):
                parsed = []
                for row, wav in zip(batch, wavs):
                    single = subprocess.run(command + [wav], capture_output=True, check=True, timeout=180)
                    stem = output / f"diagnostic-{row['id']}-{mode}"
                    stem.with_suffix(".stdout").write_bytes(single.stdout)
                    stem.with_suffix(".stderr").write_bytes(single.stderr)
                    parsed.append(parse_cli(single.stdout, single.stderr))
            results[mode] = parsed
        for index, row in enumerate(batch):
            reference = normalize(row["reference"], row["language"])
            assert reference
            record = dict(id=row["id"], language=row["language"], reference=row["reference"],
                reference_length=len(reference), duration_seconds=row["duration_seconds"], batch_start=start)
            for mode in ("baseline", "candidate"):
                record[mode] = results[mode][index]
                record[mode]["errors"] = edit_distance(reference, normalize(record[mode]["text"], row["language"]))
            a, b = record["baseline"], record["candidate"]
            record.update(identical=a["text"] == b["text"] and a["tokens"] == b["tokens"],
                new_empty=bool(a["text"]) and not bool(b["text"]),
                new_truncation=b["truncated"] and not a["truncated"],
                new_potential_repetition=b["potential_repetition"] and not a["potential_repetition"],
                new_invalid_utf8=b["invalid_utf8_bytes"] > 0 and a["invalid_utf8_bytes"] == 0)
            records.append(record)
            with (output / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(args.dtype, start + len(batch), "/", len(rows), "paired samples", flush=True)
    summary = dict(samples=len(records), identical_count=sum(r["identical"] for r in records), languages={})
    for language in ("zh", "en"):
        subset = [r for r in records if r["language"] == language]
        length = sum(r["reference_length"] for r in subset)
        a = sum(r["baseline"]["errors"] for r in subset)
        b = sum(r["candidate"]["errors"] for r in subset)
        summary["languages"][language] = dict(metric="CER" if language == "zh" else "WER",
            samples=len(subset), reference_length=length, baseline_errors=a, candidate_errors=b,
            baseline_percent=100 * a / length, candidate_percent=100 * b / length,
            delta_percentage_points=100 * (b - a) / length)
    summary["new_anomalies"] = [r["id"] for r in records if any(r[k] for k in
        ("new_empty", "new_truncation", "new_potential_repetition", "new_invalid_utf8"))]
    summary["invalid_utf8_samples"] = {mode: [r["id"] for r in records if r[mode]["invalid_utf8_bytes"]]
        for mode in ("baseline", "candidate")}
    summary["passed"] = not summary["new_anomalies"] and all(v["delta_percentage_points"] <= 0.2 for v in summary["languages"].values())
    if args.dtype == "fp32":
        summary["passed"] &= summary["identical_count"] == len(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
