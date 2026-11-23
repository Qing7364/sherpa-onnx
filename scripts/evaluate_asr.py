#!/usr/bin/env python3
"""Paired C++ ASR evaluation on a frozen, human-reference manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata


def normalize(text, language):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    if language == "zh":
        return [c for c in text if not c.isspace()]
    if language == "en":
        return text.split()
    raise ValueError("Only zh/en reference normalization is defined")


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        row = [i]
        for j, actual in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j - 1] + (expected != actual)))
        previous = row
    return previous[-1]


def parse_cli(stdout, stderr):
    # Preserve invalid UTF-8 bytes as distinct surrogate code points. They
    # remain errors in CER/WER and can be compared exactly across runtimes.
    # Never discard them or silently convert an invalid result into empty text.
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="surrogateescape")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="surrogateescape")
    data = json.loads(stdout)
    elapsed = re.search(r"Elapsed seconds: ([0-9.]+) s", stderr)
    if not isinstance(data.get("text"), str) or not isinstance(data.get("tokens"), list) or elapsed is None:
        raise ValueError("CLI output lacks text/tokens/elapsed time")
    count = re.search(r"QWEN_BENCH_GENERATED=([0-9]+)", stderr)
    context = re.search(r"PREFILL=([0-9]+) FINAL_CONTEXT=([0-9]+) EOS=([01])", stderr)
    return dict(text=data["text"], tokens=data["tokens"],
        invalid_utf8_bytes=sum(0xDC80 <= ord(c) <= 0xDCFF for c in data["text"]),
        returned_token_count=len(data["tokens"]),
        generated_token_count=int(count[1]) if count else None,
        prefill_length=int(context[1]) if context else None,
        final_context_length=int(context[2]) if context else None,
        ended_with_eos=bool(int(context[3])) if context else None,
        elapsed_seconds=float(elapsed[1]),
        truncated=("Result is truncated." in stderr or "Truncating audio placeholders" in stderr
                   or bool(context and context[2] == "512" and context[3] == "0")),
        potential_repetition=("decode collapsed into a repetition loop" in stderr
            or (len(data["tokens"]) >= 32 and len(set(data["tokens"][-32:])) <= 3)))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "manifest", "baseline-dir", "candidate-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--decoder-name", choices=("decoder.onnx", "decoder.int8.onnx"), default="decoder.onnx")
    parser.add_argument("--encoder-name", choices=("encoder.onnx", "encoder.int8.onnx"), default="encoder.onnx")
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=4)
    parser.add_argument("--require-identical", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = args.tokenizer or args.baseline_dir / "tokenizer"
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    identities = [(row["language"], row["id"]) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Duplicate sample identity")
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key != "resume"}
    configuration.update(manifest_sha256=digest(args.manifest), binary_sha256=digest(args.binary),
        baseline_binary_sha256=digest(args.baseline_binary or args.binary),
        model_sha256={str(path.resolve()): digest(path) for path in sorted({
            args.baseline_dir / "conv_frontend.onnx", args.baseline_dir / args.encoder_name,
            args.baseline_dir / args.decoder_name, args.candidate_dir / args.decoder_name})},
        normalization="NFKC, casefold, remove Unicode P; zh remove whitespace, en split whitespace",
        timing="CLI first recognition: after recognizer creation, includes file read/AcceptWaveform/decode")
    config_path = args.output / "run-config.json"
    results_path = args.output / "samples.jsonl"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != configuration:
            raise ValueError("Existing run: use a new directory or an identical --resume configuration")
    else:
        config_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    completed = {}
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            completed[record["language"], record["id"]] = record
    for index, row in enumerate(rows):
        identity = (row["language"], row["id"])
        if identity in completed:
            continue
        if row["language"] not in ("zh", "en") or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["id"]):
            raise ValueError("Invalid manifest identity")
        wav = (args.manifest.parent / row["wav"]).resolve(strict=True)
        if not wav.is_relative_to(args.manifest.parent.resolve()) or digest(wav) != row["wav_sha256"]:
            raise ValueError("Audio path/hash does not match frozen manifest")
        reference = normalize(row["reference"], row["language"])
        if not reference:
            raise ValueError("Normal-speech denominator must have a human reference")
        record = dict(id=row["id"], language=row["language"], reference=row["reference"],
            reference_length=len(reference), duration_seconds=row["duration_seconds"])
        modes = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for mode in modes:
            model = args.baseline_dir if mode == "baseline" else args.candidate_dir
            binary = (args.baseline_binary or args.binary) if mode == "baseline" else args.binary
            command = [str(binary.resolve()),
                f"--qwen3-asr-conv-frontend={(args.baseline_dir / 'conv_frontend.onnx').resolve()}",
                f"--qwen3-asr-encoder={(args.baseline_dir / args.encoder_name).resolve()}",
                f"--qwen3-asr-decoder={(model / args.decoder_name).resolve()}",
                f"--qwen3-asr-tokenizer={tokenizer.resolve()}",
                "--qwen3-asr-max-total-len=512", "--qwen3-asr-max-new-tokens=128",
                "--qwen3-asr-temperature=1e-6", "--qwen3-asr-top-p=0.8", "--qwen3-asr-seed=42",
                f"--num-threads={args.threads}", "--provider=cpu", str(wav)]
            prefix = args.output / f"{row['language']}-{row['id']}-{mode}"
            prefix.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
            result = subprocess.run(command, check=True, capture_output=True, timeout=180)
            prefix.with_suffix(".stdout").write_bytes(result.stdout)
            prefix.with_suffix(".stderr").write_bytes(result.stderr)
            record[mode] = parse_cli(result.stdout, result.stderr)
            record[mode]["errors"] = edit_distance(reference, normalize(record[mode]["text"], row["language"]))
        a, b = record["baseline"], record["candidate"]
        record["identical"] = a["text"] == b["text"] and a["tokens"] == b["tokens"]
        record["new_empty"] = bool(a["text"]) and not bool(b["text"])
        record["new_truncation"] = b["truncated"] and not a["truncated"]
        record["new_potential_repetition"] = b["potential_repetition"] and not a["potential_repetition"]
        record["new_invalid_utf8"] = b["invalid_utf8_bytes"] > 0 and a["invalid_utf8_bytes"] == 0
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        completed[identity] = record
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(rows)} paired samples", flush=True)
    records = [completed[identity] for identity in identities]
    summary = dict(samples=len(records), identical_count=sum(r["identical"] for r in records), languages={})
    for language in ("zh", "en"):
        subset = [r for r in records if r["language"] == language]
        if not subset:
            continue
        length = sum(r["reference_length"] for r in subset)
        a = sum(r["baseline"]["errors"] for r in subset)
        b = sum(r["candidate"]["errors"] for r in subset)
        summary["languages"][language] = dict(metric="CER" if language == "zh" else "WER",
            samples=len(subset), reference_length=length, baseline_errors=a, candidate_errors=b,
            baseline_percent=100 * a / length, candidate_percent=100 * b / length,
            delta_percentage_points=100 * (b - a) / length)
    summary["invalid_utf8_samples"] = {mode: [r["id"] for r in records
        if r[mode].get("invalid_utf8_bytes", 0)] for mode in ("baseline", "candidate")}
    summary["new_anomalies"] = [r["id"] for r in records if any(r.get(k, False) for k in
        ("new_empty", "new_truncation", "new_potential_repetition", "new_invalid_utf8"))]
    summary["passed"] = not summary["new_anomalies"] and all(
        x["delta_percentage_points"] <= 0.2 for x in summary["languages"].values())
    if args.require_identical:
        summary["passed"] &= summary["identical_count"] == len(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
