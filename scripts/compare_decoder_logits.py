#!/usr/bin/env python3
"""Compare fixed decoder inputs, full KV, and CPU session.Run timings."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort


def compare_outputs(baseline, candidate, seq, layers, quantized=False):
    names = {"logits"} | {f"{kind}_delta_{i}" for i in range(layers)
                           for kind in ("key", "value")}
    if set(baseline) != names or set(candidate) != names:
        raise ValueError("Missing or unexpected decoder outputs")
    metrics = {}
    passed = True
    for name in sorted(names):
        a, b = baseline[name], candidate[name]
        if name == "logits":
            if (a.ndim != 3 or b.ndim != 3 or a.shape[0] != 1 or b.shape[0] != 1
                    or a.shape[1] not in (1, seq) or b.shape[1] not in (1, seq)
                    or a.shape[2] <= 0 or a.shape[2] != b.shape[2]):
                raise ValueError("Invalid logits shape")
            a, b = a[:, -1:, :], b[:, -1:, :]
            atol = rtol = 1e-4
        else:
            if a.ndim != 4 or a.shape != b.shape or a.shape[0] != 1 or a.shape[1] != seq:
                raise ValueError("Invalid or truncated KV delta")
            atol = rtol = 1e-5
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        close = finite and bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64)) if finite else None
        metrics[name] = dict(finite=finite, within_tolerance=close,
            max_abs_error=float(difference.max()) if finite else None,
            mean_abs_error=float(difference.mean()) if finite else None)
        passed &= finite and (close or (quantized and name == "logits"))
    finite_logits = metrics["logits"]["finite"]
    a_top = int(np.argmax(baseline["logits"][:, -1:, :])) if finite_logits else None
    b_top = int(np.argmax(candidate["logits"][:, -1:, :])) if finite_logits else None
    return dict(passed=bool(passed), outputs=metrics, baseline_top1=a_top,
                candidate_top1=b_top, top1_equal=finite_logits and a_top == b_top)


def options(threads, profile_prefix=None):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = False
    if profile_prefix:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(profile_prefix)
    return opts


def worker(args):
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.worker_model),
        sess_options=options(args.threads, destination / "ort-profile" if args.profile else None),
        providers=["CPUExecutionProvider"])
    input_names = [x.name for x in session.get_inputs()]
    output_names = [x.name for x in session.get_outputs()]
    with np.load(args.case, allow_pickle=False) as data:
        if set(data.files) != set(input_names) | {"next_tokens"}:
            raise ValueError("Case inputs do not match the decoder contract")
        feed = {name: data[name] for name in input_names}
        tokens = data["next_tokens"].copy()
    seq = int(feed["input_ids"].shape[1])
    layers = len([name for name in input_names if name.startswith("cache_key_")])
    capacity = int(feed["cache_key_0"].shape[1])
    positions = feed["cache_position"]
    if (seq <= 0 or feed["input_ids"].shape != (1, seq)
            or positions.shape != (seq,) or positions[0] < 0
            or not np.array_equal(positions, np.arange(positions[0], positions[0] + seq))
            or positions[-1] >= capacity or tokens.ndim != 1):
        raise ValueError("Invalid fixed decoder case or cache capacity")
    summary = dict(case=args.case.name, sequence_length=seq, past_length=int(positions[0]),
        layers=layers, threads=args.threads, ort_version=ort.__version__,
        providers=session.get_providers(), timings_ms=[], outputs_requested=output_names,
        optimization="ORT_ENABLE_ALL", enable_mem_pattern=False,
        warmup_runs=args.warmup, measured_runs=args.runs, steps=[])
    if args.benchmark:
        for _ in range(args.warmup):
            session.run(None, feed)
        for iteration in range(args.runs):
            start = time.perf_counter_ns()
            result = session.run(None, feed)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            summary["timings_ms"].append(elapsed)
            if iteration + 1 < args.runs:
                del result
        if not all(np.isfinite(value).all() for value in result):
            raise ValueError("Non-finite benchmark output")
        summary.update(median_ms=float(np.median(summary["timings_ms"])),
            p95_ms=float(np.percentile(summary["timings_ms"], 95)),
            logits_bytes=int(result[0].nbytes), logits_shape=list(result[0].shape))
        del result
    if args.verify:
        steps = min(args.steps, len(tokens), capacity - int(positions[-1]) - 1)
        for step in range(steps + 1):
            outputs = dict(zip(output_names, session.run(None, feed)))
            current_seq = feed["input_ids"].shape[1]
            check = compare_outputs(outputs, outputs, current_seq, layers)
            if not check["passed"] or not all(np.isfinite(v).all() for v in outputs.values()):
                raise ValueError("Invalid or non-finite decoder output")
            summary["steps"].append(dict(sequence_length=current_seq,
                logits_shape=list(outputs["logits"].shape),
                logits_bytes=int(outputs["logits"].nbytes)))
            # Persist only the compared row; Run requested/materialized all rows
            # and finite checks above covered the complete logits tensor.
            outputs["logits"] = outputs["logits"][:, -1:, :].copy()
            np.savez_compressed(destination / f"step-{step}.npz", **outputs)
            pos = feed["cache_position"]
            for layer in range(layers):
                for kind in ("key", "value"):
                    feed[f"cache_{kind}_{layer}"][:, pos, :, :] = outputs[f"{kind}_delta_{layer}"]
            if step < steps:
                feed["input_ids"] = np.array([[tokens[step]]], dtype=np.int64)
                feed["attention_mask"] = np.ones((1, 1), dtype=np.int64)
                feed["cache_position"] = np.array([int(pos[-1]) + 1], dtype=np.int64)
    if args.profile:
        summary["profile"] = session.end_profiling()
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_worker(args, model, case, destination):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-model", str(model),
        "--case", str(case), "--output", str(destination), "--threads", str(args.threads),
        "--steps", str(args.steps), "--runs", str(args.runs), "--warmup", str(args.warmup)]
    for name in ("verify", "benchmark", "profile"):
        if getattr(args, name):
            command.append("--" + name)
    subprocess.run(command, check=True, timeout=1800)
    return json.loads((destination / "summary.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "case-dir", "worker-model", "case"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=4)
    parser.add_argument("--steps", type=int, choices=range(0, 513), default=16)
    parser.add_argument("--warmup", type=int, choices=range(0, 101), default=3)
    parser.add_argument("--runs", type=int, choices=range(1, 10001), default=20)
    parser.add_argument("--dtype", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--order", choices=("ab", "ba"), default="ab")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not args.verify and not args.benchmark:
        parser.error("Select --verify or --benchmark")
    if args.worker_model:
        if not args.case:
            parser.error("Worker requires --case")
        worker(args)
        return
    if not all((args.baseline, args.candidate, args.case_dir)):
        parser.error("Comparison requires --baseline, --candidate and --case-dir")
    cases = sorted(args.case_dir.glob("*.npz"))
    if not cases:
        parser.error("Case directory contains no NPZ inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(passed=True, dtype=args.dtype, order=args.order,
        baseline=str(args.baseline), candidate=str(args.candidate), cases=[])
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="decoder-compare-", dir=args.output.parent) as scratch:
            root = Path(scratch)
            summaries = {}
            entries = [("baseline", args.baseline), ("candidate", args.candidate)]
            if args.order == "ba":
                entries.reverse()
            for name, model in entries:
                summaries[name] = run_worker(args, model, case, root / name)
                if args.profile:
                    profile = Path(summaries[name]["profile"])
                    preserved = args.output.parent / (args.output.stem + "-profiles") / case.stem
                    preserved.mkdir(parents=True, exist_ok=True)
                    destination = preserved / (name + ".json")
                    shutil.copyfile(profile, destination)
                    summaries[name]["profile"] = str(destination)
            item = dict(case=case.name, **summaries, comparisons=[])
            if args.verify:
                if len(summaries["baseline"]["steps"]) != len(summaries["candidate"]["steps"]):
                    raise ValueError("Different continuation lengths")
                for step, metadata in enumerate(summaries["baseline"]["steps"]):
                    with np.load(root / "baseline" / f"step-{step}.npz", allow_pickle=False) as a, \
                            np.load(root / "candidate" / f"step-{step}.npz", allow_pickle=False) as b:
                        comparison = compare_outputs(a, b, metadata["sequence_length"],
                            summaries["baseline"]["layers"], quantized=args.dtype == "int8")
                    comparison["step"] = step
                    item["comparisons"].append(comparison)
                    report["passed"] &= comparison["passed"]
                item["first_top1_difference"] = next((c["step"] for c in item["comparisons"] if not c["top1_equal"]), None)
            if args.benchmark:
                item["median_improvement_percent"] = 100 * (1 - summaries["candidate"]["median_ms"] / summaries["baseline"]["median_ms"])
            report["cases"].append(item)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(case.name, "verified" if report["passed"] else "FAILED", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
