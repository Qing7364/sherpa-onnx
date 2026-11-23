#!/usr/bin/env python3
"""Summarize raw timing repetitions without hiding individual A/B rounds."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    grouped = {}
    for path in sorted(root.glob("prefill-*-t*-r*.json")):
        report = json.loads(path.read_text())
        for case in report["cases"]:
            key = (report["dtype"], case["baseline"]["threads"], case["baseline"]["sequence_length"])
            grouped.setdefault(key, []).append(case)
    prefill = []
    for (dtype, threads, seq), cases in sorted(grouped.items()):
        if len(cases) != 3:
            raise ValueError("Expected exactly three prefill rounds")
        values = {}
        for name in ("baseline", "candidate"):
            raw = [v for c in cases for v in c[name]["timings_ms"]]
            assert len(raw) == 60
            values[name] = dict(median_ms=float(np.median(raw)), p95_ms=float(np.percentile(raw, 95)),
                round_medians_ms=[c[name]["median_ms"] for c in cases],
                logits_bytes=cases[0][name]["logits_bytes"])
        improvements = [c["median_improvement_percent"] for c in cases]
        prefill.append(dict(dtype=dtype, threads=threads, sequence_length=seq, **values,
            median_improvement_percent=100 * (1 - values["candidate"]["median_ms"] / values["baseline"]["median_ms"]),
            round_improvements_percent=improvements,
            every_round_above_5_percent=min(improvements) >= 5))
    cli = json.loads((root / "cli-results.json").read_text())
    cli_groups = {}
    for row in cli:
        key = (row["measurement"], row["dtype"], row["language"], row["length_group"])
        cli_groups.setdefault(key, []).append(row)
    recognition = []
    for (measurement, dtype, language, length), rows in sorted(cli_groups.items()):
        metric = "elapsed_seconds" if measurement == "e2e" else "peak_rss_mib"
        modes = {mode: sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["repeat"]) for mode in ("all", "last")}
        assert all(len(r) == 3 for r in modes.values())
        values = {mode: [r[metric] for r in entries] for mode, entries in modes.items()}
        medians = {mode: float(np.median(entries)) for mode, entries in values.items()}
        regressions = [100 * (b / a - 1) for a, b in zip(values["all"], values["last"])]
        first = modes["all"][0]
        identical = all(r["text"] == first["text"] and r["tokens"] == first["tokens"]
            and r["generated_token_count"] == first["generated_token_count"] for r in rows)
        recognition.append(dict(measurement=measurement, dtype=dtype, language=language, length_group=length,
            sample_id=first["sample_id"], duration_seconds=first["duration_seconds"], metric=metric,
            raw=values, medians=medians, change_percent=100 * (medians["last"] / medians["all"] - 1),
            round_changes_percent=regressions,
            repeatable_regression_over_3_percent=(medians["last"] / medians["all"] > 1.03
                                                  and sum(x > 3 for x in regressions) >= 2),
            text_and_token_counts_identical=identical,
            generated_token_counts={mode: [r["generated_token_count"] for r in entries] for mode, entries in modes.items()},
            truncated={mode: any(r["truncated"] for r in entries) for mode, entries in modes.items()}))
    profiles = []
    for directory in sorted(root.glob("profile-*-*")):
        summary = json.loads((directory / "summary.json").read_text())
        events = json.loads((directory / Path(summary["profile"]).name).read_text())
        runs = [event for event in events if event.get("name") == "model_run"][-5:]
        if len(runs) != 5:
            raise ValueError("Expected five post-warmup profiling runs")
        kernels = [event for event in events if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")
                   and any(run["ts"] <= event["ts"] < run["ts"] + run["dur"] for run in runs)]
        projection = [event for event in kernels if "/lm_head/" in event["name"]]
        if not projection:
            raise ValueError("No named lm_head kernel in profile")
        profiles.append(dict(name=directory.name, projection_kernel_ms_per_run=sum(e["dur"] for e in projection) / 5000,
            projection_share_of_kernel_durations_percent=100 * sum(e["dur"] for e in projection) / sum(e["dur"] for e in kernels),
            kernel_names=sorted({e["name"] for e in projection}),
            input_shapes=[e.get("args", {}).get("input_type_shape") for e in projection[:4]],
            note="Five post-warmup profiler runs, excluded from formal latency series"))
    result = dict(prefill=prefill, recognition=recognition, profiles=profiles,
        all_prefill_long_rounds_above_5_percent=all(c["every_round_above_5_percent"] for c in prefill if c["threads"] == 4 and c["sequence_length"] >= 128),
        repeatable_e2e_or_rss_regressions=[r for r in recognition if r["repeatable_regression_over_3_percent"]],
        count_confounded=[r for r in recognition if not r["text_and_token_counts_identical"]])
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("all_prefill_long_rounds_above_5_percent", "repeatable_e2e_or_rss_regressions", "count_confounded")}), flush=True)


if __name__ == "__main__":
    main()
