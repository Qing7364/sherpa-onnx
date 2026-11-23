#!/usr/bin/env python3
"""Repeat every e2e sample with any original paired slowdown above 3%."""
import json
from pathlib import Path
import subprocess

import numpy as np

from evaluate_asr import parse_cli
from run_performance_suite import power

root = Path(__file__).resolve().parents[1]
original = root / "results/performance-ac"
summary = json.loads((original / "summary.json").read_text())
source = json.loads((original / "cli-results.json").read_text())
selected = [r for r in summary["recognition"] if r["measurement"] == "e2e"
            and any(change > 3 for change in r["round_changes_percent"])]
output = root / "results/e2e-followup"
power_before = power()
output.mkdir(exist_ok=False)
(output / "selection.json").write_text(json.dumps(dict(rule="Every e2e sample with any original paired slowdown >3%; five additional alternating pairs", samples=selected), indent=2) + "\n")
records, comparisons = [], []
for sample in selected:
    commands = {mode: next(r["command"] for r in source if r["measurement"] == "e2e"
        and r["sample_id"] == sample["sample_id"] and r["dtype"] == sample["dtype"]
        and r["mode"] == mode) for mode in ("all", "last")}
    pairs = []
    for repeat in range(5):
        pair = {}
        for mode in (("all", "last") if repeat % 2 == 0 else ("last", "all")):
            before = power()
            run = subprocess.run(commands[mode], check=True, capture_output=True, timeout=180)
            name = f"{sample['dtype']}-{sample['sample_id']}-r{repeat + 1}-{mode}"
            (output / (name + ".stdout")).write_bytes(run.stdout)
            (output / (name + ".stderr")).write_bytes(run.stderr)
            row = parse_cli(run.stdout, run.stderr)
            row.update(dtype=sample["dtype"], sample_id=sample["sample_id"], mode=mode,
                repeat=repeat + 1, command=commands[mode], power_before=before, power_after=power())
            pair[mode] = row
            records.append(row)
            (output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        assert pair["all"]["text"] == pair["last"]["text"]
        assert pair["all"]["tokens"] == pair["last"]["tokens"]
        assert pair["all"]["generated_token_count"] == pair["last"]["generated_token_count"]
        pairs.append(pair)
        print(sample["sample_id"], repeat + 1, "pair complete", flush=True)
    changes = [100 * (pair["last"]["elapsed_seconds"] / pair["all"]["elapsed_seconds"] - 1) for pair in pairs]
    medians = {mode: float(np.median([pair[mode]["elapsed_seconds"] for pair in pairs])) for mode in ("all", "last")}
    comparisons.append(dict(sample_id=sample["sample_id"], dtype=sample["dtype"], length_group=sample["length_group"],
        medians=medians, round_changes_percent=changes,
        median_change_percent=100 * (medians["last"] / medians["all"] - 1),
        slowdown_rounds_over_3_percent=sum(c > 3 for c in changes), text_and_counts_identical=True,
        repeatable_regression_over_3_percent=(medians["last"] / medians["all"] > 1.03
                                             and sum(c > 3 for c in changes) >= 3)))
report = dict(power_before=power_before, power_after=power(), comparisons=comparisons)
(output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
