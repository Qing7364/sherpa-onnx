#!/usr/bin/env python3
"""Audit all per-sample scores and separate release/re-export differences."""
import json
from pathlib import Path

from evaluate_asr import edit_distance, normalize

root = Path(__file__).resolve().parents[1]
manifest = [json.loads(line) for line in (root / "eval-manifest.jsonl").read_text().splitlines()]
expected = {row["id"]: row for row in manifest}
groups = {}
for name in ("A-compatibility", "B-fp32-batched", "C-int8-batched"):
    records = [json.loads(line) for line in (root / f"results/accuracy/{name}/samples.jsonl").read_text().splitlines()]
    assert len(records) == len(expected) == 200
    assert {r["id"] for r in records} == set(expected)
    for row in records:
        sample = expected[row["id"]]
        assert row["reference"] == sample["reference"] and row["language"] == sample["language"]
        reference = normalize(sample["reference"], sample["language"])
        assert row["reference_length"] == len(reference)
        for mode in ("baseline", "candidate"):
            assert row[mode]["errors"] == edit_distance(reference, normalize(row[mode]["text"], row["language"]))
    groups[name] = {r["id"]: r for r in records}


def safe(text):
    return text.encode("utf-8", errors="backslashreplace").decode("utf-8")


lines = ["# 逐样本差异审查", "", "固定同一人工参考与归一化，对所有 600 个配对结果重新核算编辑距离和样本身份。", ""]
differences = {}
for name, records in groups.items():
    changed = [row for row in records.values() if not row["identical"]]
    differences[name] = changed
    lines += [f"## {name}", "", f"文本/token 不同：{len(changed)}/200。", ""]
    for row in changed:
        a, b = row["baseline"], row["candidate"]
        lines += [f"### {row['id']} ({row['language']})", "", "参考：" + safe(row["reference"]),
            "", "A：" + safe(a["text"]), "", "B：" + safe(b["text"]),
            "", f"错误数 {a['errors']} → {b['errors']}；新空输出={row['new_empty']}，新截断={row['new_truncation']}，新循环候选={row['new_potential_repetition']}。", ""]
bridge = []
for identity in expected:
    old = groups["A-compatibility"][identity]
    new = groups["C-int8-batched"][identity]
    a, b = old["baseline"], new["baseline"]
    bridge.append(dict(id=identity, language=old["language"], reference=old["reference"],
        official_text=a["text"], reexported_all_text=b["text"], official_errors=a["errors"],
        reexported_all_errors=b["errors"], reference_length=old["reference_length"],
        identical=a["text"] == b["text"] and a["tokens"] == b["tokens"]))
bridge_summary = {}
for language in ("zh", "en"):
    rows = [r for r in bridge if r["language"] == language]
    denominator = sum(r["reference_length"] for r in rows)
    a = sum(r["official_errors"] for r in rows)
    b = sum(r["reexported_all_errors"] for r in rows)
    bridge_summary[language] = dict(reference_length=denominator, official_errors=a, reexported_all_errors=b,
        official_percent=100 * a / denominator, reexported_all_percent=100 * b / denominator,
        changed_samples=sum(not r["identical"] for r in rows))
lines += ["## 官方 INT8 包 → 同源重导出 all INT8", "", "此比较包含发布包与重导出流程/前端差异，不是本次 all/last 优化对照。发布包未提供可核实的原始 checkpoint revision，不能假定内部权重来源完全相同。", "",
    "| 语言 | 官方错误数/参考长度 | 重导出 all 错误数/参考长度 | 错误率 | 变化样本 |", "| --- | --- | --- | --- | --- |"]
for language, row in bridge_summary.items():
    lines.append(f"| {language} | {row['official_errors']}/{row['reference_length']} | {row['reexported_all_errors']}/{row['reference_length']} | {row['official_percent']:.4f}% → {row['reexported_all_percent']:.4f}% | {row['changed_samples']} |")
for row in bridge:
    if not row["identical"]:
        lines += ["", f"### {row['id']}", "", "参考：" + safe(row["reference"]),
            "", "官方：" + safe(row["official_text"]), "", "重导出 all：" + safe(row["reexported_all_text"])]
output = root / "results/accuracy"
(output / "differences.md").write_text("\n".join(lines) + "\n")
(output / "difference-audit.json").write_text(json.dumps(dict(checked_pairs=600,
    changed={key: [row["id"] for row in rows] for key, rows in differences.items()}, bridge_summary=bridge_summary,
    bridge_samples=bridge), indent=2, ensure_ascii=True) + "\n")
print(json.dumps(dict(checked_pairs=600, changed={key: len(rows) for key, rows in differences.items()}, bridge_summary=bridge_summary)), flush=True)
