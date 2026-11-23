#!/usr/bin/env python3
"""Freeze public human-reference samples before running recognition."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess
import tarfile
from urllib.parse import urlencode, urlsplit

import soundfile as sf
import pyarrow.parquet as pq


def fetch(url, *, params=None, limit=4 * 1024 * 1024):
    # System curl uses the machine's trusted CA store; keep TLS verification on.
    if params:
        url += "?" + urlencode(params)
    try:
        result = subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "15", "--max-time", "45", "--retry", "2",
            "--max-filesize", str(limit), url], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=150)
    except subprocess.CalledProcessError as exc:
        diagnostic = re.sub(r"https?://\S+", "[URL]", exc.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"Public dataset download failed ({exc.returncode}): {diagnostic}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("Public dataset download exceeded timeout") from None
    if len(result.stdout) > limit:
        raise ValueError("Response exceeded download size limit")
    return result.stdout


def record(root, sample_id, lang, raw_audio, reference, **source):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", sample_id):
        raise ValueError("Invalid public sample ID")
    audio, rate = sf.read(io.BytesIO(raw_audio), dtype="int16")
    if rate != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 16000 * 60:
        raise ValueError("Unexpected audio format or duration")
    path = root / "audio" / f"{lang}-{sample_id}.wav"
    if path.is_symlink():
        raise ValueError("Refusing symlink output")
    sf.write(path, audio, rate, subtype="PCM_16")
    return dict(id=sample_id, language=lang, wav=str(path.relative_to(root)),
                wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                duration_seconds=len(audio) / rate, sample_rate=rate,
                reference=reference, reference_type="human", **source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--librispeech-archive", type=Path, required=True)
    parser.add_argument("--aishell-parquet-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)
    if (root / "eval-manifest.jsonl").exists():
        raise SystemExit("Manifest already exists; keep the frozen selection")

    dataset = "TwinkStart/AISHELL-1"
    revision = "2a509ef3a7a88d3234205eaeecba39fdd9d50518"
    zh_indices = sorted(random.Random(42).sample(range(7176), 100))
    with tarfile.open(args.librispeech_archive, "r:gz") as tar:
        members = tar.getmembers()
        flacs = {Path(m.name).stem: m for m in members
                 if m.isfile() and m.name.startswith("LibriSpeech/test-clean/")
                 and m.name.endswith(".flac")}
        references = {}
        for member in members:
            if member.isfile() and member.name.endswith(".trans.txt"):
                if member.size > 1024 * 1024:
                    raise ValueError("Unexpected transcript size")
                for line in tar.extractfile(member).read().decode("utf-8").splitlines():
                    key, text = line.split(" ", 1)
                    references[key] = text
        en_ids = sorted(random.Random(42).sample(sorted(flacs), 100))
        selection = dict(seed=42, zh_dataset=dataset, zh_revision=revision,
                         zh_indices=zh_indices, en_ids=en_ids,
                         en_archive_sha256=None, en_split="test-clean",
                         note="Selection frozen before recognition; no model output used.")
        with args.librispeech_archive.open("rb") as stream:
            selection["en_archive_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        records = []
        for sample_id in en_ids:
            member = flacs[sample_id]
            if member.size > 4 * 1024 * 1024:
                raise ValueError("Unexpected FLAC size")
            records.append(record(root, sample_id, "en", tar.extractfile(member).read(),
                references[sample_id], dataset="LibriSpeech", split="test-clean",
                source="https://www.openslr.org/12/", license="CC-BY-4.0",
                archive_member=member.name))
    print("Frozen and prepared 100 LibriSpeech test-clean samples", flush=True)

    # Read the fixed test shards directly: the viewer rate-limits per-row
    # queries. Keep exactly the selection already frozen above.
    offset = 0
    for number in range(3):
        table = pq.read_table(args.aishell_parquet_dir / f"test-{number:05d}-of-00003.parquet")
        for index in zh_indices:
            if not offset <= index < offset + len(table):
                continue
            values = table.slice(index - offset, 1).to_pylist()[0]
            records.append(record(root, values["name"], "zh", values["audio"]["bytes"], values["text"],
                dataset="AISHELL-1", split="test", source="https://www.openslr.org/33/",
                mirror=dataset, source_revision=revision, row_index=index,
                original_wav_path=values["WavPath"], license="Apache-2.0"))
        offset += len(table)
        print(f"Read AISHELL-1 test shard {number + 1}/3", flush=True)
    if offset != 7176:
        raise ValueError("AISHELL row count changed")
    if len(records) != 200 or len({(x["language"], x["id"]) for x in records}) != 200:
        raise ValueError("Sample count/identity mismatch")
    (root / "eval-manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    print(json.dumps({lang: {"count": sum(r["language"] == lang for r in records),
        "duration_seconds": sum(r["duration_seconds"] for r in records if r["language"] == lang)}
        for lang in ("zh", "en")}), flush=True)


if __name__ == "__main__":
    main()
