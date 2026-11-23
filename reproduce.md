# Reproduce the experiment

This is a text-only evidence checkout. Model weights, binaries, WAVs and NPZ
inputs are not bundled. Keep this checkout immutable and use a separate, fresh
run directory. Paths under `/path/to/...` in historical logs are placeholders,
not executable commands for the current machine.

The 21.72-minute corpus and decoder inputs can be reconstructed from the pinned
public sources below. Edge-case results and the edge manifest are included,
but the original edge WAVs are not redistributed. Full synthetic EOS control
flow can be rerun with `scripts/test_full_eos_fallback.py` after model/build setup.

Use the revisions, patches and hashes in `manifest.json` and
`patches/commits.json`. Keep generated models outside source repositories. Each
export/evaluation/performance output directory below must be new; substitute
fresh names when repeating a run. Do not run the upstream `run.sh`, which clears
its output directory.

Set these variables to the actual local directories:

```bash
export QWEN3_ASR_EVIDENCE=/path/to/this-evidence-checkout
export QWEN3_ASR_ARTIFACTS=/path/to/a-new-run-directory
export QWEN3_ASR_SHERPA_ROOT=/path/to/isolated/sherpa-onnx
export QWEN3_ASR_EXPORT_ROOT=/path/to/Qwen3-ASR-onnx
export QWEN3_ASR_CHECKPOINT="$QWEN3_ASR_ARTIFACTS/checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
export QWEN3_ASR_PYTHON="$QWEN3_ASR_ARTIFACTS/venv/bin/python"
```

Create the fresh run directory and copy only the scripts and dependency pins:

```bash
mkdir "$QWEN3_ASR_ARTIFACTS"
cp -R "$QWEN3_ASR_EVIDENCE/scripts" "$QWEN3_ASR_ARTIFACTS/scripts"
cp "$QWEN3_ASR_EVIDENCE/requirements.lock.txt" "$QWEN3_ASR_ARTIFACTS/requirements.lock.txt"
export PYTHONPATH="$QWEN3_ASR_EXPORT_ROOT:$QWEN3_ASR_EXPORT_ROOT/tests:$QWEN3_ASR_ARTIFACTS/scripts"
```

Obtain `Qwen/Qwen3-ASR-0.6B` at revision
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0` in `QWEN3_ASR_CHECKPOINT`.
Use the frozen LibriSpeech archive hash and AISHELL revision in `selection.json`.
Download AISHELL's three `test-00000-of-00003.parquet` through
`test-00002-of-00003.parquet` shards from that revision, not a moving dataset head.
The official compatibility package is
`sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`; its archive hash is in `environment.md`.

The source patches are independently based on upstream sherpa master
`8a667ab114e8a007f671678ddb031729dd81dce4` and exporter main
`0829a577ee741e408e65763dfe01ff4fe75408e9`. Apply the corresponding patch in a
clean checkout of its base with `git am "$QWEN3_ASR_EVIDENCE/patches/<repository>.patch"`. For compatibility A, first build and preserve the original runtime binary at its base revision, before applying the runtime patch. Both local task branches already
contain these commits; do not apply them a second time there.

## Environment and model preparation

Use native arm64 Python 3.12.13 on this machine. The system x86_64 Python was not
used. Create an isolated environment with `uv venv --python 3.12.13 "$QWEN3_ASR_ARTIFACTS/venv"`, then install the exact
versions using `uv pip sync --python "$QWEN3_ASR_PYTHON" "$QWEN3_ASR_ARTIFACTS/requirements.lock.txt"`.
The checkpoint must be the complete pinned local safetensors revision, with
the manifest SHA256 verified before use. Pytest does not download a checkpoint.

```bash
cd "$QWEN3_ASR_EXPORT_ROOT"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$QWEN3_ASR_PYTHON" \
  export_qwen3_asr_onnx.py --model "$QWEN3_ASR_CHECKPOINT" \
  --outdir "$QWEN3_ASR_ARTIFACTS/exports/all" --device cpu \
  --max-total-len 512 --logits-mode all --no-int8 --verify
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$QWEN3_ASR_PYTHON" \
  export_qwen3_asr_onnx.py --model "$QWEN3_ASR_CHECKPOINT" \
  --outdir "$QWEN3_ASR_ARTIFACTS/exports/last" --device cpu \
  --max-total-len 512 --logits-mode last --no-int8 --verify
"$QWEN3_ASR_PYTHON" -m pytest tests/test_decoder_logits_mode.py \
  --checkpoint "$QWEN3_ASR_CHECKPOINT" -q
```

Generate INT8 decoder A/B with the unchanged exporter functions
`monkeypatch_ort_get_opset_version()` and `quantize_decoder_int8(src, dst)`;
quantize `exports/all/encoder.onnx` once with `quantize_encoder_int8`.
Run `"$QWEN3_ASR_PYTHON" "$QWEN3_ASR_ARTIFACTS/scripts/quantize_pairs.py" --root "$QWEN3_ASR_ARTIFACTS"`; the exact quantization calls are in that script. No lm_head exclusion or
quantization-setting change is used. Shared conv/encoder always come from
`exports/all`; shared tokenizer is the pinned checkpoint directory.

## C++ build and tests

Preserve the original binary before applying the runtime patch if reproducing
the old/new compatibility comparison. The original experiment stored its two binaries under `baseline-runtime/` and `candidate-runtime/`; rebuild these locally because binaries are not distributed in this snapshot.

```bash
cd "$QWEN3_ASR_SHERPA_ROOT"
cmake -S . -B build-prefill -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSHERPA_ONNX_ENABLE_TESTS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF
cmake --build build-prefill -j 4
ctest --test-dir build-prefill --output-on-failure
```

## Fixed inputs and quality evaluation

`selection.json` fixes seed 42 and sample identities before recognition.
`scripts/prepare_eval.py` reconstructs the 100 Chinese + 100 English manifest
from the pinned AISHELL test Parquet shards and OpenSLR LibriSpeech test-clean
archive. It takes `--root`, `--librispeech-archive`, and `--aishell-parquet-dir`.
Their sources/hashes are recorded in the selection and model manifests. The
published `eval-manifest.jsonl` and `normalization.json` document the frozen
evaluation set. Rebuild WAVs in the fresh run directory before copying any
manifest there; `prepare_eval.py` refuses to overwrite a frozen manifest:

```bash
"$QWEN3_ASR_PYTHON" "$QWEN3_ASR_ARTIFACTS/scripts/prepare_eval.py" \
  --root "$QWEN3_ASR_ARTIFACTS" \
  --librispeech-archive /path/to/test-clean.tar.gz \
  --aishell-parquet-dir /path/to/pinned-test-shards
```

Compare regenerated sample IDs, human references and WAV hashes with the
published manifest before running inference. Do not replace samples after
observing recognition results.

`scripts/make_decoder_inputs.py` creates fixed synthetic fixtures by importing
the tested exporter input constructor; `scripts/capture_real_inputs.py` captures
real prefill and subsequent token inputs from the original Python inference
path. These need `PYTHONPATH` pointing to the exporter (and its `tests` directory
for the synthetic constructor). JSON provenance and original input hashes are published under
`decoder-inputs/`; regenerate the actual NPZ files in the fresh run directory:

```bash
"$QWEN3_ASR_PYTHON" "$QWEN3_ASR_ARTIFACTS/scripts/make_decoder_inputs.py" \
  --checkpoint "$QWEN3_ASR_CHECKPOINT" \
  --decoder "$QWEN3_ASR_ARTIFACTS/exports/all/decoder.onnx" \
  --output "$QWEN3_ASR_ARTIFACTS/decoder-inputs"
"$QWEN3_ASR_PYTHON" "$QWEN3_ASR_ARTIFACTS/scripts/capture_real_inputs.py" \
  --root "$QWEN3_ASR_ARTIFACTS"
```

Before corpus evaluation, write `manifest.json` in the run directory with the
regenerated model hashes and the pinned source/checkpoint revisions. The
batched evaluator records this manifest's hash with the binary and WAV hashes;
the published manifest describes the original run, so it must not be mistaken
for verification of newly generated artifacts.

For example, hash the regenerated model files before evaluation:

```bash
"$QWEN3_ASR_PYTHON" - "$QWEN3_ASR_ARTIFACTS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
manifest_path = root / "manifest.json"
if manifest_path.exists():
    raise SystemExit("Keep the existing run manifest; use a fresh run directory")
files = {}
for directory in ("exports", "checkpoint"):
    for path in sorted((root / directory).rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        with path.open("rb") as stream:
            files[str(relative)] = {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            }
manifest_path.write_text(json.dumps({
    "checkpoint_revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "sherpa_base": "8a667ab114e8a007f671678ddb031729dd81dce4",
    "exporter_base": "0829a577ee741e408e65763dfe01ff4fe75408e9",
    "files": files,
}, indent=2) + "\n", encoding="utf-8")
PY
```

```bash
cd "$QWEN3_ASR_ARTIFACTS"
"$QWEN3_ASR_PYTHON" scripts/compare_decoder_logits.py \
  --baseline exports/all/decoder.onnx --candidate exports/last/decoder.onnx \
  --case-dir decoder-inputs --threads 4 --verify --steps 16 \
  --output results/decoder-fp32-rerun.json
"$QWEN3_ASR_PYTHON" scripts/compare_decoder_logits.py \
  --baseline exports/all/decoder.int8.onnx --candidate exports/last/decoder.int8.onnx \
  --case-dir decoder-inputs --threads 4 --verify --steps 16 --dtype int8 \
  --output results/decoder-int8-rerun.json
"$QWEN3_ASR_PYTHON" scripts/evaluate_asr_batched.py \
  --root "$QWEN3_ASR_ARTIFACTS" --dtype fp32 \
  --binary "$QWEN3_ASR_SHERPA_ROOT/build-prefill/bin/sherpa-onnx-offline" \
  --output results/accuracy/B-fp32-batched
"$QWEN3_ASR_PYTHON" scripts/evaluate_asr_batched.py \
  --root "$QWEN3_ASR_ARTIFACTS" --dtype int8 \
  --binary "$QWEN3_ASR_SHERPA_ROOT/build-prefill/bin/sherpa-onnx-offline" \
  --output results/accuracy/C-int8-batched
```

For compatibility A, use `scripts/evaluate_asr.py` with `--binary` pointing to
the new CLI, `--baseline-binary` to the preserved original CLI, both model
directories pointing to the official package, `--decoder-name decoder.int8.onnx`,
`--encoder-name encoder.int8.onnx`, `--require-identical`, and the frozen manifest.
The exact original commands are retained per sample under
`results/accuracy/A-compatibility/`. Valid completed single-file B samples are
retained as additional evidence. Final B/C quality summaries use the batched
directories; batches execute separate streams sequentially and are not latency
measurements. Four 10-sample comparisons validated single-file equivalence.

## Formal performance, RSS and profiling

Build the local test harness with the exact Release flags. Its only differences
are passing an existing stream language option and logging generated-token and
context counts. The generated source copies and build commands remain in
`harness/`; they are not production source patches.

```bash
cd "$QWEN3_ASR_ARTIFACTS"
"$QWEN3_ASR_PYTHON" scripts/build_test_harness.py \
  --source "$QWEN3_ASR_SHERPA_ROOT" --output harness
"$QWEN3_ASR_PYTHON" scripts/run_performance_suite.py \
  --root "$QWEN3_ASR_ARTIFACTS" --output results/performance-ac
"$QWEN3_ASR_PYTHON" scripts/summarize_performance.py results/performance-ac
```

Connect AC power first. The driver refuses battery operation and records power
state throughout. Run it with no other model export, build or evaluation active.
It runs 3 alternating rounds, 3 warmups + 20 full-output timings per case/model,
4 and 1 threads, independent CLI first-invocation timings, separate RSS processes
(macOS `/usr/bin/time -l`, bytes), and separate ORT profiling. A repeated >3%
regression is flagged when the ratio of medians worsens by >3% and at least two
of three paired rounds also worsen by >3%. Any individual e2e round above 3%
triggers five additional alternating pairs on the same sample, retained in
`results/e2e-followup/`; the follow-up uses the same median criterion and at
least three of five pairs. Every original round remains visible.
Do not use the Python inference script's printed RTF: it excludes prefill.

Run artifact-tool tests with:

```bash
"$QWEN3_ASR_PYTHON" -m pytest scripts/test_compare_decoder_logits.py \
  scripts/test_evaluate_asr.py scripts/test_batched_evaluation.py -q
```

Source code checks, complete CTest/pytest output, raw quality results, profiling
events retained for metric recomputation and all timing repetitions are in `results/`. Historical profiler paths are resolved by basename beside each profile summary. Nothing in these steps
publishes a PR, comment or model.

For any e2e sample with an individual paired slowdown above 3%, run
`"$QWEN3_ASR_PYTHON" scripts/measure_e2e_followup.py` from the fresh run directory.
It reads that run's `results/performance-ac/` and writes five additional pairs
under `results/e2e-followup/`, refusing to overwrite an existing follow-up.
The performance driver checks AC power; it is specific to macOS/M4 Pro.

To independently recompute published timing statistics without running models,
use a disposable copy of this evidence checkout and run
`python scripts/summarize_performance.py results/performance-ac` there.
The snapshot's original and published SHA256 values are documented separately
in `publication-provenance.json` and `published-files.sha256.json`.
