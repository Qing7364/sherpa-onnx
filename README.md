# Qwen3-ASR prefill last-position logits: review evidence

This snapshot supports the exporter and sherpa-onnx changes for Qwen3-ASR-0.6B. It contains the report, reproducibility instructions, validation scripts, per-clip recognition records, raw timing repetitions, compiler/test logs, and source patches. The two production PRs contain only their source changes and tests.

- [Current exporter default-model coverage and verification shapes](results/default-coverage/README.md): offline default 20 passed / 5 skipped; with the checkpoint 25 passed / no skips
- [PR review follow-up validation](results/review-followup/README.md): default test coverage, C++ formatting/test naming, and fresh test logs at the follow-up commits
- [Report and measured results](report.md) (Chinese, with numerical tables)
- [Reproduction instructions](reproduce.md) and [pinned environment](environment.md)
- [Validation and benchmark scripts](scripts)
- [Per-sample accuracy audit](results/accuracy/differences.md)
- [Raw performance repetitions](results/performance-ac) and [FP32 follow-up](results/e2e-followup)
- [Original artifact/model hashes](manifest.json) and [published-file provenance](publication-provenance.json)

The decoder optimization reduces the vocabulary projection to the last input position while preserving full Transformer computation and all-layer KV deltas. The runtime change accepts that single-row output. Python ONNX Runtime 1.24.4 measured paired decoder graphs; the same patched C++ runtime with ORT 1.28.2 was used on both sides of recognition/RSS comparisons.

| Check | Observed result |
| --- | --- |
| Released INT8 model, old/new runtime | Identical text/tokens on 200/200 clips |
| Paired FP32 all/last graphs | Identical text/tokens on 200/200 clips |
| Paired INT8 all/last graphs | Identical text/tokens on 199/200 clips |
| INT8 Chinese CER | 42/1461 to 26/1461 errors; the entire change is one clip with 16 to 0 errors |
| INT8 English WER | 76/2166 errors in both modes |
| Four-thread decoder prefill, S=128/256/448 | Median latency reductions: FP32 13.8–16.5%; INT8 19.5–20.3% |
| Six fixed clips, INT8 C++ first recognition | Median latency reductions of 3.6–7.3%, excluding model initialization |
| Fixed longest English clip, median peak RSS | FP32 reduced by 12.4%; INT8 reduced by 22.4% |

The corpus has 100 AISHELL-1 test Chinese and 100 LibriSpeech test-clean English clips, totaling 21.72 minutes. The Chinese INT8 change does not establish a general accuracy improvement. FP32 short-clip measurements include slower individual runs; follow-up measurements did not show a repeatable regression. These are local M4 Pro CPU engineering results.

## Scope of the published snapshot

Model weights, ONNX graphs, binaries, audio, and NPZ decoder inputs are not redistributed. Their original hashes remain in the manifest. Obtain the pinned checkpoint and public datasets, then regenerate the inputs in a fresh run directory as described in the reproduction instructions. The edge-case manifest and observed edge results are included; the original edge audio is not included.

Absolute developer-machine paths in logs and commands are replaced with `/path/to/...` examples. Recognition text, token IDs, scores, individual latency/RSS measurements, and compiler warnings are retained. Invalid UTF-8 stdout is preserved as bytes; the JSON records use the documented decoding policy. The artifact manifest describes the original experiment, not a promise that every original artifact is present in this text snapshot.

Profiler JSON keeps every `model_run` and node `kernel_time` event and their timing, name, category, and input-shape fields. Other event fields are omitted. The retained fields are sufficient for the published summarizer to recompute the reported projection duration and fraction. Original and published hashes plus transformations are recorded in `publication-provenance.json`.

Three script changes make the snapshot usable outside the original machine: the batched evaluator requires an explicit `--binary`; dataset preparation records the frozen AISHELL revision; and the performance summarizer locates each profiler file beside its summary. No model, scoring, sampling, quantization, or measured value was changed. The artifact-tool tests and summary recomputation are checked on these published copies.

The report and measurements below retain their original source revisions. The linked review follow-up records the later test/formatting commits separately; corpus accuracy, performance, RSS and synthetic full-recognizer EOS checks were not repeated for those changes.

## Source and dataset provenance

- Exporter: upstream `Wasser1462/Qwen3-ASR-onnx`, base `0829a577ee741e408e65763dfe01ff4fe75408e9`, change `58f40e56b13c5b09ceb55a855ab8f4b09b575c41`.
- Runtime: upstream `k2-fsa/sherpa-onnx`, base `8a667ab114e8a007f671678ddb031729dd81dce4`, change `1d4a7b76ef4b498def5f66a3682983c10b70d0f6`.
- Checkpoint: `Qwen/Qwen3-ASR-0.6B`, revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.
- Chinese corpus: [AISHELL-1](https://www.openslr.org/33/), Apache-2.0; mirror `TwinkStart/AISHELL-1`, revision `2a509ef3a7a88d3234205eaeecba39fdd9d50518`.
- English corpus: [LibriSpeech test-clean](https://www.openslr.org/12/), CC-BY-4.0.

`eval-manifest.jsonl` identifies every sample and its source, license, reference transcript and WAV hash. Source patches retain their upstream notices. The C++ build succeeded with existing macOS SDK/libc++ availability and duplicate-library warnings; the full log is included. The PR template's “without errors or warnings” checkbox therefore remains unchecked.
