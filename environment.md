# Qwen3-ASR prefill last-logits experiment

Status: implementation, numerical/corpus regression and formal AC-power
performance validation completed. Evidence snapshot for PR review.

## Paths and revisions

- Sherpa worktree: `/path/to/sherpa-onnx`
- Exporter: `/path/to/Qwen3-ASR-onnx` (new independent clone)
- Branch in each: `codex/qwen3-asr-prefill-last-logits`
- Sherpa upstream master: `8a667ab114e8a007f671678ddb031729dd81dce4`
- Exporter upstream main: `0829a577ee741e408e65763dfe01ff4fe75408e9`
- Checkpoint: `Qwen/Qwen3-ASR-0.6B`, revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`; complete, safetensors size 1876091704 bytes, SHA256 `79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`.
- Original artifact hashes: manifest.json. Models and binaries are not redistributed here.

## Hardware and toolchain

Apple M4 Pro (Mac16,7), 24 GiB RAM, macOS 26.7.1 (25G241), arm64.
Apple clang 21.0.0, CMake 4.4.4, Release build, C++17.
The planned CMake options are recorded in `results/baseline/cmake-configure.log`.

Native arm64 Python 3.12.13 in `venv/`; the system default Python is x86_64 and was not used for inference/export.

| Package | Version |
| --- | --- |
| PyTorch | 2.10.0 |
| transformers | 4.57.6 |
| ONNX | 1.20.1 |
| Python ONNX Runtime | 1.24.4 |
| numpy | 2.2.6 |
| C++ ONNX Runtime | 1.28.2 (upstream static arm64 package) |
| pytest | 9.0.2 |
| cpplint | 2.0.2 |

The two ORT versions differ: paired decoder measurements must use the same Python runtime for both modes, and paired CLI measurements the same C++ runtime. Do not compare their raw timings as an optimization effect. Full Python pins: `requirements.lock.txt`; packages installed from PyPI with `uv`, without changing global environments. Exporter modules import successfully.

## Baseline evidence

- `cmake-configure.log`: configure succeeded.
- `cmake-build.log`: CLI and existing Qwen test target built successfully; upstream deployment-target/duplicate-library compiler warnings remain.
- `ctest-qwen.log`: Qwen test executable passed (three pre-existing TrimAudioFeatures cases).
- `baseline-runtime/bin/sherpa-onnx-offline`: saved original binary, SHA256 `7b41b5ca442f71b523f55d99128111f5f3fdc844d23d3ccbf573b0c1c69f510d`. Only macOS system dylibs are referenced; ORT and project libraries are static.
- Official archive SHA256 verified against GitHub release digest: `393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96`. Archive 878702423 bytes; validated extraction 1000089677 bytes.

## Timing and local instrumentation

Initial benefit screening ran on battery power and is provisional. Formal
measurements completed on AC power with existing normal power mode 0; the
benchmark driver checked and recorded this before and after each group. No
system power settings were changed. Python ORT intra_op_num_threads was 4 or 1,
inter_op_num_threads 1, ORT_ENABLE_ALL, enable_mem_pattern=False. C++ uses the
same Release CLI with --num-threads=4 for each paired measurement. Correctness
and corpus comparisons are not timing claims.

The C++ CLI's `Elapsed seconds` starts after recognizer construction and includes
WAV reading, AcceptWaveform and decoding. Each process performs its first
recognition, not a warmed service request. No TTFT is measured. A local-only
harness, built with the same Release flags, exposes a language hint through the
existing stream option and logs generated token count before scaffold removal.
Its old/new copies reproduce the original zh/en/silence smoke JSON exactly.
The harness is not included in either source patch and adds no public API.

All three 200-sample comparisons passed. Formal prefill used 3 alternating
rounds × 3 warmups + 20 measurements, with full logits/KV outputs. End-to-end
and RSS used separate independent-process series. Two noisy FP32 Chinese
first-invocation samples received five additional alternating pairs each;
outliers and original measurements remain visible in the report. No stable
median regression was observed, but individual short-clip timings are noisy.
