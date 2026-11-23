# PR review follow-up validation

This follow-up changes test coverage, a test name, and C++ continuation indentation. It does not change inference behavior. Validation ran on 2026-11-23 UTC using the existing macOS arm64 environment.

- Exporter: [7121b260693bb39d3a2981c6c57448d08dbae7d7](https://github.com/Qing7364/Qwen3-ASR-onnx/commit/7121b260693bb39d3a2981c6c57448d08dbae7d7), after `58f40e56b13c5b09ceb55a855ab8f4b09b575c41`. The four invalid-mode cases now call the real constructor without the checkpoint fixture, so they run by default. Invalid modes must be rejected before model access.
- Runtime: [2f597138fa9f23063152e103fb40667ed780abfb](https://github.com/Qing7364/sherpa-onnx/commit/2f597138fa9f23063152e103fb40667ed780abfb), after `1d4a7b76ef4b498def5f66a3682983c10b70d0f6`. The helper's declaration and definition have aligned continuation parameters. `SelectedRowHasExpectedTokenScores` names the row-access unit test accurately; its assertions do not exercise the recognizer's private EOS policy.

| Check | Result | Log |
| --- | --- | --- |
| Exporter without checkpoint | 8 passed, 5 skipped; previously 4 passed, 9 skipped | [Default pytest](exporter-default.log) |
| Exporter with Qwen3-ASR-0.6B checkpoint | 13 passed, 0 skipped, 8 existing warnings | [Checkpoint pytest](exporter-checkpoint.log) |
| C++ Qwen target rebuild | Passed, existing warnings retained | [Target build](runtime-build.log) |
| Full configured C++ build | Passed, existing warnings retained | [Full build](runtime-full-build.log) |
| Qwen C++ unit tests | 28 passed | [Qwen tests](runtime-qwen-tests.log) |
| All enabled CTest executables | 36 passed | [CTest](runtime-ctest.log) |
| cpplint on the three edited C++ files | Passed | [cpplint](cpplint.log) |
| Both repository diffs and worktrees | Diff checks passed; clean after commits | [Source and result metadata](summary.json) |

The five skipped default tests require the real checkpoint: three sequence-length cases, default/all equivalence, and paired ONNX export/load validation. The checkpoint run exported and loaded both all/last graphs with external data and checked logits/KV for S=1, 8 and 64. Python ONNX Runtime was 1.24.4 and the C++ build used ONNX Runtime 1.28.2. The pinned checkpoint revision was `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`.

The warning log retains the existing PyTorch ONNX export deprecation/tracer warnings and macOS SDK/libc++ availability and duplicate-library warnings. No warning-free build is claimed.

## Reproduction commands

Use the dependencies and CMake configuration in the [existing reproduction guide](../../reproduce.md). Run the exporter commands from its checkout at the commit above, with the pinned checkpoint already available locally. Substitute the example paths for your environment; no checkpoint is downloaded by these tests.

```sh
/path/to/venv/bin/python -m pytest tests/test_decoder_logits_mode.py -q -ra
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /path/to/venv/bin/python -m pytest \
  tests/test_decoder_logits_mode.py \
  --checkpoint /path/to/checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 -q -ra
git diff HEAD^ HEAD --check
```

Run the following from the runtime checkout at the commit above. `build-prefill` is the previously configured Release build with tests enabled.

```sh
cmake --build build-prefill --target offline-recognizer-qwen3-asr-impl-test -j 4
cmake --build build-prefill -j 4
build-prefill/bin/offline-recognizer-qwen3-asr-impl-test
ctest --test-dir build-prefill --output-on-failure
/path/to/venv/bin/cpplint \
  sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.cc \
  sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h \
  sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl-test.cc
git diff HEAD^ HEAD --check
```

The follow-up patches are [exporter](exporter-followup.patch) and [runtime](sherpa-onnx-followup.patch). Corpus accuracy, performance, RSS and synthetic full-recognizer EOS checks were not rerun for these test/formatting changes. The [original report](../../report.md) retains the earlier measurements at exporter `58f40e56b13c5b09ceb55a855ab8f4b09b575c41` and runtime `1d4a7b76ef4b498def5f66a3682983c10b70d0f6`; these follow-up commits do not alter their inference logic.
