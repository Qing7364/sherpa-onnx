# Default model coverage and decoder verification shapes

Validated exporter commit [549e88622b8e73733deb5b040462e449b55a64d7](https://github.com/Qing7364/Qwen3-ASR-onnx/commit/549e88622b8e73733deb5b040462e449b55a64d7), following `7121b260693bb39d3a2981c6c57448d08dbae7d7`, on 2026-11-23 UTC.

The decoder comparison used by `--verify` now rejects different ONNX/PyTorch logits shapes before subtracting the arrays. This prevents NumPy broadcasting from hiding a mismatch between full and last-position logits, or mismatched batch, rank or vocabulary dimensions. The comparison still reports numerical differences without enforcing a numerical tolerance.

The existing model regression cases now run against a small, randomly initialized `Qwen3ASRThinkerForConditionalGeneration` by default: two text decoder layers, hidden size 32, vocabulary size 128, seed 42. This uses the real repository model classes and decoder wrapper, including grouped-query attention. It loads no checkpoint and makes no network requests. The same cases additionally run against the pinned Qwen3-ASR-0.6B checkpoint when `--checkpoint` is provided.

| Validation | Observed result | Evidence |
| --- | --- | --- |
| Broadcastable mismatches before adding the guard | Expected failure: 5 cases failed to raise; 2 matching-shape cases passed | [Red-phase log](shape-guard-red.log) |
| Default suite with offline flags and no checkpoint | 20 passed, 5 skipped, 8 existing export warnings; pytest reported 0.90 seconds | [Default log](exporter-default.log) |
| Full suite with offline flags and the pinned checkpoint | 25 passed, no skips, 16 existing export warnings; pytest reported 74.86 seconds | [Checkpoint log](exporter-checkpoint.log) |
| Repository diff check | Passed; source clean after commit | [Metadata](summary.json) |

The default 20 tests comprise 8 existing CLI/invalid-mode cases, 7 decoder-comparison cases, and 5 small-model cases. The 5 skipped cases repeat the model checks with pretrained weights. Model checks cover the actual `lm_head` input length, full versus last-position logits, all-layer full-length KV deltas, default/all equivalence, and paired ONNX export/load/inference at S=1, 8 and 64. Both the small-model and checkpoint export cases pass ONNX checker and exercise ONNX Runtime 1.24.4. Small-model results validate behavior, not speech-recognition quality.

The warnings are the existing TorchScript ONNX export deprecation and tensor-constant TracerWarnings. The full run now exercises two model configurations, producing 16 warnings in total.

## Reproduction

Use the pinned environment from the [reproduction guide](../../reproduce.md). From the exporter checkout at the commit above:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /path/to/venv/bin/python -m pytest tests/test_decoder_logits_mode.py -q -ra

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /path/to/venv/bin/python -m pytest tests/test_decoder_logits_mode.py \
  --checkpoint /path/to/checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0 -q -ra

git diff HEAD^ HEAD --check
```

The [follow-up patch](exporter-followup.patch) contains the guard, test fixture, regression tests and README changes. `export_decoder`, the external-data save logic and `DecoderCoreWrapper` computation are unchanged. The sherpa-onnx source remains at `2f597138fa9f23063152e103fb40667ed780abfb`; its [prior C++ verification](../review-followup/README.md) remains applicable. No C++ tests, corpus accuracy, performance or RSS measurements were rerun in this follow-up. The original measurements remain in the [report](../../report.md).
