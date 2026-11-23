import numpy as np
import pytest

from compare_decoder_logits import compare_outputs


def outputs():
    return {"logits": np.arange(21, dtype=np.float32).reshape(1, 3, 7),
            "key_delta_0": np.ones((1, 3, 2, 4), dtype=np.float32),
            "value_delta_0": np.ones((1, 3, 2, 4), dtype=np.float32)}


def test_identical_and_last_row_pass():
    baseline = outputs()
    assert compare_outputs(baseline, baseline, 3, 1)["passed"]
    candidate = dict(baseline, logits=baseline["logits"][:, -1:])
    assert compare_outputs(baseline, candidate, 3, 1)["passed"]


def test_changed_logits_fail_fp32_but_are_recorded_for_int8():
    baseline = outputs()
    candidate = dict(baseline, logits=baseline["logits"][:, -1:] + 1)
    assert not compare_outputs(baseline, candidate, 3, 1)["passed"]
    result = compare_outputs(baseline, candidate, 3, 1, quantized=True)
    assert result["passed"]
    assert result["outputs"]["logits"]["max_abs_error"] == 1


def test_changed_or_truncated_kv_fail_even_for_int8():
    baseline = outputs()
    candidate = dict(baseline, key_delta_0=baseline["key_delta_0"] + 1)
    assert not compare_outputs(baseline, candidate, 3, 1, quantized=True)["passed"]
    candidate["key_delta_0"] = baseline["key_delta_0"][:, :1]
    with pytest.raises(ValueError):
        compare_outputs(baseline, candidate, 3, 1)


def test_missing_kv_and_nonfinite_are_rejected():
    baseline = outputs()
    candidate = dict(baseline)
    del candidate["value_delta_0"]
    with pytest.raises(ValueError):
        compare_outputs(baseline, candidate, 3, 1)
    candidate = dict(baseline, logits=np.full((1, 1, 7), np.nan, dtype=np.float32))
    assert not compare_outputs(baseline, candidate, 3, 1)["passed"]
