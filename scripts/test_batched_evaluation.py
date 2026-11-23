import pytest

from evaluate_asr_batched import parse_batch


def test_batch_preserves_order_and_does_not_claim_individual_timing():
    rows = parse_batch(b'{"text":"one","tokens":["one"]}\n'
                       b'{"text":"\x8d","tokens":[]}\n',
                       b'Elapsed seconds: 2.000 s\n', 2)
    assert rows[0]["text"] == "one"
    assert rows[1]["text"] == "\udc8d"
    assert rows[1]["invalid_utf8_bytes"] == 1
    assert all(r["elapsed_seconds"] is None and r["batch_elapsed_seconds"] == 2 for r in rows)


def test_missing_batch_result_fails_instead_of_shifting_sample_labels():
    with pytest.raises(ValueError, match="count"):
        parse_batch(b'{"text":"one","tokens":[]}\n', b'Elapsed seconds: 2.000 s\n', 2)
