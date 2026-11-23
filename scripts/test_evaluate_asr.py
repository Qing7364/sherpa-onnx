import pytest

from evaluate_asr import edit_distance, normalize, parse_cli


def test_normalization_is_fixed_for_both_languages():
    assert normalize(" 你好，Ａ！世界。 ", "zh") == list("你好a世界")
    assert normalize(" DON'T  add Numbers: 42. ", "en") == ["dont", "add", "numbers", "42"]
    assert normalize("forty two", "en") != normalize("42", "en")


@pytest.mark.parametrize("ref,hyp,expected", [
    (list("甲乙丙"), list("甲丙"), 1),
    (["one", "two"], ["one", "three"], 1),
    ([], ["extra"], 1), (["missing"], [], 1),
    (["same"], ["same"], 0),
])
def test_hand_checked_edit_counts(ref, hyp, expected):
    assert edit_distance(ref, hyp) == expected


def test_cli_requires_result_and_elapsed_time():
    result = parse_cli('{"text":"hello", "tokens":["hello"]}',
                       'Elapsed seconds: 1.250 s\n')
    assert result["text"] == "hello"
    assert result["elapsed_seconds"] == 1.25
    with pytest.raises(ValueError):
        parse_cli('{}', 'Elapsed seconds: 1 s')
    with pytest.raises(ValueError):
        parse_cli('{"text":"hello", "tokens":[]}', '')


def test_invalid_utf8_is_preserved_and_counted_as_an_error():
    result = parse_cli(b'{"text":"\x8d", "tokens":[]}',
                       b'Elapsed seconds: 1.000 s\n')
    assert result["text"] == "\udc8d"
    assert result["invalid_utf8_bytes"] == 1
    assert edit_distance(normalize("\u4e00", "zh"), normalize(result["text"], "zh")) == 1
