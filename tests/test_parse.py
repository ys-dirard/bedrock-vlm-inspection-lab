import pytest

from vlmlab.parse import ParseSpec, parse_object, parse_response


def test_plain_object_is_parsed():
    """CI-P1, CI-P4, CI-P5"""
    r = parse_response('  {"judgement": "NG", "defect": "傷", "confidence": 0.8}\n')
    assert r.ok and r.pred == "NG" and r.confidence == 0.8 and r.defect == "傷"
    assert r.parsed == {"judgement": "NG", "defect": "傷", "confidence": 0.8}


@pytest.mark.parametrize("text", [
    'はい。{"judgement": "OK", "confidence": 0.9}',          # 前に文章
    '{"judgement": "OK", "confidence": 0.9} 以上です',       # 後ろに文章
    '[{"judgement": "OK"}]',                                  # 配列
    '{"judgement": "OK", "confidence": 0.9',                  # 壊れている
    '',
    None,
])
def test_not_whole_object_fails_without_exception(text):
    """CI-P1, CI-P2：救済せず、例外も投げない"""
    r = parse_response(text)
    assert not r.ok and r.pred is None and r.parsed is None
    assert isinstance(r.parse_error, str) and r.parse_error


def test_code_fence_fails_by_default_and_passes_only_when_allowed():
    """CI-P3"""
    text = '```json\n{"judgement": "OK", "confidence": 0.7}\n```'
    assert parse_response(text).parse_error.startswith("json:")
    r = parse_response(text, ParseSpec(allow_code_fence=True))
    assert r.ok and r.pred == "OK"


def test_code_fence_with_trailing_text_still_fails():
    """CI-P3：囲みが本文全体を包んでいないときは外さない"""
    text = '```json\n{"judgement": "OK"}\n```\n補足です'
    assert not parse_response(text, ParseSpec(allow_code_fence=True)).ok


@pytest.mark.parametrize("value", ["ng", "Ng", "NOK", "", None, 1])
def test_unexpected_judgement_value_is_schema_error(value):
    """CI-P4：大文字・小文字も区別する"""
    r = parse_response(f'{{"judgement": {__import__("json").dumps(value)}}}')
    assert not r.ok and r.parse_error.startswith("schema:")
    assert r.parsed is not None  # 読めた JSON は残す


def test_missing_judgement_key_is_schema_error():
    """CI-P4"""
    r = parse_response('{"confidence": 0.5}')
    assert r.parse_error == "schema: missing key 'judgement'"


def test_aws_sample_shape_maps_nok_to_ng():
    """CI-P4：AWS サンプルの qc（OK/NOK）"""
    spec = ParseSpec(judgement_key="qc", ok_values=("OK",), ng_values=("NOK",), confidence_key=None)
    assert parse_response('{"qc": "NOK", "reason": "x"}', spec).pred == "NG"
    assert parse_response('{"qc": "OK"}', spec).pred == "OK"
    assert not parse_response('{"qc": "NG"}', spec).ok


@pytest.mark.parametrize("c", [-0.01, 1.01, True, "0.8", None, float("nan")])
def test_invalid_confidence_is_schema_error(c):
    """CI-P5：範囲外、bool、文字列、null は失敗"""
    import json
    text = json.dumps({"judgement": "OK", "confidence": c}) if c == c else '{"judgement": "OK", "confidence": NaN}'
    r = parse_response(text)
    assert not r.ok and r.parse_error.startswith("schema:")


@pytest.mark.parametrize("c", [0, 1, 0.0, 1.0, 0.5])
def test_confidence_boundaries_are_valid(c):
    """CI-P5：0 と 1 はちょうど有効"""
    import json
    r = parse_response(json.dumps({"judgement": "NG", "confidence": c}))
    assert r.ok and r.confidence == float(c)


def test_missing_confidence_is_valid_with_none():
    """CI-P5"""
    r = parse_response('{"judgement": "OK"}')
    assert r.ok and r.confidence is None


def test_spec_from_dict_converts_lists():
    spec = ParseSpec.from_dict({"judgement_key": "qc", "ng_values": ["NOK"]})
    assert spec.ng_values == ("NOK",)
    assert parse_object('{"a": 1}') == ({"a": 1}, None)
