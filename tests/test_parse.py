import json

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


# ---- AWS サンプルの形（mode="objects"）----

SAMPLE_SPEC = ParseSpec(mode="objects", judgement_key="qc", ok_values=("OK",), ng_values=("NOK",),
                        confidence_key=None, defect_key="reason", allow_code_fence=True)


def _sample(objects):
    return json.dumps({"text": "desc", "objects": objects})


@pytest.mark.parametrize("objects, pred", [
    ([], "OK"),                                                     # 欠陥なし
    ([{"name": "d1", "qc": "OK", "reason": ""}], "OK"),             # すべて OK
    ([{"name": "d1", "qc": "NOK", "reason": "scratch"}], "NG"),
    ([{"name": "d1", "qc": "OK"}, {"name": "d2", "qc": "NOK", "reason": "hole"}], "NG"),  # 1つでも NOK なら NG
])
def test_objects_mode_judgement(objects, pred):
    """CI-P6"""
    r = parse_response(_sample(objects), SAMPLE_SPEC)
    assert r.ok and r.pred == pred and r.confidence is None


def test_objects_mode_collects_reasons_of_nok():
    """CI-P6：defect は NOK の reason を '; ' でつなぐ"""
    r = parse_response(_sample([{"qc": "NOK", "reason": "a"}, {"qc": "OK", "reason": "x"},
                                {"qc": "NOK", "reason": "b"}]), SAMPLE_SPEC)
    assert r.defect == "a; b"


@pytest.mark.parametrize("payload", [
    {"text": "no objects key"},
    {"objects": {"qc": "NOK"}},                    # 配列でない
    {"objects": ["NOK"]},                          # 要素がオブジェクトでない
    {"objects": [{"name": "d1"}]},                 # qc が無い
    {"objects": [{"qc": "nok"}]},                  # 大文字・小文字も区別
    {"objects": [{"qc": "NG"}]},                   # サンプルの値ではない
])
def test_objects_mode_schema_failures(payload):
    """CI-P6, CI-P2：形が違えば失敗。読めた JSON は parsed に残す"""
    r = parse_response(json.dumps(payload), SAMPLE_SPEC)
    assert not r.ok and r.pred is None and r.parse_error.startswith("schema:")
    assert r.parsed == payload


def test_objects_mode_still_requires_whole_object():
    """CI-P1, CI-P3：objects でも救済はしない（囲みは allow_code_fence のときだけ外す）"""
    fenced = "```json\n" + _sample([]) + "\n```"
    assert parse_response(fenced, SAMPLE_SPEC).pred == "OK"
    assert not parse_response("Here is the JSON: " + _sample([]), SAMPLE_SPEC).ok


def test_unknown_mode_fails():
    r = parse_response('{"judgement": "OK"}', ParseSpec(mode="foo"))
    assert not r.ok and r.parse_error.startswith("spec:")


# ---- CI-P7：説明の文章のあとの、末尾の囲みを読む（11/26 の記事、Claude の応答の形）----

_CLAUDE_LIKE = (
    "2枚目の画像（検査対象）を詳しく確認します。\n\n"
    "**比較:**\n- 異物: あり\n\n"
    '```json\n{"judgement": "NG", "defect": "異物", "confidence": 0.87}\n```'
)


def test_trailing_fence_fails_by_default():
    """CI-P1, CI-P3, CI-P7：既定（False）では、前に文章がある応答は失敗のまま"""
    assert not parse_response(_CLAUDE_LIKE).ok
    assert not parse_response(_CLAUDE_LIKE, ParseSpec(allow_code_fence=True)).ok


def test_trailing_fence_is_read_when_allowed():
    """CI-P7：前置きの文章は捨て、末尾の囲みの中身に CI-P4・P5 を当てる"""
    r = parse_response(_CLAUDE_LIKE + "\n  ", ParseSpec(allow_trailing_code_fence=True))
    assert r.ok and r.pred == "NG" and r.confidence == 0.87 and r.defect == "異物"


@pytest.mark.parametrize("text", [
    '{"judgement": "OK", "confidence": 0.9}',                        # 囲みなし（本文全体が JSON）
    '```json\n{"judgement": "OK", "confidence": 0.9}\n```',          # 本文全体が囲み
    '```\n{"judgement": "OK", "confidence": 0.9}\n```',              # 情報文字列なし
    '説明\n```JSON\n{"judgement": "OK", "confidence": 0.9}\n```',    # 大文字の JSON
])
def test_trailing_fence_option_keeps_plain_and_whole_fence(text):
    """CI-P1, CI-P7：囲みが無い応答・本文全体の囲みも読める"""
    r = parse_response(text, ParseSpec(allow_trailing_code_fence=True))
    assert r.ok and r.pred == "OK"


def test_trailing_fence_takes_last_block():
    """CI-P7：囲みが2つあれば、末尾のものだけを読む"""
    text = ('見本の判定:\n```json\n{"judgement": "NG", "confidence": 0.1}\n```\n'
            '検査対象:\n```json\n{"judgement": "OK", "confidence": 0.6}\n```')
    r = parse_response(text, ParseSpec(allow_trailing_code_fence=True))
    assert r.ok and r.pred == "OK" and r.confidence == 0.6


@pytest.mark.parametrize("text", [
    '説明\n```json\n{"judgement": "OK"}\n```\n以上です',              # 囲みの後ろに文章
    '説明 {"judgement": "OK", "confidence": 0.9}',                   # 囲みが無い文章（探し出さない）
    '説明\n```python\n{"judgement": "OK"}\n```',                     # 情報文字列が json 以外
    '説明 ```json\n{"judgement": "OK"}\n```',                        # 開きが行の先頭にない
    '説明\n```json\n{"judgement": "OK"}```',                         # 閉じが行の先頭にない
    '説明\n```json {"judgement": "OK"}\n```',                        # 開きの行に中身がある
    '```',                                                           # 囲みだけ
])
def test_trailing_fence_malformed_fails_without_exception(text):
    """CI-P2, CI-P3, CI-P7：形が崩れていれば失敗。例外は投げない"""
    r = parse_response(text, ParseSpec(allow_trailing_code_fence=True))
    assert not r.ok and r.pred is None and r.parsed is None
    assert r.parse_error.startswith("json:")


def test_trailing_fence_content_still_schema_checked():
    """CI-P4, CI-P5, CI-P7：囲みの中身にも判定と確信度の規則を当てる"""
    bad = '説明\n```json\n{"judgement": "ok", "confidence": 0.9}\n```'
    r = parse_response(bad, ParseSpec(allow_trailing_code_fence=True))
    assert not r.ok and r.parse_error.startswith("schema:") and r.parsed is not None
    bad_c = '説明\n```json\n{"judgement": "OK", "confidence": 1.5}\n```'
    assert parse_response(bad_c, ParseSpec(allow_trailing_code_fence=True)).parse_error.startswith("schema:")
