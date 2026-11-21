"""12/18 のスパンから画像を除く処理（CI-AC-S1〜S3）と、評価の費用（CI-AC-S4）。AWS は呼ばない。"""

import base64
import io
import json

import pytest
from PIL import Image

from agentlab.evalcost import eval_cost
from agentlab.spans import PLACEHOLDER, assert_text_only, find_images, strip_images


def b64(fmt="JPEG", size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 30)).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def spans():
    jpg, png = b64("JPEG"), b64("PNG")
    message = [{"role": "user", "content": [{"image": {"format": "jpeg", "source": {"bytes": jpg}}},
                                             {"text": "判定して"}]}]
    return [
        {"traceId": "t1", "spanId": "s1", "attributes": {"session.id": "sess-1", "gen_ai.prompt": "判定して"}},
        # ADOT のイベントのように、本文が JSON の文字列になっていて、/ が \/ にエスケープされている
        {"traceId": "t1", "spanId": "s2",
         "body": json.dumps({"content": message}, ensure_ascii=False).replace("/", "\\/")},
        {"traceId": "t1", "spanId": "s3", "body": {"tool_result": [{"image": {"format": "png",
                                                                               "source": {"bytes": png}}}]}},
        {"traceId": "t1", "spanId": "s4", "attributes": {"hash": "A" * 64, "short": "iVBORw0KGgo"}},
    ]


def test_strip_removes_all_images_and_keeps_text():  # CI-AC-S1, S2
    src = spans()
    before = json.dumps(src)
    out, n = strip_images(src)
    assert n == 2
    assert json.dumps(src) == before  # 元は変えない
    assert find_images(out) == []
    assert out[0] == src[0] and out[3] == src[3]  # 画像の無いスパン・短い文字列はそのまま
    assert "判定して" in out[1]["body"] and PLACEHOLDER in out[1]["body"]
    assert out[2]["body"]["tool_result"][0]["image"]["source"]["bytes"] == PLACEHOLDER


def test_find_images_reports_paths_and_assert_stops():  # CI-AC-S3
    src = spans()
    left = find_images(src)
    assert "$[1].body" in left and "$[2].body.tool_result[0].image.source.bytes" in left
    with pytest.raises(ValueError, match="image-like content remains"):
        assert_text_only(src)
    out, _ = strip_images(src)
    assert assert_text_only(out) < len(json.dumps(src).encode())


def test_raw_bytes_are_removed():
    out, n = strip_images([{"x": b"\xff\xd8\xff"}])
    assert n == 1 and out[0]["x"] == PLACEHOLDER and find_images(out) == []


def test_eval_cost():  # CI-AC-S4
    assert eval_cost("builtin", 5000, 300) == pytest.approx(5000 * 2.40e-6 + 300 * 12.00e-6)
    assert eval_cost("custom", 5000, 300, judge_in_per_mtok=1.10, judge_out_per_mtok=5.50) == pytest.approx(
        0.0015 + 5000 * 1.10e-6 + 300 * 5.50e-6)
    with pytest.raises(ValueError):
        eval_cost("custom", 1, 1)  # 判定のモデルの単価が無ければ費用を決めない
    with pytest.raises(ValueError):
        eval_cost("other", 1, 1)
