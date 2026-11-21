"""12/14：人に回す仕組み（vlmlab.review）の単体テスト。AWS は呼ばない（クライアントは偽物）。"""

from __future__ import annotations

import base64
import io
import json

import pytest
from botocore.exceptions import ClientError

from vlmlab.parse import ParseSpec
from vlmlab.review import route
from vlmlab.review.judge_handler import Judge, build_messages, image_format, judge_result, validate_event
from vlmlab.review.review_handler import ReviewApp, parse_form
from vlmlab.review.route import RoutePolicy, anomaly_score, decide, sample_unit

BAND = RoutePolicy.from_dict({"kind": "confidence_band", "lo": 0.4, "hi": 0.8})


# ---------------- route.py ----------------

def test_anomaly_score_rounding():
    """CI-HR1：NG は c、OK は 1 − c を小数9桁に丸める。判定か確信度が無ければ None。"""
    assert anomaly_score("NG", 0.7) == 0.7
    assert anomaly_score("OK", 0.3) == 0.7  # 1 - 0.3 = 0.7000000000000001 → 0.7
    assert anomaly_score("OK", None) is None
    assert anomaly_score(None, 0.5) is None


@pytest.mark.parametrize("judgement,conf,route_,auto,reason", [
    ("OK", 0.61, "auto", "OK", "below_band"),    # score 0.39
    ("OK", 0.6, "human", None, "in_band"),       # score 0.4（下端は人へ）
    ("NG", 0.79, "human", None, "in_band"),
    ("NG", 0.8, "auto", "NG", "above_band"),     # 上端は自動で NG
    ("OK", 0.2, "auto", "NG", "above_band"),     # OK でも不良らしさ 0.8 なら NG（12/12 の CI-CB7 と同じ）
    (None, None, "human", None, "parse_failed"),
    ("OK", None, "human", None, "no_confidence"),
])
def test_confidence_band(judgement, conf, route_, auto, reason):
    """CI-HR2：[lo, hi) を人へ、lo 未満は自動 OK、hi 以上は自動 NG、解析失敗・確信度なしは人へ。"""
    d = decide(BAND, "x", judgement, conf)
    assert (d.route, d.auto_judgement, d.reason) == (route_, auto, reason)


def test_confidence_band_matches_1212_definition():
    """CI-HR2：境界は丸めてから比べる（0.4 と 1 − 0.6 が同じ扱い）。"""
    assert decide(BAND, "x", "OK", 0.6).route == "human"
    assert decide(RoutePolicy.from_dict({"kind": "confidence_band", "lo": 0.5, "hi": 0.5}), "x", "NG", 0.5).route == "auto"


def test_sample_policy():
    """CI-HR3：NG と解析失敗は必ず人へ。OK は sample_unit(seed, image_id) < rate のときだけ人へ。"""
    p0 = RoutePolicy.from_dict({"kind": "sample", "rate": 0.0, "seed": "s"})
    p1 = RoutePolicy.from_dict({"kind": "sample", "rate": 1.0, "seed": "s"})
    assert decide(p0, "a", "NG", 0.9).reason == "judged_ng"
    assert decide(p0, "a", None, None).reason == "parse_failed"
    assert decide(p0, "a", "OK", 0.9) == route.RouteDecision("auto", "OK", "not_sampled_ok", 0.1)
    assert decide(p1, "a", "OK", 0.9).reason == "sampled_ok"
    # 確信度が無くても OK なら抜き取りの対象（確信度を使わない）
    assert decide(p0, "a", "OK", None).route == "auto"


def test_sample_unit_deterministic_and_rate():
    """CI-HR3：同じ seed・画像なら同じ値。[0, 1)。多数の ID で割合がおおよそ rate になる。"""
    assert sample_unit("s", "pcb1/normal/0001") == sample_unit("s", "pcb1/normal/0001")
    assert sample_unit("s", "a") != sample_unit("t", "a")
    vals = [sample_unit("ac14", f"img{i}") for i in range(4000)]
    assert all(0 <= v < 1 for v in vals)
    assert abs(sum(v < 0.3 for v in vals) / 4000 - 0.3) < 0.03


def test_all_policy():
    """CI-HR5：all はすべて人へ（判定が NG でも、解析失敗でも）。"""
    p = RoutePolicy.from_dict({"kind": "all"})
    assert {decide(p, "a", j, c).route for j, c in [("OK", 0.99), ("NG", 0.99), (None, None)]} == {"human"}


@pytest.mark.parametrize("bad", [
    {"kind": "nope"}, {"kind": "confidence_band", "lo": 0.8, "hi": 0.4}, {"kind": "confidence_band", "lo": 0.4},
    {"kind": "confidence_band", "lo": True, "hi": 0.8}, {"kind": "sample", "rate": 1.5},
    {"kind": "sample", "rate": 0.5, "seed": 3}, {"kind": "all", "rate": 0.5}, [],
])
def test_policy_validation(bad):
    """CI-HR4：形が違う設定は ValueError（黙って既定値にしない）。"""
    with pytest.raises(ValueError):
        RoutePolicy.from_dict(bad)


# ---------------- judge_handler.py ----------------

class FakeBody(io.BytesIO):
    pass


class FakeS3:
    def __init__(self, objects):
        self.objects, self.calls = objects, []

    def get_object(self, Bucket, Key):
        self.calls.append(Key)
        return {"Body": FakeBody(self.objects[Key])}


class FakeBedrock:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.kwargs = text, exc, None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return {"output": {"message": {"content": [{"text": self.text}]}},
                "usage": {"inputTokens": 608, "outputTokens": 27}, "stopReason": "end_turn"}


def _judge(text=None, exc=None, policy=BAND):
    s3 = FakeS3({"ref.JPG": b"R", "img.JPG": b"T"})
    br = FakeBedrock(text, exc)
    j = Judge(s3, br, "jp.amazon.nova-2-lite-v1:0", "PROMPT", {"temperature": 0, "maxTokens": 300},
              ParseSpec(allow_code_fence=True), policy)
    return j, s3, br


EVENT = {"image_id": "pcb1/normal/0001", "bucket": "b", "key": "img.JPG", "ref_keys": ["ref.JPG"]}


def test_judge_layout_and_result():
    """CI-HR6・HR7：見本 → 検査対象 → 指示の順で送り、判定・回し方・usage を返す。"""
    j, s3, br = _judge('{"judgement": "OK", "defect": "", "confidence": 0.9}')
    out = j(EVENT)
    content = br.kwargs["messages"][0]["content"]
    assert [c.get("image", {}).get("source", {}).get("bytes") for c in content[:2]] == [b"R", b"T"]
    assert content[2] == {"text": "PROMPT"}
    assert br.kwargs["inferenceConfig"] == {"temperature": 0, "maxTokens": 300}
    assert out["judgement"] == "OK" and out["route"] == "auto" and out["auto_judgement"] == "OK"
    assert out["input_tokens"] == 608 and out["output_tokens"] == 27
    json.dumps(out)  # 状態の出力にできる


def test_judge_parse_failure_goes_to_human():
    """CI-HR7：読めない応答は解析失敗として人へ（救済しない）。"""
    j, _, _ = _judge('判定は OK です {"judgement": "OK", "confidence": 0.9}')
    out = j(EVENT)
    assert out["judgement"] is None and out["parse_error"] and out["route"] == "human"


def test_judge_api_error_raises():
    """CI-HR6：Bedrock の呼び出しが失敗したら例外（自動で OK にしない）。"""
    err = ClientError({"Error": {"Code": "ThrottlingException", "Message": "x"}}, "Converse")
    j, _, _ = _judge(exc=err)
    with pytest.raises(ClientError):
        j(EVENT)


@pytest.mark.parametrize("bad", [
    {}, {**EVENT, "image_id": ""}, {**EVENT, "ref_keys": "ref.JPG"}, {**EVENT, "ref_keys": [""]},
])
def test_validate_event(bad):
    """CI-HR6（pre）：入力の形が違えば ValueError。"""
    with pytest.raises(ValueError):
        validate_event(bad)


def test_image_format():
    assert image_format("a/b/0001.JPG") == "jpeg" and image_format("x.png") == "png"
    with pytest.raises(ValueError):
        image_format("x.bmp")


def test_raw_text_truncated():
    """CI-HR7：応答の本文は 4,000 文字で切り、切ったことを記録する。"""
    resp = {"output": {"message": {"content": [{"text": "x" * 5000}]}}, "usage": {}}
    out = judge_result("a", "m", resp, 1.0, ParseSpec(), BAND)
    assert len(out["raw_text"]) == 4000 and out["raw_text_truncated"] is True


def test_build_messages_no_refs():
    assert build_messages([(b"T", "jpeg")], "P")[0]["content"][-1] == {"text": "P"}


# ---------------- review_handler.py ----------------

class FakeDdb:
    def __init__(self, items):
        self.items = {k: dict(v) for k, v in items.items()}
        self.updates = []

    def get_item(self, TableName, Key, ConsistentRead):
        it = self.items.get(Key["review_id"]["S"])
        return {"Item": dict(it)} if it else {}

    def update_item(self, TableName, Key, UpdateExpression, ConditionExpression, ExpressionAttributeNames,
                    ExpressionAttributeValues):
        rid = Key["review_id"]["S"]
        it = self.items[rid]
        expected = ExpressionAttributeValues[":pending" if ":pending" in ExpressionAttributeValues else ":sub"]
        if it["status"] != expected:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": ""}}, "UpdateItem")
        if ":pending" in ExpressionAttributeValues:
            it["status"] = ExpressionAttributeValues[":sub"]
            it["human_decision"] = ExpressionAttributeValues[":d"]
        else:
            it["status"] = ExpressionAttributeValues[":exp"]
        self.updates.append((rid, it["status"]["S"]))

    def scan(self, **kwargs):
        s = kwargs["ExpressionAttributeValues"][":s"]["S"]
        return {"Items": [v for v in self.items.values() if v["status"]["S"] == s]}


class FakeSfn:
    def __init__(self, error_code=None):
        self.error_code, self.sent = error_code, []

    def send_task_success(self, taskToken, output):
        if self.error_code:
            raise ClientError({"Error": {"Code": self.error_code, "Message": ""}}, "SendTaskSuccess")
        self.sent.append((taskToken, json.loads(output)))


class FakePresign:
    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://s3.example/{Params['Key']}?sig=1&x=<y>"


def _item(status="pending", token="TOKEN"):
    it = {"review_id": {"S": "r1"}, "image_id": {"S": "pcb1/anomaly/001"}, "status": {"S": status},
          "s3_bucket": {"S": "b"}, "s3_key": {"S": "k.JPG"}, "requested_at": {"S": "2026-12-01T00:00:00Z"},
          "vlm_result": {"S": json.dumps({"judgement": "OK", "confidence": 0.7, "defect": "<b>", "route_reason": "in_band"})}}
    if token:
        it["task_token"] = {"S": token}
    return it


def _app(items, sfn=None):
    ddb = FakeDdb(items)
    sfn = sfn or FakeSfn()
    return ReviewApp(ddb, sfn, FakePresign(), "T", now=lambda: "2026-12-01T00:00:05.000Z"), ddb, sfn


def test_decide_sends_token_once():
    """CI-HR9：pending を submitting にしてから SendTaskSuccess を1回だけ返す。2回目は 409。"""
    app, ddb, sfn = _app({"r1": _item()})
    assert app.decide("r1", "NG") == (303, "submitted")
    assert sfn.sent == [("TOKEN", {"decision": "NG", "submitted_at": "2026-12-01T00:00:05.000Z"})]
    assert ddb.items["r1"]["status"]["S"] == "submitting"
    assert app.decide("r1", "OK")[0] == 409
    assert len(sfn.sent) == 1


@pytest.mark.parametrize("rid,decision,status", [("r1", "ok", 400), ("r1", "", 400), ("", "OK", 400),
                                                 ("nope", "OK", 404)])
def test_decide_validation(rid, decision, status):
    """CI-HR8：判定は OK / NG の完全一致だけ。無い review_id は 404。どちらもトークンを返さない。"""
    app, _, sfn = _app({"r1": _item()})
    assert app.decide(rid, decision)[0] == status
    assert sfn.sent == []


def test_decide_not_pending():
    """CI-HR9：pending でない、またはトークンが無ければ 409。"""
    app, _, sfn = _app({"r1": _item(status="timed_out", token=None)})
    assert app.decide("r1", "OK")[0] == 409 and sfn.sent == []


@pytest.mark.parametrize("code", ["TaskTimedOut", "TaskDoesNotExist", "InvalidToken"])
def test_decide_expired(code):
    """CI-HR10：もう待っていない実行なら 410 を返し、status を expired にする。"""
    app, ddb, _ = _app({"r1": _item()}, FakeSfn(code))
    assert app.decide("r1", "OK")[0] == 410
    assert ddb.items["r1"]["status"]["S"] == "expired"


def test_decide_other_error_raises():
    """CI-HR10：それ以外のエラーは握りつぶさない。"""
    app, _, _ = _app({"r1": _item()}, FakeSfn("AccessDeniedException"))
    with pytest.raises(ClientError):
        app.decide("r1", "OK")


def test_page_escapes_and_hides_token():
    """CI-HR11：一覧に値をエスケープして出し、タスクトークンは出さない。"""
    app, _, _ = _app({"r1": _item(), "r2": {**_item(status="timed_out", token=None), "review_id": {"S": "r2"}}})
    body = app({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/"})["body"]
    assert "TOKEN" not in body
    assert "&lt;b&gt;" in body and "<b>" not in body
    assert "x=&lt;y&gt;" in body
    assert "確認待ち 1 件" in body and "要エスカレーション） 1 件" in body


def test_http_routing_and_form():
    """CI-HR8：POST /decide はフォームを読む（base64 も）。重複したキーは 400。"""
    app, _, sfn = _app({"r1": _item()})
    body = base64.b64encode(b"review_id=r1&decision=OK").decode()
    r = app({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/decide", "body": body,
             "isBase64Encoded": True})
    assert r["statusCode"] == 303 and r["headers"]["location"] == "/" and len(sfn.sent) == 1
    with pytest.raises(ValueError):
        parse_form({"body": "decision=OK&decision=NG"})
    r = app({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/decide", "body": "decision=OK&decision=NG"})
    assert r["statusCode"] == 400
    assert app({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/x"})["statusCode"] == 404


def test_decide_expired_after_timeout_recorded():
    """CI-HR10：時間切れの記録（timed_out）が先に書かれていても 410 を返し、timed_out を残す。"""
    app, ddb, _ = _app({"r1": _item()}, FakeSfn("TaskTimedOut"))
    orig = ddb.update_item

    def racing_update(**kw):
        orig(**kw)  # pending → submitting
        ddb.items["r1"]["status"] = {"S": "timed_out"}  # その直後に状態機械が timed_out を書いた
        ddb.update_item = orig

    ddb.update_item = racing_update
    assert app.decide("r1", "OK")[0] == 410
    assert ddb.items["r1"]["status"]["S"] == "timed_out"


# ---------------- runlog.py・driver.py（AWS は呼ばない部分） ----------------

import pandas as pd  # noqa: E402

from vlmlab.cost import Price  # noqa: E402
from vlmlab.review.driver import execution_name, make_review_split, plan_start, write_run_log  # noqa: E402
from vlmlab.review.runlog import build_rows, summarize  # noqa: E402

PRICE = Price("jp.amazon.nova-2-lite-v1:0", "standard", 0.396, 3.311)
LABELS = {"a/anomaly/1": {"category": "a", "label": "anomaly"}, "a/normal/1": {"category": "a", "label": "normal"},
          "a/anomaly/2": {"category": "a", "label": "anomaly"}}


def _ddb_item(rid, image_id, status, final=None, by=None, req=None, dec="2026-12-01T00:00:10Z", tokens=(608, 27)):
    vlm = {"judgement": "OK", "confidence": 0.9, "route": "auto" if status == "auto_decided" else "human",
           "input_tokens": tokens[0], "output_tokens": tokens[1], "latency_s": 0.8}
    it = {"review_id": {"S": rid}, "image_id": {"S": image_id}, "status": {"S": status},
          "vlm_result": {"S": json.dumps(vlm)}, "started_at": {"S": "2026-12-01T00:00:00Z"}}
    for k, v in (("final_judgement", final), ("decided_by", by), ("requested_at", req), ("decided_at", dec)):
        if v is not None:
            it[k] = {"S": v}
    return it


def test_build_rows_and_summary():
    """CI-HR12・HR13：見逃し・誤検知・人の待ち時間・費用（トークン×単価）・状態遷移の見積り。"""
    items = [
        _ddb_item("t-1", "a/anomaly/1", "auto_decided", "OK", "vlm"),                      # 見逃し
        _ddb_item("t-2", "a/normal/1", "human_decided", "NG", "human", req="2026-12-01T00:00:01Z"),  # 誤検知
        _ddb_item("t-3", "a/anomaly/2", "timed_out", None, "none", req="2026-12-01T00:00:01Z"),
    ]
    rows = build_rows(items, LABELS, PRICE)
    r = {x["review_id"]: x for x in rows}
    assert r["t-1"]["missed"] is True and r["t-1"]["correct"] is False
    assert r["t-2"]["false_ng"] is True and r["t-2"]["human_wait_s"] == 9.0
    assert r["t-3"]["final_judgement"] is None and r["t-3"]["missed"] is None and r["t-3"]["human_wait_s"] is None
    assert r["t-1"]["bedrock_cost_usd"] == pytest.approx((608 * 0.396 + 27 * 3.311) / 1e6)
    s = summarize(rows, 0.000025)
    assert s["n"] == 3 and s["n_human"] == 2 and s["undecided"] == 1
    assert s["missed"] == 1 and s["miss_rate"] == 0.5 and s["false_ng_rate"] == 1.0
    assert s["est_state_transitions"] == 3 + 5 + 4
    assert s["est_sfn_cost_usd_without_free_tier"] == pytest.approx(12 * 0.000025)
    assert s["human_wait_s_median"] == 9.0


def test_build_rows_rejects_unfinished_or_unknown():
    """CI-HR12：終わっていない記録や、正解の無い画像があれば止める（数え漏れを黙って出さない）。"""
    with pytest.raises(ValueError):
        build_rows([_ddb_item("t-1", "a/anomaly/1", "pending")], LABELS, PRICE)
    with pytest.raises(ValueError):
        build_rows([_ddb_item("t-1", "zzz", "auto_decided", "OK", "vlm")], LABELS, PRICE)
    with pytest.raises(ValueError):
        build_rows([_ddb_item("t-1", "a/anomaly/1", "auto_decided", "ok", "vlm")], LABELS, PRICE)


def test_summary_zero_denominators():
    """CI-HR13：不良品・良品が0件なら率は None（0/0 を 0 にしない）。"""
    s = summarize([], 0.000025)
    assert s["miss_rate"] is None and s["false_ng_rate"] is None and s["bedrock_cost_usd"] == 0


def test_write_run_log_filters_tag(tmp_path):
    """CI-HR12：tag で始まる実行だけを書く（試運転と本番を混ぜない）。"""
    split = pd.DataFrame([{"image_id": k, **v} for k, v in LABELS.items()])
    items = [_ddb_item("trial-001-x", "a/anomaly/1", "auto_decided", "OK", "vlm"),
             _ddb_item("main-001-x", "a/normal/1", "auto_decided", "OK", "vlm")]
    s = write_run_log(items, split, PRICE, tmp_path / "run_log.csv", "main")
    assert s["n"] == 1
    assert (tmp_path / "run_log_summary.json").exists()
    assert (tmp_path / "run_log.csv").read_text(encoding="utf-8").count("\n") == 2


def test_make_review_split_and_plan():
    """CI-HR15・HR16：品目×正解ごとに n 枚、同じ seed なら同じ。入力に正解を入れない。実行名は規則どおり。"""
    rows = [{"image_id": f"{c}/{l}/{i:03d}", "category": c, "label": l, "split": "test",
             "path": f"{c}/Data/Images/{l}/{i:03d}.JPG"} for c in ("pcb1", "capsules") for l in ("normal", "anomaly")
            for i in range(50)]
    t = pd.DataFrame(rows)
    a, b = make_review_split(t), make_review_split(t)
    assert len(a) == 20 and a.equals(b)
    assert a.groupby(["category", "label"]).size().tolist() == [5, 5, 5, 5]
    refs = pd.DataFrame([{"image_id": f"{c}/normal/r{k}", "ref_rank": k, "category": c, "label": "normal",
                          "split": "train", "path": f"{c}/ref{k}.JPG"} for c in ("pcb1", "capsules") for k in (2, 1)])
    plan = plan_start(a, refs, "ac14-main")
    assert all("label" not in p["input"] for p in plan)
    assert plan[0]["input"]["ref_keys"] == [f"refs/{plan[0]['input']['image_id'].split('/')[0]}/ref1.JPG"]
    assert plan[0]["name"].startswith("ac14-main-001-")
    with pytest.raises(ValueError):
        execution_name("x" * 80, 1, "a")
    assert execution_name("t", 2, "pcb1/normal/0797") == "t-002-pcb1-normal-0797"


def test_content_type():
    """CI-HR17：S3 に上げる画像の Content-Type は拡張子から決める（試運転で binary/octet-stream だったため）。"""
    from vlmlab.review.driver import content_type

    assert content_type("images/pcb1/Data/Images/Normal/0011.JPG") == "image/jpeg"
    assert content_type("x.png") == "image/png"
    with pytest.raises(ValueError):
        content_type("x.gif")
