"""請求書（12/23 の記事）の単体テスト。各テストの docstring に契約 ID（docs/contracts.md の invoice の節）を書く。"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from vlmlab.bedrock import ModelRequest, ModelResponse, build_converse_kwargs
from vlmlab.cost import BudgetExceeded, Price, PricingTable
from vlmlab.data import EncodedImage
from vlmlab.invoice import report as irep
from vlmlab.invoice import run as irun
from vlmlab.invoice.render import format_date, split_halves
from vlmlab.invoice.schema import SCHEMA, build_text, output_config
from vlmlab.invoice.score import field_correct, line_item_matches, parse_invoice, schema_errors, score
from vlmlab.invoice.truth import (
    FIELDS, check_digit, consistent, is_valid_registration_number, make_all,
)


# ---------- 正解（truth.py） ----------

def test_check_digit_nta_example():
    """CI-IT1: 国税庁の例。会社法人等番号 700110005901 の検査用数字は 8"""
    assert check_digit("700110005901") == "8"
    assert is_valid_registration_number("T8700110005901")
    assert not is_valid_registration_number("T7700110005901")
    assert not is_valid_registration_number("8700110005901")
    assert not is_valid_registration_number("T870011000590")
    assert not is_valid_registration_number(None)


def test_check_digit_rejects_bad_base():
    """CI-IT1: 12桁の ASCII 数字でなければ ValueError"""
    for bad in ["70011000590", "７００１１０００５９０１", "70011000590a"]:
        with pytest.raises(ValueError):
            check_digit(bad)


def test_make_all_deterministic_and_consistent():
    """CI-IT2〜IT5: 同じシードで同じ正解。60枚、登録番号は発行元ごとに1つで重複なし、合計＝小計＋税、税は税率ごとに切り捨て"""
    a, issuers = make_all(42)
    b, _ = make_all(42)
    assert a == b
    assert len(a) == 60 and len(issuers) == 30
    rns = [i["registration_number"] for i in issuers]
    assert len(set(rns)) == 30 and all(is_valid_registration_number(r) for r in rns)
    for t in a:
        assert 3 <= len(t["line_items"]) <= 8
        assert sum(it["reduced_rate"] for it in t["line_items"]) <= 3
        for it in t["line_items"]:
            assert it["amount"] == it["quantity"] * it["unit_price"]
        assert t["tax_10"] == t["subtotal_10"] * 10 // 100 and t["tax_8"] == t["subtotal_8"] * 8 // 100
        assert t["total"] == t["subtotal_10"] + t["tax_10"] + t["subtotal_8"] + t["tax_8"]
        assert consistent(t)["all_ok"]
    assert {t["date_style"] for t in a} == {"wareki", "seireki"}


def test_format_date():
    """CI-IR1: 西暦と令和（令和元年 = 2019年）"""
    assert format_date("2026-03-14", "seireki") == "2026年3月14日"
    assert format_date("2026-03-14", "wareki") == "令和8年3月14日"
    assert format_date("2019-05-01", "wareki") == "令和元年5月1日"
    with pytest.raises(ValueError):
        format_date("2018-05-01", "wareki")


def test_split_halves_overlap():
    """CI-IR4: 上下に分け、重なりは100px。合わせると元の高さ＋重なり"""
    img = Image.new("RGB", (1240, 1754), "white")
    top, bottom = split_halves(img, overlap=100)
    assert top.size == (1240, 877 + 50) and bottom.size == (1240, 1754 - 827)
    assert top.size[1] + bottom.size[1] == 1754 + 100


# ---------- スキーマ・プロンプト（schema.py） ----------

def _objects(s):
    if isinstance(s, dict):
        if s.get("type") == "object":
            yield s
        for v in s.values():
            yield from _objects(v)


def test_schema_obeys_bedrock_limits():
    """CI-IP1: すべての object に additionalProperties: false。省略できる項目 0、null を許す項目は registration_number だけ。数値・長さの制約なし"""
    for o in _objects(SCHEMA):
        assert o["additionalProperties"] is False
        assert set(o["required"]) == set(o["properties"])
    txt = json.dumps(SCHEMA)
    for kw in ["minimum", "maximum", "minLength", "maxLength", "multipleOf"]:
        assert kw not in txt
    nullable = [k for k, v in SCHEMA["properties"].items() if isinstance(v.get("type"), list)]
    assert nullable == ["registration_number"]


def test_build_text_and_output_config():
    """CI-IP2、IP3: 構造化出力なしだけスキーマを文に付ける。outputConfig の schema は JSON の文字列"""
    assert "JSON スキーマ" not in build_text(True, False)
    assert json.dumps(SCHEMA, ensure_ascii=False) in build_text(False, False)
    assert "上下に分けた" in build_text(False, True) and "上下に分けた" not in build_text(False, False)
    oc = output_config()["textFormat"]
    assert oc["type"] == "json_schema"
    assert json.loads(oc["structure"]["jsonSchema"]["schema"]) == SCHEMA


def test_converse_kwargs_output_config_only_when_set():
    """CI-B3: output_config を渡したときだけ outputConfig を送る"""
    img = EncodedImage(data=b"x", format="png", width=1, height=1)
    base = ModelRequest(model_id="m", images=[img], text="t")
    assert "outputConfig" not in build_converse_kwargs(base)
    oc = output_config()
    req = ModelRequest(model_id="m", images=[img], text="t", output_config=oc)
    assert build_converse_kwargs(req)["outputConfig"] == oc


# ---------- 読み取り・採点（score.py） ----------

def _truth():
    return make_all(42)[0][0]


def _pred_from(t):
    return {k: t[k] for k in SCHEMA["properties"] if k != "line_items"} | {
        "line_items": [dict(it) for it in t["line_items"]]}


def test_parse_strict_fenced_and_failures():
    """CI-IS1: 本文全体が JSON → strict、```json 囲み1つ → fenced、それ以外は失敗（救済しない）"""
    p = _pred_from(_truth())
    s = json.dumps(p, ensure_ascii=False)
    assert parse_invoice(s)["how"] == "strict"
    assert parse_invoice(f"```json\n{s}\n```")["how"] == "fenced"
    assert parse_invoice(f"```\n{s}\n```")["how"] == "fenced"
    assert parse_invoice(f"以下です。\n{s}")["error"] == "json: decode error"
    assert parse_invoice(f"```json\n{s}\n```\n補足です。")["parsed"] is None
    assert parse_invoice("[1, 2]")["error"] == "json: not an object"
    assert parse_invoice(None)["error"] == "no text"


def test_schema_errors():
    """CI-IS2: スキーマに合わない点を列挙する（足りない・余分・型・日付の形・登録番号の形・明細）"""
    p = _pred_from(_truth())
    assert schema_errors(p) == []
    q = dict(p, total="12,345", extra=1, issue_date="2026/03/04", registration_number="T123")
    q.pop("tax_8")
    errs = set(schema_errors(q))
    assert {"type:total", "extra:extra", "format:issue_date", "pattern:registration_number", "missing:tax_8"} <= errs
    assert schema_errors(dict(p, registration_number=None)) == []
    r = dict(p, line_items=[{"description": "a", "quantity": True, "unit_price": 1, "amount": 1, "reduced_rate": False}])
    assert "item0:type" in schema_errors(r)


def test_field_correct_rules():
    """CI-IS3: 文字列は空白（全角を含む）だけ除いて完全一致。金額は整数で完全一致（文字列・bool・小数は不正解）"""
    assert field_correct("issuer_name", "株式会社 ミナホル　商事", "株式会社ミナホル商事")
    assert not field_correct("issuer_name", "（株）ミナホル商事", "株式会社ミナホル商事")
    assert not field_correct("issue_date", None, "2026-03-04")
    assert field_correct("total", 100, 100)
    for bad in ["100", 100.0, True]:
        assert not field_correct("total", bad, 100)
    assert not field_correct("tax_8", True, 1)


def test_line_item_matches_order_free_multiset():
    """CI-IS4: 並びは問わず、全列一致の行を数える。応答の1行は1回だけ使う"""
    t = [{"description": "弁当", "quantity": 2, "unit_price": 500, "amount": 1000, "reduced_rate": True},
         {"description": "弁当", "quantity": 2, "unit_price": 500, "amount": 1000, "reduced_rate": True},
         {"description": "配送料", "quantity": 1, "unit_price": 800, "amount": 800, "reduced_rate": False}]
    p = [dict(t[2]), dict(t[0], description="弁 当")]
    assert line_item_matches(p, t) == 2
    assert line_item_matches([dict(t[0], reduced_rate=False)], t) == 0
    assert line_item_matches("x", t) == 0
    assert line_item_matches([{"description": "弁当"}], t) == 0


def test_score_and_consistency_do_not_use_truth():
    """CI-IS5、IS6: 正しい読み取りは全項目正解で引っかからない。読めない応答は全項目不正解で引っかかる。整合は読み取り値だけで決まる"""
    t = _truth()
    good = score(_pred_from(t), t)
    assert good["all_fields"] and not good["flagged"] and good["n_items_matched"] == len(t["line_items"])
    none = score(None, t)
    assert not any(none[f"ok_{f}"] for f in FIELDS) and none["flagged"] and none["n_items_matched"] == 0
    # 会社名を誤っても金額と番号が整合すれば引っかからない（チェックで見えない誤り）
    wrong_name = score(dict(_pred_from(t), issuer_name="別の会社"), t)
    assert not wrong_name["all_fields"] and wrong_name["checkable_all"] and not wrong_name["flagged"]
    # 登録番号の1桁違いは検査用数字で引っかかる
    rn = t["registration_number"]
    bad_rn = rn[:-1] + str((int(rn[-1]) + 1) % 10)
    s = score(dict(_pred_from(t), registration_number=bad_rn), t)
    assert not s["cons_rn_ok"] and s["flagged"]
    # 税を誤ると tax と sum が崩れる
    s = score(dict(_pred_from(t), tax_10=t["tax_10"] + 1), t)
    assert not s["cons_tax_ok"] and not s["cons_sum_ok"]
    assert consistent({"subtotal_10": True, "tax_10": 0, "subtotal_8": 0, "tax_8": 0, "total": 1})["sum_ok"] is False


# ---------- 実行（run.py） ----------

class FakeConverse:
    def __init__(self, texts, in_tok=1000, out_tok=300):
        self.texts = list(texts)
        self.requests: list[ModelRequest] = []
        self.in_tok, self.out_tok = in_tok, out_tok

    def invoke(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        t = self.texts.pop(0)
        if isinstance(t, Exception):
            return ModelResponse(raw_text=None, api_error=str(t))
        return ModelResponse(raw_text=t, input_tokens=self.in_tok, output_tokens=self.out_tok,
                             latency_s=1.0, stop_reason="end_turn")


def _setup(tmp_path: Path, n: int = 3):
    truths = make_all(42)[0][:n]
    rows = []
    for v in ("clean", "degraded"):
        for t in truths:
            rows.append({"doc_id": f"{t['invoice_id']}_{v}", "invoice_id": t["invoice_id"], "variant": v,
                         "path": f"img/{t['invoice_id']}_{v}.png"})
    (tmp_path / "img").mkdir()
    for r in rows:
        Image.new("RGB", (100, 140), "white").save(tmp_path / r["path"])
    split = tmp_path / "split.csv"
    with split.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    pricing = PricingTable([Price("m", "standard", 1.0, 5.0)])
    return truths, rows, pricing


def _cfg(**kw):
    d = dict(name="t", condition="H-SO", model_id="m", region="r", budget_usd=1.0, structured=True,
             split="split.csv")
    d.update(kw)
    return irun.InvoiceConfig(**d)


def test_select_trial_takes_same_invoice_clean_and_degraded(tmp_path):
    """CI-IR5: 一覧はきれい → 劣化の順。2件なら同じ請求書のきれい・劣化"""
    _, rows, _ = _setup(tmp_path)
    got = irun.select_trial(rows, 2)
    assert [r["doc_id"] for r in got] == ["inv0001_clean", "inv0001_degraded"]
    with pytest.raises(ValueError):
        irun.select_trial(rows, 0)


def test_execute_records_and_structured_flag(tmp_path):
    """CI-IR6〜IR8: 1件1行の記録。構造化出力の条件だけ outputConfig を付け、N-T は上下2枚の PNG"""
    truths, rows, pricing = _setup(tmp_path)
    s = json.dumps(_pred_from(truths[0]), ensure_ascii=False)
    fake = FakeConverse([s] * 6)
    out = irun.execute(_cfg(), tmp_path, tmp_path / "runs", pricing, fake)
    recs = irun.load_records(out / "calls.jsonl")
    assert len(recs) == 6 and all(r["cost_usd"] == pytest.approx((1000 * 1 + 300 * 5) / 1e6) for r in recs)
    assert all(q.output_config is not None for q in fake.requests)
    fake2 = FakeConverse([s] * 2)
    irun.execute(_cfg(name="nt", condition="N-T", structured=False, split_halves=True), tmp_path,
                 tmp_path / "runs", pricing, fake2, limit=2)
    assert all(q.output_config is None and len(q.images) == 2 and q.images[0].format == "png" for q in fake2.requests)
    assert [q.images[0].height for q in fake2.requests] == [70 + 50, 70 + 50]


def test_execute_resume_and_identity_guard(tmp_path):
    """CI-IR7、IR8: 成功した doc は飛ばし、API エラーの doc は呼び直す。設定が違えば止める"""
    truths, rows, pricing = _setup(tmp_path)
    s = json.dumps(_pred_from(truths[0]), ensure_ascii=False)
    irun.execute(_cfg(), tmp_path, tmp_path / "runs", pricing, FakeConverse([s, RuntimeError("Throttling")] + [s] * 4))
    fake = FakeConverse([s])
    out = irun.execute(_cfg(), tmp_path, tmp_path / "runs", pricing, fake)
    assert len(fake.requests) == 1 and len(irun.load_records(out / "calls.jsonl")) == 7
    with pytest.raises(ValueError, match="settings differ"):
        irun.execute(_cfg(structured=False), tmp_path, tmp_path / "runs", pricing, FakeConverse([]))


def test_execute_budget_stops_before_exceeding(tmp_path):
    """CI-C4（invoice）: 累計＋最大の1件が予算を超える前に止める"""
    truths, rows, pricing = _setup(tmp_path)
    s = json.dumps(_pred_from(truths[0]), ensure_ascii=False)
    with pytest.raises(BudgetExceeded):
        irun.execute(_cfg(budget_usd=0.006), tmp_path, tmp_path / "runs", pricing, FakeConverse([s] * 6))
    recs = irun.load_records(tmp_path / "runs" / "invoice" / "t" / "calls.jsonl")
    assert len(recs) == 2  # 0.0025 × 2 = 0.005、次で 0.0075 > 0.006


# ---------- 集計（report.py） ----------

def _rec(doc, inv, variant, text, api_error=None, seq=0, latency=1.0, cost=0.001):
    return {"run_name": "r", "condition": "N", "model_id": "m", "repeat": 1, "doc_id": doc, "invoice_id": inv,
            "variant": variant, "raw_text": text, "api_error": api_error, "input_tokens": 100, "output_tokens": 50,
            "latency_s": latency, "cost_usd": cost, "stop_reason": "end_turn", "seq": seq}


def test_report_metrics(tmp_path):
    """CI-IM1〜IM5: API エラーは分母から外す。囲みの扱いで解析失敗率が変わる。検出率・誤報率、0/0 は None"""
    truths = {t["invoice_id"]: t for t in make_all(42)[0][:3]}
    t1, t2, t3 = truths.values()
    good1 = json.dumps(_pred_from(t1), ensure_ascii=False)
    bad2 = json.dumps(dict(_pred_from(t2), tax_10=t2["tax_10"] + 1), ensure_ascii=False)  # 引っかかる誤り
    name3 = "```json\n" + json.dumps(dict(_pred_from(t3), issuer_name="別"), ensure_ascii=False) + "\n```"  # 見えない誤り
    recs = [_rec("inv0001_clean", "inv0001", "clean", good1, seq=0, latency=90.0),
            _rec("inv0002_clean", "inv0002", "clean", bad2, seq=1),
            _rec("inv0003_clean", "inv0003", "clean", name3, seq=2),
            _rec("inv0001_degraded", "inv0001", "degraded", None, api_error="Throttling", seq=3, cost=0.0)]
    df = irep.scored_frame(recs, truths, allow_fence=True)
    s = irep.summarize(df)
    assert s["n_calls"] == 4 and s["n_api_error"] == 1 and s["n"] == 3
    assert s["parse_fail_rate"] == 0 and s["fenced_rate"] == pytest.approx(1 / 3)
    assert s["acc_all_fields"] == pytest.approx(1 / 3)
    assert s["detect_rate"] == pytest.approx(1 / 2) and s["false_alarm_rate"] == 0
    assert s["n_wrong_checkable"] == 1 and s["detect_rate_checkable"] == 1
    assert s["first_call_latency_s"] == 90.0
    strict = irep.summarize(irep.scored_frame(recs, truths, allow_fence=False))
    assert strict["parse_fail_rate"] == pytest.approx(1 / 3) and strict["acc_issuer_name"] == pytest.approx(2 / 3) and strict["acc_total"] == pytest.approx(2 / 3)
    only_good = irep.summarize(irep.scored_frame(recs[:1], truths, allow_fence=True))
    assert only_good["detect_rate"] is None  # 誤りが0件なら検出率は計算しない


def test_latest_per_doc_prefers_success():
    """CI-IM1: doc ごとに最後の成功、成功が無ければ最後の失敗"""
    a = _rec("d", "inv0001", "clean", "x", seq=0)
    b = _rec("d", "inv0001", "clean", None, api_error="e", seq=1)
    assert irep.latest_per_doc([a, b]) == [a]
    assert irep.latest_per_doc([b]) == [b]


def test_agreement():
    """CI-IM6: 両方で読めた doc のうち10項目の値が同じ割合。読めなければ一致としない"""
    truths = {t["invoice_id"]: t for t in make_all(42)[0][:2]}
    t1, t2 = truths.values()
    r1 = [_rec("inv0001_clean", "inv0001", "clean", json.dumps(_pred_from(t1), ensure_ascii=False)),
          _rec("inv0002_clean", "inv0002", "clean", "壊れた")]
    r2 = [_rec("inv0001_clean", "inv0001", "clean", json.dumps(dict(_pred_from(t1), line_items=[]), ensure_ascii=False)),
          _rec("inv0002_clean", "inv0002", "clean", "壊れた")]
    ag = irep.agreement(irep.scored_frame(r1, truths, True), irep.scored_frame(r2, truths, True))
    assert ag == {"n_common": 2, "n_same": 1, "agreement": 0.5}
