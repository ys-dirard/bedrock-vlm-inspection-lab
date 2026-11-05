"""MMAD（11/19 の記事）の単体テスト。各テストの docstring に契約 ID（docs/contracts.md の mmad の節）を書く。"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from vlmlab.bedrock import ModelRequest, ModelResponse
from vlmlab.cost import BudgetExceeded, Price, PricingTable
from vlmlab.data import EncodedImage
from vlmlab.mmad import data as md
from vlmlab.mmad import run as mrun
from vlmlab.mmad.invoke import response_from_anthropic
from vlmlab.mmad.prompt import anthropic_body, build_text
from vlmlab.mmad.report import build_report, load_calls
from vlmlab.mmad.score import accuracy_table, extract_answer, unparsed_rate, wilson_interval

REPO = Path(__file__).resolve().parents[1]


# ---------- 合成データ ----------

def _q(qtype: str, answer: str = "A", n_opt: int = 4, text: str = "Q?") -> dict:
    opts = {k: f"opt {k}." for k in "ABCD"[:n_opt]}
    return {"Question": text, "Answer": answer, "Options": opts, "type": qtype}


def synthetic_mmad(n_per_cell: int = 6) -> dict:
    """4つの元データセット（DS-MVTec を含む）× 良品・不良品の画像。各画像に5問。"""
    out = {}
    sources = ["DS-MVTec", "MVTec-LOCO", "VisA", "GoodsAD"]
    for src in sources:
        for i in range(n_per_cell):
            for kind in ("good", "crack"):
                key = f"{src}/cat/test/{kind}/{i:03d}.png"
                conv = [_q("Anomaly Detection", "B" if kind == "good" else "A", 2)]
                if kind == "good":
                    conv += [_q("Object Classification"), _q("Object Structure"), _q("Object Details")]
                else:
                    conv += [_q("Defect Classification"), _q("Defect Localization"), _q("Defect Description"),
                             _q("Defect Analysis"), _q("Object Analysis")]
                out[key] = {"conversation": conv, "random_templates": [f"{src}/cat/train/good/{i:03d}.png"]}
    return out


# ---------- data ----------

def test_question_table_merges_names_and_flags_normal():
    """CI-MD2: DS-MVTec→MVTec-AD、Object Structure/Details→Object Analysis、is_normal は 'good' を含むか、template は random_templates[0]。"""
    t = md.question_table(synthetic_mmad(1))
    assert set(t["dataset"]) == set(md.DATASETS)
    assert set(t["task"]) == set(md.TASKS)
    row = t[t["qid"] == "DS-MVTec/cat/test/good/000.png#2"].iloc[0]
    assert row.dataset == "MVTec-AD" and row.task == "Object Analysis" and row.raw_type == "Object Structure"
    assert bool(row.is_normal) and row.template == "DS-MVTec/cat/train/good/000.png"
    assert not t[t["image_key"].str.contains("/crack/")]["is_normal"].any()


def test_question_table_rejects_unknown_type():
    """CI-MD2: 知らない課題名があれば止める。"""
    m = {"VisA/x/test/good/0.png": {"conversation": [_q("Something Else")], "random_templates": ["t"]}}
    with pytest.raises(ValueError):
        md.question_table(m)


def test_quotas_split_evenly_in_name_order():
    """CI-MD3: 150問を名前の順に 38・38・37・37。"""
    assert md.quotas(150) == {"GoodsAD": 38, "MVTec-AD": 38, "MVTec-LOCO": 37, "VisA": 37}
    assert sum(md.quotas(7).values()) == 7


def test_sample_is_stratified_and_deterministic():
    """CI-MD3、MD4: 層ごとの件数、異常の判別の良品・不良品の割り振り、同じシードで同じ結果。"""
    t = md.question_table(synthetic_mmad(6))
    a = md.sample_questions(t, per_task=8)
    b = md.sample_questions(t.sample(frac=1, random_state=0), per_task=8)  # 入力の並びが違っても同じ
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 8 * 7
    counts = a.groupby(["task", "dataset"]).size()
    assert (counts == 2).all()
    ad = a[a["task"] == "Anomaly Detection"]
    assert (ad.groupby(["dataset", "is_normal"]).size() == 1).all()  # q=2 → 良品1・不良品1
    assert list(a["record_id"][:3]) == ["m0001", "m0002", "m0003"]


def test_sample_odd_quota_gives_extra_to_anomalous():
    """CI-MD3: 異常の判別で端数が出たら不良品へ。"""
    t = md.question_table(synthetic_mmad(6))
    s = md.sample_questions(t, per_task=12)  # 各データセット3問 → 良品1・不良品2
    ad = s[s["task"] == "Anomaly Detection"].groupby("is_normal").size()
    assert ad[True] == 4 and ad[False] == 8


def test_sample_refuses_short_stratum():
    """CI-MD3: 層の問題が足りなければ、他で埋めずに止める。"""
    t = md.question_table(synthetic_mmad(1))
    with pytest.raises(ValueError):
        md.sample_questions(t, per_task=40)


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("w,h,expected", [(1024, 768, (512, 384)), (900, 900, (512, 512)), (300, 200, (512, 341))])
def test_encode_like_gpt4o_sizes(w, h, expected):
    """CI-MD5: scale = min(512/w, 512/h)、大きさは int で切り捨て。小さい画像は拡大する。JPEG にする。"""
    img = md.encode_like_gpt4o(_png(w, h))
    assert (img.width, img.height) == expected and img.format == "jpeg"
    with Image.open(io.BytesIO(img.data)) as im:
        assert im.format == "JPEG" and im.size == expected


# ---------- prompt ----------

INSTR = (REPO / "prompts" / "mmad_gpt4o_instruction.txt").read_text(encoding="utf-8")


def test_instruction_file_is_gpt4o_text():
    """CI-MP1: 指示文は gpt4o.py の instruction と同じ（先頭と末尾の改行も含む）。"""
    assert INSTR.startswith("\nYou are an industrial inspector who checks products by images.")
    assert INSTR.endswith("1. Answer: B.\n2. Answer: B.\n3. Answer: A.\n...\n")
    assert "\r" not in INSTR


def test_build_text_matches_gpt4o_concatenation():
    """CI-MP1、MP2: gpt4o.py の get_query と同じ連結。i 問目は1〜i 問目を並べ、前の答えは入れない。"""
    conv = [
        {"Question": "Is there any defect in the object?", "Answer": "A", "Options": {"A": "Yes.", "B": "No."}},
        {"Question": "What is the type of the defect?", "Answer": "C",
         "Options": {"A": "a.", "B": "b.", "C": "c.", "D": "d."}},
        {"Question": "Unused?", "Answer": "A", "Options": {"A": "x.", "B": "y."}},
    ]
    expected = (
        INSTR + "The first 1 image is the normal sample, which can be used as a template to compare."
        + "The last image is the query imageFollowing is the question list: \n"
        + "Question: Is there any defect in the object? \nA. Yes.\nB. No.\n\n"
        + "Question: What is the type of the defect? \nA. a.\nB. b.\nC. c.\nD. d.\n\n"
    )
    assert build_text(INSTR, conv, 1, n_shots=1) == expected
    zero = build_text(INSTR, conv, 0, n_shots=0)
    assert zero == INSTR + "The last image is the query imageFollowing is the question list: \n" \
        + "Question: Is there any defect in the object? \nA. Yes.\nB. No.\n\n"
    with pytest.raises(IndexError):
        build_text(INSTR, conv, 3, n_shots=1)


def test_anthropic_body_puts_images_before_text():
    """CI-MP3: Messages API の本文は [見本, 検査対象, 文] の順。画像は base64、media_type は image/jpeg。"""
    ref = EncodedImage(b"ref", "jpeg", 1, 1)
    tgt = EncodedImage(b"tgt", "jpeg", 1, 1)
    body = anthropic_body([ref, tgt], "TEXT", 200, 0.0)
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert content[0]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "cmVm"}
    assert content[1]["source"]["data"] == "dGd0" and content[2]["text"] == "TEXT"
    assert body["anthropic_version"] == "bedrock-2023-05-31" and body["max_tokens"] == 200 and body["temperature"] == 0.0


# ---------- score ----------

@pytest.mark.parametrize("text,expected", [
    ("Answer: B.", "B"),
    ("1. Answer: A.\n2. Answer: C.", "C"),          # 最後の記号
    ("A crack is visible. Answer: D", "D"),
    ("B", "B"),
    ("E.", "E"),
    ("The answer is b.", None),                      # 小文字は数えない
    ("No defect found.", None),
    ("", None),
    (None, None),
    ("ABC", None),                                   # 単語の一部は数えない（\b）
])
def test_extract_answer_like_gpt4o(text, expected):
    """CI-MS1: 正規表現 \\b([A-E])\\b の最後の一致。無ければ None（選択肢の文で合わせ直さない）。"""
    assert extract_answer(text) == expected


def _frame(rows):
    return pd.DataFrame(rows, columns=["dataset", "task", "is_normal", "answer", "pred"])


def test_accuracy_table_formula():
    """CI-MS2、MS3、MS4: データセットごとの課題の正答率、異常の判別は良品・不良品の平均、平均は単純平均。"""
    rows = []
    for ds in ("VisA", "GoodsAD"):
        # 異常の判別: 良品 2問中1正解（50%）、不良品 1問中1正解（100%）→ 75
        rows += [(ds, "Anomaly Detection", True, "B", "B"), (ds, "Anomaly Detection", True, "B", "A"),
                 (ds, "Anomaly Detection", False, "A", "A")]
        for t in md.TASKS[1:]:
            rows += [(ds, t, False, "A", "A"), (ds, t, False, "A", "C")]  # 50%
    rows[-1] = ("GoodsAD", md.TASKS[-1], False, "A", "A")  # GoodsAD の物体の分析だけ 100%
    table = accuracy_table(_frame(rows), "strict")
    assert table.loc["VisA", "Anomaly Detection"] == pytest.approx(75.0)
    assert table.loc["VisA", "Average"] == pytest.approx((75 + 50 * 6) / 7)
    assert table.loc["GoodsAD", "Average"] == pytest.approx((75 + 50 * 5 + 100) / 7)
    assert table.loc["Average", "Object Analysis"] == pytest.approx(75.0)
    assert table.loc["Average", "Average"] == pytest.approx(((75 + 300) / 7 + (75 + 350) / 7) / 2)
    assert list(table.index) == ["GoodsAD", "VisA", "Average"]


def test_accuracy_table_strict_vs_script():
    """CI-MS5: strict は読めない答えを不正解、script は分母から外す。"""
    rows = [("VisA", t, False, "A", "A") for t in md.TASKS[1:]]
    rows += [("VisA", "Anomaly Detection", True, "B", "B"), ("VisA", "Anomaly Detection", False, "A", "A")]
    rows += [("VisA", "Defect Analysis", False, "A", None)]
    strict = accuracy_table(_frame(rows), "strict")
    script = accuracy_table(_frame(rows), "script")
    assert strict.loc["VisA", "Defect Analysis"] == pytest.approx(50.0)
    assert script.loc["VisA", "Defect Analysis"] == pytest.approx(100.0)
    assert unparsed_rate(_frame(rows)) == pytest.approx(1 / 9)


def test_accuracy_table_empty_cell_is_none():
    """CI-MS6: 分母 0 のセルは None、None を含む平均も None（0 で埋めない）。"""
    rows = [("VisA", t, False, "A", "A") for t in md.TASKS[1:]]
    rows += [("VisA", "Anomaly Detection", False, "A", "A")]  # 良品が無い
    table = accuracy_table(_frame(rows), "strict")
    assert table.loc["VisA", "Anomaly Detection"] is None or pd.isna(table.loc["VisA", "Anomaly Detection"])
    assert pd.isna(table.loc["VisA", "Average"]) and pd.isna(table.loc["Average", "Average"])


def test_accuracy_table_rejects_bad_answer():
    """CI-MS2: 正解が A〜E 以外なら止める。"""
    with pytest.raises(ValueError):
        accuracy_table(_frame([("VisA", "Defect Analysis", False, "Z", "A")]))


def test_wilson_interval():
    """CI-MS7: 95% の Wilson 区間。n=0 は None。"""
    lo, hi = wilson_interval(75, 100)
    assert lo == pytest.approx(0.6569, abs=1e-3) and hi == pytest.approx(0.8245, abs=1e-3)
    assert wilson_interval(0, 0) is None


# ---------- invoke ----------

def test_response_from_anthropic_ignores_thinking():
    """CI-MB1: text ブロックだけを本文にする。usage を読む。thinking の数は extra に残す。"""
    body = {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Answer: C."}],
            "usage": {"input_tokens": 900, "output_tokens": 12}, "stop_reason": "end_turn"}
    r = response_from_anthropic(body, 1.5)
    assert r.raw_text == "Answer: C." and r.input_tokens == 900 and r.output_tokens == 12
    assert r.stop_reason == "end_turn" and r.extra["n_thinking_blocks"] == 1


# ---------- run（偽のモデル） ----------

class FakeConverse:
    def __init__(self, text="Answer: A.", in_tok=1000, out_tok=10):
        self.calls: list[ModelRequest] = []
        self.text, self.in_tok, self.out_tok = text, in_tok, out_tok

    def invoke(self, req):
        self.calls.append(req)
        return ModelResponse(raw_text=self.text, input_tokens=self.in_tok, output_tokens=self.out_tok, latency_s=0.1)


class FakeAnthropic:
    def __init__(self):
        self.calls = []

    def invoke(self, model_id, body):
        self.calls.append((model_id, body))
        return response_from_anthropic({"content": [{"type": "text", "text": "B"}],
                                        "usage": {"input_tokens": 800, "output_tokens": 3}})


def _setup(tmp_path: Path, n_per_cell=6, per_task=8):
    (tmp_path / "prompts").mkdir()
    shutil.copy(REPO / "prompts" / "mmad_gpt4o_instruction.txt", tmp_path / "prompts")
    mmad = synthetic_mmad(n_per_cell)
    split = md.sample_questions(md.question_table(mmad), per_task)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    split.to_csv(tmp_path / "data" / "splits" / "mmad_1050.csv", index=False, lineterminator="\n")
    return mmad, split


def _cfg(**kw):
    base = dict(name="t", split="data/splits/mmad_1050.csv", model_id="m", api="converse",
                region="ap-northeast-1", budget_usd=1.0)
    base.update(kw)
    return mrun.MmadConfig(**base)


PRICING = PricingTable([Price("m", "standard", 1.0, 5.0), Price("m", "batch", 0.5, 2.5)])
LOADER = lambda key: EncodedImage(key.encode(), "jpeg", 1, 1)  # noqa: E731


def test_select_trial_spreads_over_tasks():
    """CI-MR1: 試運転は等間隔に選ぶ。"""
    split = pd.DataFrame({"record_id": [f"m{i:04d}" for i in range(1050)]})
    picked = mrun.select_trial(split, 10)
    assert list(picked.index) == [0, 105, 210, 315, 420, 525, 630, 735, 840, 945]


def test_ondemand_trial_records_and_prompt(tmp_path):
    """CI-MR1、MR3: 試運転で10問、画像は [見本, 検査対象]、記録に正誤の材料と費用がある。"""
    mmad, split = _setup(tmp_path)
    model = FakeConverse()
    out = mrun.execute_ondemand(_cfg(), tmp_path, tmp_path / "runs", PRICING, model, limit=10,
                                mmad=mmad, image_loader=LOADER)
    assert out.name == "t__trial10"
    recs = mrun.load_records(out / "calls.jsonl")
    assert len(recs) == 10 and len(model.calls) == 10
    first = split.iloc[0]
    req = model.calls[0]
    assert [i.data.decode() for i in req.images] == [first.template, first.image_key]
    assert req.max_tokens == 200 and req.temperature == 0 and req.layout == "refs_first"
    assert recs[0]["pred"] == "A" and recs[0]["cost_usd"] == pytest.approx((1000 * 1 + 10 * 5) / 1e6)


def test_ondemand_resume_skips_done_and_checks_identity(tmp_path):
    """CI-MR2、MR3: 成功した問題は飛ばす。設定が違えば止める。"""
    mmad, _ = _setup(tmp_path)
    m1 = FakeConverse()
    mrun.execute_ondemand(_cfg(), tmp_path, tmp_path / "runs", PRICING, m1, limit=5, mmad=mmad, image_loader=LOADER)
    m2 = FakeConverse()
    mrun.execute_ondemand(_cfg(), tmp_path, tmp_path / "runs", PRICING, m2, limit=5, mmad=mmad, image_loader=LOADER)
    assert len(m2.calls) == 0
    with pytest.raises(ValueError, match="settings differ"):
        mrun.execute_ondemand(_cfg(image_max=768), tmp_path, tmp_path / "runs", PRICING, FakeConverse(),
                              limit=5, mmad=mmad, image_loader=LOADER)


def test_ondemand_stops_before_budget(tmp_path):
    """CI-C4: 予算を超える前に止める（記録は残る）。"""
    mmad, _ = _setup(tmp_path)
    model = FakeConverse(in_tok=100_000)  # 1呼び出し 0.10005 USD
    with pytest.raises(BudgetExceeded):
        mrun.execute_ondemand(_cfg(budget_usd=0.35), tmp_path, tmp_path / "runs", PRICING, model,
                              mmad=mmad, image_loader=LOADER)
    assert len(model.calls) == 3


def test_ondemand_anthropic_uses_messages_body(tmp_path):
    """CI-MP3: api=anthropic は Messages API の本文を invoke に渡す。"""
    mmad, _ = _setup(tmp_path)
    model = FakeAnthropic()
    out = mrun.execute_ondemand(_cfg(api="anthropic"), tmp_path, tmp_path / "runs", PRICING, model, limit=2,
                                mmad=mmad, image_loader=LOADER)
    model_id, body = model.calls[0]
    assert model_id == "m" and body["messages"][0]["content"][-1]["type"] == "text"
    assert mrun.load_records(out / "calls.jsonl")[0]["pred"] == "B"


def _write_batch_output(path: Path, split: pd.DataFrame, skip: set, errors: set, extra=()):
    lines = []
    for rid in split["record_id"]:
        if rid in skip:
            continue
        if rid in errors:
            lines.append({"recordId": rid, "modelInput": {}, "error": {"errorCode": 400, "errorMessage": "bad"}})
        else:
            lines.append({"recordId": rid, "modelInput": {}, "modelOutput": {
                "content": [{"type": "text", "text": "Answer: A."}], "usage": {"input_tokens": 1000, "output_tokens": 4}}})
    lines += list(extra)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")


def test_batch_prepare_and_collect(tmp_path):
    """CI-MB2、MB3: 入力は recordId と Messages API の本文。出力に無い問題・エラーは api_error として残す。費用はバッチの単価。"""
    mmad, split = _setup(tmp_path, n_per_cell=20, per_task=16)  # 112問（最小件数100以上）
    cfg = _cfg(api="anthropic", mode="batch")
    manifest = mrun.prepare_batch(cfg, tmp_path, tmp_path / "runs", mmad=mmad, image_loader=LOADER)
    assert manifest["n_records"] == 112
    lines = (tmp_path / "runs" / "t" / "batch_input.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["recordId"] == "m0001" and first["modelInput"]["messages"][0]["content"][0]["type"] == "image"
    out = tmp_path / "out.jsonl.out"
    _write_batch_output(out, split, skip={"m0002"}, errors={"m0003"})
    counts = mrun.collect_batch(cfg, tmp_path, tmp_path / "runs", [out], PRICING, "arn:aws:bedrock:x:job/abc")
    assert counts == {"ok": 110, "api_error": 1, "missing": 1}
    recs = {r["record_id"]: r for r in mrun.load_records(tmp_path / "runs" / "t" / "calls.jsonl")}
    assert recs["m0002"]["api_error"] == "missing in batch output"
    assert recs["m0001"]["cost_usd"] == pytest.approx((1000 * 0.5 + 4 * 2.5) / 1e6)
    assert recs["m0001"]["run_id"] == "abc"


def test_batch_prepare_refuses_under_minimum(tmp_path):
    """CI-MB2: 100件未満ならバッチの入力を作らない。"""
    mmad, _ = _setup(tmp_path)  # 56問
    with pytest.raises(ValueError, match="at least 100"):
        mrun.prepare_batch(_cfg(api="anthropic", mode="batch"), tmp_path, tmp_path / "runs", mmad=mmad, image_loader=LOADER)


def test_batch_collect_rejects_unknown_and_duplicate(tmp_path):
    """CI-MB3: 一覧に無い recordId、同じ recordId の重複は止める。"""
    mmad, split = _setup(tmp_path, n_per_cell=20, per_task=16)
    cfg = _cfg(api="anthropic", mode="batch")
    mrun.prepare_batch(cfg, tmp_path, tmp_path / "runs", mmad=mmad, image_loader=LOADER)
    out = tmp_path / "o.jsonl.out"
    _write_batch_output(out, split, set(), set(), extra=[{"recordId": "zzz", "modelOutput": {"content": []}}])
    with pytest.raises(ValueError, match="unknown"):
        mrun.collect_batch(cfg, tmp_path, tmp_path / "runs", [out], PRICING)
    _write_batch_output(out, split, set(), set(), extra=[{"recordId": "m0001", "modelOutput": {"content": []}}])
    with pytest.raises(ValueError, match="duplicate"):
        mrun.collect_batch(cfg, tmp_path, tmp_path / "runs", [out], PRICING)


# ---------- report ----------

def test_load_calls_prefers_last_success(tmp_path):
    """CI-MR4: record_id ごとに、成功があれば最後の成功、無ければ最後の失敗。"""
    d = tmp_path / "r"
    d.mkdir()
    base = {"run_name": "r", "qid": "q", "dataset": "VisA", "task": "Defect Analysis", "is_normal": False,
            "answer": "A", "cost_usd": 0.0}
    recs = [
        {**base, "record_id": "m1", "pred": None, "api_error": "Throttling"},
        {**base, "record_id": "m1", "pred": "A", "api_error": None},
        {**base, "record_id": "m1", "pred": None, "api_error": "Later failure"},
        {**base, "record_id": "m2", "pred": None, "api_error": "x"},
        {**base, "record_id": "m2", "pred": None, "api_error": "y"},
    ]
    (d / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    df = load_calls(d)
    assert list(df["record_id"]) == ["m1", "m2"]
    assert df.loc[0, "pred"] == "A" and df.loc[1, "api_error"] == "y"


def test_build_report_end_to_end(tmp_path):
    """CI-MS2〜MS5: 試運転の記録から4つの表を出せる。"""
    mmad, _ = _setup(tmp_path)
    out = mrun.execute_ondemand(_cfg(), tmp_path, tmp_path / "runs", PRICING, FakeConverse(), mmad=mmad,
                                image_loader=LOADER)
    paths = build_report([out], tmp_path / "rep")
    sub = pd.read_csv(paths["accuracy_by_subtask"])
    assert set(sub["count"]) == {"strict", "script"} and "Average" in sub.columns
    summ = pd.read_csv(paths["run_summary"])
    assert summ.loc[0, "n_questions"] == 56 and summ.loc[0, "n_api_error"] == 0
