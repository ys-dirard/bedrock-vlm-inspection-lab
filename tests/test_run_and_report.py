"""run.py と report.py を、偽物のモデルで通しで動かす（Bedrock は呼ばない）。"""

import io
import json

import pandas as pd
import pytest
from PIL import Image

from vlmlab.bedrock import ModelResponse, build_converse_kwargs, response_from_converse, ModelRequest
from vlmlab.cost import Price, PricingTable
from vlmlab.data import EncodedImage
from vlmlab.report import build_report, load_run
from vlmlab.run import RunConfig, execute, fill_template, plan_run, select_limit


class FakeModel:
    """検査対象の画像（最後の画像）の色で正解を見分け、決まった応答を返す。

    lab の画像は、不良品だけ緑の成分を 200 にしてある。flip に入れた番号の呼び出しは判定を逆にする。
    """

    def __init__(self, fail_on=None, flip=None):
        self.requests = []
        self.fail_on = fail_on or set()
        self.flip = flip or set()

    def invoke(self, req):
        self.requests.append(req)
        n = len(self.requests)
        if n in self.fail_on:
            return ModelResponse(raw_text="判定できません", input_tokens=700, output_tokens=10, latency_s=1.0)
        with Image.open(io.BytesIO(req.images[-1].data)) as im:
            is_anomaly = im.convert("RGB").getpixel((0, 0))[1] > 100
        j = "NG" if is_anomaly else "OK"
        if n in self.flip:
            j = "OK" if j == "NG" else "NG"
        return ModelResponse(raw_text=json.dumps({"judgement": j, "confidence": 0.7 + (n % 3) / 10}),
                             input_tokens=700, output_tokens=40, latency_s=1.0 + n / 100)


@pytest.fixture
def lab(tmp_path):
    visa = tmp_path / "visa"
    rows = []
    for cat in ("pcb1", "capsules"):
        for label, folder in (("normal", "Normal"), ("anomaly", "Anomaly")):
            d = visa / cat / "Data" / "Images" / folder
            d.mkdir(parents=True)
            for i in range(3):
                p = d / f"{i:03d}.JPG"
                Image.new("RGB", (64, 32), (i * 40, 200 if label == "anomaly" else 0, 0)).save(p, "JPEG")
                rows.append({"image_id": f"{cat}/{label}/{i:03d}", "category": cat, "label": label,
                             "split": "test", "path": f"{cat}/Data/Images/{folder}/{i:03d}.JPG"})
    (tmp_path / "data").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "split.csv", index=False)
    refs = [{"image_id": r["image_id"], "ref_rank": k + 1, **{x: r[x] for x in ("category", "label", "split", "path")}}
            for cat in ("pcb1", "capsules")
            for k, r in enumerate([r for r in rows if r["category"] == cat and r["label"] == "normal"])]
    pd.DataFrame(refs).to_csv(tmp_path / "data" / "refs.csv", index=False)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "v1.txt").write_text('{{item}} を検査し {"judgement": "OK"} の形で返す', encoding="utf-8")
    cfg = {
        "name": "t", "split": "data/split.csv", "refs": "data/refs.csv", "n_refs": 1,
        "model_id": "m", "region": "ap-northeast-1", "prompt": "prompts/v1.txt",
        "prompt_vars": {"pcb1": {"item": "基板"}, "capsules": {"item": "カプセル"}},
        "repeats": 2, "budget_usd": 1.0,
    }
    (tmp_path / "c.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    pricing = PricingTable([Price("m", "standard", 0.33, 2.75)])
    return tmp_path, visa, pricing


def test_run_writes_records_and_resumes(lab):
    base, visa, pricing = lab
    cfg = RunConfig.load(base / "c.yaml")
    plan = plan_run(cfg, base, base / "runs")
    model = FakeModel(fail_on={3})
    s = execute(plan, model, pricing, visa, base, log=lambda _: None)
    assert s["called"] == 24 and s["stopped"] is None
    # 見本1枚 + 検査対象。品目ごとに差し替えたプロンプト
    assert len(model.requests[0].images) == 2
    assert model.requests[0].text.startswith(("基板", "カプセル"))
    recs = [json.loads(l) for l in (plan.out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {"run_name", "run_id", "repeat", "image_id", "category", "label", "model_id",
            "service_tier_requested", "service_tier_resolved", "prompt_version", "n_refs", "image_long_edge",
            "raw_text", "parsed", "parse_error", "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "latency_s", "stop_reason", "cost_usd", "timestamp"}
    assert keys <= set(recs[0])
    assert recs[2]["parsed"] is None and recs[2]["parse_error"]
    assert recs[0]["cost_usd"] == pytest.approx((700 * 0.33 + 40 * 2.75) / 1e6)
    assert recs[0]["timestamp"].endswith("+09:00")
    # 再開：全部終わっているので呼ばない
    s2 = execute(plan, FakeModel(), pricing, visa, base, log=lambda _: None)
    assert s2["called"] == 0 and s2["skipped"] == 24


def test_api_errors_are_retried_on_resume(lab):
    base, visa, pricing = lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")

    class Flaky(FakeModel):
        def invoke(self, req):
            r = super().invoke(req)
            if len(self.requests) == 1:
                return ModelResponse(raw_text=None, input_tokens=0, output_tokens=0, api_error="ThrottlingException: x")
            return r

    execute(plan, Flaky(), pricing, visa, base, log=lambda _: None)
    s2 = execute(plan, FakeModel(), pricing, visa, base, log=lambda _: None)
    assert s2["called"] == 1
    df = load_run(plan.out_dir)
    assert len(df) == 24 and df.attrs["n_api_error"] == 1


def test_budget_stops_run(lab):
    """CI-C4：予算で止まる"""
    base, visa, pricing = lab
    cfg = RunConfig.load(base / "c.yaml")
    cfg.budget_usd = 0.0012  # 1呼び出し 0.000341 → 3回で 0.001023、4回目の前に止まる
    plan = plan_run(cfg, base, base / "runs")
    s = execute(plan, FakeModel(), pricing, visa, base, log=lambda _: None)
    assert s["called"] == 3 and s["stopped"]
    assert s["spent_usd"] <= cfg.budget_usd


def test_null_price_refuses_to_start(lab):
    """CI-C5"""
    base, visa, _ = lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    model = FakeModel()
    with pytest.raises(RuntimeError, match="null"):
        execute(plan, model, PricingTable([Price("m", "standard", None, 2.75)]), visa, base, log=lambda _: None)
    assert model.requests == []


def test_report_outputs(lab):
    base, visa, pricing = lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    # 1回目：解析失敗1件、判定の反転2件。2回目は全部正しい
    execute(plan, FakeModel(fail_on={1}, flip={2, 3}), pricing, visa, base, log=lambda _: None)
    paths = build_report([plan.out_dir], base / "results", tag="summary_test")
    summary = pd.read_csv(paths["summary"])
    assert summary.loc[0, "n_total_sum"] == 24 and summary.loc[0, "n_parse_fail_sum"] == 1
    assert summary.loc[0, "f1_max"] == 1.0 and summary.loc[0, "f1_min"] < 1.0
    assert summary.loc[0, "agreement_rate"] == pytest.approx(9 / 12)
    assert "解析失敗" in paths["summary_md"].read_text(encoding="utf-8")
    biz = pd.read_csv(paths["business_metric"])
    assert list(biz["target_miss_rate"]) == [0.02, 0.05]
    assert biz["pass_rate_max"].iloc[0] == 1.0  # 2回目は全部正しいので、良品は全部通る
    wrong = pd.read_csv(paths["misclassified"])
    assert len(wrong) == 3


def test_resume_counts_previous_cost_and_stops_immediately(lab):
    """CI-C6：再開のとき、既にある記録の費用を累計に入れる"""
    base, visa, pricing = lab
    cfg = RunConfig.load(base / "c.yaml")
    cfg.budget_usd = 0.0012
    plan = plan_run(cfg, base, base / "runs")
    s = execute(plan, FakeModel(), pricing, visa, base, log=lambda _: None)
    assert s["called"] == 3
    model = FakeModel()
    s2 = execute(plan, model, pricing, visa, base, log=lambda _: None)
    assert s2["called"] == 0 and s2["stopped"] and model.requests == []


def test_resume_with_different_config_is_refused(lab):
    """CI-R1"""
    base, visa, pricing = lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    execute(plan, FakeModel(), pricing, visa, base, log=lambda _: None)
    (base / "prompts" / "v1.txt").write_text("{{item}} 変えた", encoding="utf-8")
    model = FakeModel()
    with pytest.raises(RuntimeError, match="prompt_sha"):
        execute(plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs"), model, pricing, visa, base,
                log=lambda _: None)
    assert model.requests == []


def test_missing_prompt_vars_fails_before_any_call(lab):
    """CI-R2"""
    base, _, _ = lab
    cfg = RunConfig.load(base / "c.yaml")
    cfg.prompt_vars = {"pcb1": {"item": "基板"}}
    with pytest.raises(ValueError, match="unfilled"):
        plan_run(cfg, base, base / "runs")


def test_load_run_keeps_last_success_per_key(lab):
    """CI-R3"""
    base, visa, pricing = lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    execute(plan, FakeModel(), pricing, visa, base, log=lambda _: None)
    jsonl = plan.out_dir / "calls.jsonl"
    recs = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    dup = dict(recs[0], raw_text=json.dumps({"judgement": "OK" if recs[0]["label"] == "anomaly" else "NG",
                                             "confidence": 0.9}))
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dup, ensure_ascii=False) + "\n")
    df = load_run(plan.out_dir)
    assert len(df) == 24
    row = df[(df.image_id == recs[0]["image_id"]) & (df.repeat == recs[0]["repeat"])]
    assert row["confidence"].iloc[0] == 0.9  # 後から足した記録が使われる


def test_trial_limit_takes_each_group_in_turn():
    split = pd.DataFrame({"category": ["a"] * 4 + ["b"] * 4, "label": ["normal", "normal", "anomaly", "anomaly"] * 2,
                          "image_id": list("12345678")})
    picked = select_limit(split, 4)
    assert sorted(zip(picked.category, picked.label)) == sorted({("a", "normal"), ("a", "anomaly"), ("b", "normal"), ("b", "anomaly")})


def test_fill_template_rejects_unfilled():
    assert fill_template('{{x}} {"a": 1}', {"x": "y"}) == 'y {"a": 1}'
    with pytest.raises(ValueError):
        fill_template("{{x}}", {})


def test_converse_kwargs_order_and_tier():
    img = EncodedImage(b"x", "jpeg", 1, 1)
    req = ModelRequest(model_id="m", images=[img, img], text="t", service_tier="flex", max_tokens=300)
    k = build_converse_kwargs(req)
    content = k["messages"][0]["content"]
    assert [list(c)[0] for c in content] == ["image", "image", "text"]
    assert k["serviceTier"] == {"type": "flex"}
    assert "serviceTier" not in build_converse_kwargs(ModelRequest(model_id="m", images=[], text="t"))
    r = response_from_converse({"output": {"message": {"content": [{"text": "{}"}]}},
                                "usage": {"inputTokens": 5, "outputTokens": 2}, "stopReason": "end_turn",
                                "serviceTier": {"type": "default"}}, 0.5, 1)
    assert (r.raw_text, r.input_tokens, r.output_tokens, r.service_tier_resolved) == ("{}", 5, 2, "default")


def test_converse_kwargs_target_first_layout_and_top_p():
    """AWS サンプルの並び：検査対象 → 指示 → ref_note → 見本（1通の user メッセージ）"""
    ref, target = EncodedImage(b"r", "jpeg", 1, 1), EncodedImage(b"t", "jpeg", 1, 1)
    req = ModelRequest(model_id="m", images=[ref, target], text="inst", layout="target_first",
                       ref_note="This is the reference image.", top_p=0.1, temperature=0.1, max_tokens=2500)
    k = build_converse_kwargs(req)
    assert len(k["messages"]) == 1 and k["messages"][0]["role"] == "user"
    c = k["messages"][0]["content"]
    assert c[0]["image"]["source"]["bytes"] == b"t"
    assert c[1] == {"text": "inst"}
    assert c[2] == {"text": "This is the reference image."}
    assert c[3]["image"]["source"]["bytes"] == b"r"
    assert k["inferenceConfig"] == {"temperature": 0.1, "maxTokens": 2500, "topP": 0.1}
    # topP を指定しなければ送らない
    k0 = build_converse_kwargs(ModelRequest(model_id="m", images=[target], text="t"))
    assert "topP" not in k0["inferenceConfig"]


def test_converse_kwargs_rejects_bad_layout():
    img = EncodedImage(b"x", "jpeg", 1, 1)
    with pytest.raises(ValueError):
        build_converse_kwargs(ModelRequest(model_id="m", images=[img], text="t", layout="nope"))
    with pytest.raises(ValueError):  # ref_note は target_first のときだけ
        build_converse_kwargs(ModelRequest(model_id="m", images=[img], text="t", ref_note="x"))


@pytest.mark.parametrize("extra", [
    {"layout": "nope"},
    {"ref_note": "x"},                              # layout が refs_first のまま
    {"inference": {"temperature": 0, "topK": 20}},  # topK は additional_fields に書く
])
def test_config_rejects_invalid_layout_and_inference(tmp_path, extra):
    import yaml
    d = {"name": "x", "split": "s.csv", "model_id": "m", "region": "r", "prompt": "p.txt", "budget_usd": 1.0, **extra}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(ValueError):
        RunConfig.load(p)
