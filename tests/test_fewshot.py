"""12/3 の記事の追加分：不良品の見本（CI-D2）、layout labeled と cachePoint（CI-B3）、
不良品の見本の読み込みと再開（CI-R4、CI-R5）、キャッシュの集計（CI-M11）、判定の変化（CI-M12）。Bedrock は呼ばない。"""

import json

import pandas as pd
import pytest
import yaml
from PIL import Image

from vlmlab import metrics as M
from vlmlab.bedrock import ModelRequest, build_converse_kwargs
from vlmlab.cost import Price, PricingTable
from vlmlab.data import EncodedImage, defect_note, make_defect_refs
from vlmlab.report import build_changes, build_report, load_run
from vlmlab.run import RunConfig, execute, plan_run

# ---- CI-D2：不良品の見本 -------------------------------------------------------------


def anomaly_table():
    """pcb1 の test の不良品 20 枚：melt 8、missing 5、bent 4、scratch 3。"""
    types = ["melt"] * 8 + ["missing"] * 5 + ["bent"] * 4 + ["scratch"] * 3
    rows = [{"image_id": f"pcb1/anomaly/{i:03d}", "category": "pcb1", "label": "anomaly", "split": "test",
             "path": f"pcb1/A/{i:03d}.JPG"} for i in range(len(types))]
    table = pd.DataFrame(rows)
    defect_types = pd.DataFrame({"path": table["path"], "defect_type": types})
    return table, defect_types


def test_defect_refs_pick_most_common_distinct_types_and_skip_excluded():
    """CI-D2：種類の多い順に1枚ずつ。除外した画像は選ばない。"""
    table, types = anomaly_table()
    # melt を 8→2 枚に減らすと（6枚除外）、missing 5・bent 4・scratch 3 が上位3種類（melt 2 は入らない）
    exclude = {f"pcb1/anomaly/{i:03d}" for i in range(6)}
    out = make_defect_refs(table, types, exclude, categories=("pcb1",))
    assert list(out["defect_type"]) == ["missing", "bent", "scratch"]
    assert list(out["ref_rank"]) == [1, 2, 3]
    assert not set(out["image_id"]) & exclude
    assert list(out["note"]) == ["部品の欠落", "曲がり", "傷"]


def test_defect_refs_are_deterministic_regardless_of_input_order():
    """CI-D2：入力の並びによらず同じ結果"""
    table, types = anomaly_table()
    a = make_defect_refs(table, types, set(), categories=("pcb1",))
    b = make_defect_refs(table.sample(frac=1, random_state=1), types.sample(frac=1, random_state=2), set(),
                         categories=("pcb1",))
    assert a.equals(b)


def test_defect_refs_tie_is_broken_by_name_and_too_few_types_raise():
    table, types = anomaly_table()
    # bent 4 と missing 4 の同数 → 名前の順（bent が先）
    keep = table.index[~table.index.isin([8])]  # missing を1枚減らす
    out = make_defect_refs(table.loc[keep], types, set(), categories=("pcb1",))
    assert list(out["defect_type"]) == ["melt", "bent", "missing"]
    with pytest.raises(ValueError, match="defect types"):
        make_defect_refs(table, types, set(), n_per_category=5, categories=("pcb1",))


def test_defect_note_translates_combined_types_and_rejects_unknown():
    assert defect_note("bubble,discolor") == "気泡・変色"
    with pytest.raises(ValueError, match="unknown"):
        defect_note("crack")


# ---- CI-B3：layout labeled と cachePoint ---------------------------------------------


def _req(**kw):
    g, d, t = EncodedImage(b"g", "jpeg", 1, 1), EncodedImage(b"d", "jpeg", 1, 1), EncodedImage(b"t", "jpeg", 1, 1)
    base = dict(model_id="m", images=[g, d, t], text="inst", layout="labeled",
                ref_labels=("良品の見本:", "不良品の見本1（傷）:"), target_label="検査対象:")
    return ModelRequest(**{**base, **kw})


def test_labeled_layout_order_without_cache_point():
    """CI-B3：[説明, 見本]… → 指示 → 検査対象の説明 → 検査対象"""
    c = build_converse_kwargs(_req())["messages"][0]["content"]
    kinds = [("image:" + b["image"]["source"]["bytes"].decode()) if "image" in b else
             "cachePoint" if "cachePoint" in b else b["text"] for b in c]
    assert kinds == ["良品の見本:", "image:g", "不良品の見本1（傷）:", "image:d", "inst", "検査対象:", "image:t"]


def test_labeled_layout_places_cache_point_after_instruction_without_ttl():
    """CI-B3：cachePoint は指示の直後。ttl は書かない"""
    c = build_converse_kwargs(_req(cache_point=True))["messages"][0]["content"]
    i = next(k for k, b in enumerate(c) if "cachePoint" in b)
    assert c[i] == {"cachePoint": {"type": "default"}}
    assert c[i - 1] == {"text": "inst"} and c[i + 1] == {"text": "検査対象:"}
    # cachePoint の有無以外は同じ並び
    without = build_converse_kwargs(_req())["messages"][0]["content"]
    assert [b for b in c if "cachePoint" not in b] == without


@pytest.mark.parametrize("kw", [
    {"ref_labels": ("良品の見本:",)},           # 見本の数と合わない
    {"target_label": None},
    {"layout": "refs_first"},                   # labeled 以外で ref_labels / cache_point
    {"layout": "refs_first", "ref_labels": None, "target_label": None, "cache_point": True},
])
def test_labeled_layout_rejects_inconsistent_requests(kw):
    with pytest.raises(ValueError):
        build_converse_kwargs(_req(**kw))


# ---- CI-R4・R5：設定、不良品の見本の読み込み、再開 ------------------------------------


class EchoModel:
    def __init__(self):
        self.requests = []

    def invoke(self, req):
        from vlmlab.bedrock import ModelResponse
        self.requests.append(req)
        n = len(self.requests)
        return ModelResponse(raw_text=json.dumps({"judgement": "NG", "confidence": 0.8}), input_tokens=300,
                             output_tokens=20, cache_read_tokens=1000 if n > 1 else 0,
                             cache_write_tokens=0 if n > 1 else 1000, latency_s=0.5)


@pytest.fixture
def fewshot_lab(tmp_path):
    visa = tmp_path / "visa"
    rows, defects = [], []
    for cat in ("pcb1", "capsules"):
        for label, folder, n in (("normal", "Normal", 4), ("anomaly", "Anomaly", 6)):
            d = visa / cat / "Data" / "Images" / folder
            d.mkdir(parents=True)
            for i in range(n):
                p = d / f"{i:03d}.JPG"
                Image.new("RGB", (16, 16), (i * 30, 0, 0)).save(p, "JPEG")
                r = {"image_id": f"{cat}/{label}/{i:03d}", "category": cat, "label": label, "split": "test",
                     "path": f"{cat}/Data/Images/{folder}/{i:03d}.JPG"}
                (defects if (label == "anomaly" and i >= 3) else rows).append(r)
    (tmp_path / "data").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "split.csv", index=False)
    goods = [dict(r, ref_rank=1) for r in rows if r["label"] == "normal" and r["image_id"].endswith("000")]
    pd.DataFrame(goods).to_csv(tmp_path / "data" / "refs.csv", index=False)
    dref = [dict(r, ref_rank=k % 3 + 1, defect_type="scratch", note=f"傷{k % 3 + 1}") for k, r in enumerate(defects)]
    pd.DataFrame(dref).to_csv(tmp_path / "data" / "defect_refs.csv", index=False)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "v2.txt").write_text('判定して {"judgement": "OK"}', encoding="utf-8")
    cfg = {
        "name": "c3", "split": "data/split.csv", "refs": "data/refs.csv", "n_refs": 1,
        "defect_refs": "data/defect_refs.csv", "n_defect_refs": 3, "layout": "labeled",
        "labels": {"good": "良品の見本:", "defect": "不良品の見本{{n}}（{{note}}）:", "target": "検査対象:"},
        "cache_point": True, "model_id": "m", "region": "r", "prompt": "prompts/v2.txt",
        "repeats": 1, "budget_usd": 1.0,
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    pricing = PricingTable([Price("m", "standard", 0.396, 3.311, 0.099, 0.0)])
    return tmp_path, visa, pricing, cfg


def test_labeled_run_sends_good_then_defects_then_target_and_records_cache(fewshot_lab):
    """CI-R4：良品 → 不良品（rank の順、説明つき）→ 検査対象。費用はキャッシュの単価で数える（CI-C1）"""
    base, visa, pricing, _ = fewshot_lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    model = EchoModel()
    execute(plan, model, pricing, visa, base, log=lambda _: None)
    r0 = model.requests[0]
    assert len(r0.images) == 5
    assert r0.ref_labels == ("良品の見本:", "不良品の見本1（傷1）:", "不良品の見本2（傷2）:", "不良品の見本3（傷3）:")
    assert r0.target_label == "検査対象:" and r0.cache_point is True
    recs = [json.loads(l) for l in (plan.out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs[0]["n_defect_refs"] == 3 and recs[0]["cache_point"] is True and recs[0]["defect_refs_sha"]
    assert recs[0]["cost_usd"] == pytest.approx((300 * 0.396 + 20 * 3.311 + 1000 * 0.0) / 1e6)
    assert recs[1]["cost_usd"] == pytest.approx((300 * 0.396 + 20 * 3.311 + 1000 * 0.099) / 1e6)


def test_defect_refs_overlapping_test_images_are_refused(fewshot_lab):
    """CI-R4：不良品の見本が検査対象と重なれば呼ぶ前に止める"""
    base, _, _, _ = fewshot_lab
    split = pd.read_csv(base / "data" / "split.csv")
    d = pd.read_csv(base / "data" / "defect_refs.csv")
    pd.concat([split, d[split.columns]]).to_csv(base / "data" / "split.csv", index=False)
    with pytest.raises(ValueError, match="overlap"):
        plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")


def test_not_enough_defect_refs_are_refused(fewshot_lab):
    base, _, _, _ = fewshot_lab
    cfg = RunConfig.load(base / "c.yaml")
    cfg.n_defect_refs = 4
    with pytest.raises(ValueError, match="not enough defect refs"):
        plan_run(cfg, base, base / "runs")


def test_resume_refused_when_cache_point_or_defects_change(fewshot_lab):
    """CI-R5：cachePoint の有無・不良品の見本の枚数が違う記録には続けて書かない"""
    base, visa, pricing, _ = fewshot_lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    execute(plan, EchoModel(), pricing, visa, base, log=lambda _: None)
    for change in ({"cache_point": False}, {"n_defect_refs": 1}):
        cfg = RunConfig.load(base / "c.yaml")
        for k, v in change.items():
            setattr(cfg, k, v)
        model = EchoModel()
        with pytest.raises(RuntimeError, match=list(change)[0]):
            execute(plan_run(cfg, base, base / "runs"), model, pricing, visa, base, log=lambda _: None)
        assert model.requests == []


@pytest.mark.parametrize("change", [
    {"layout": "refs_first"},                              # n_defect_refs・labels・cache_point は labeled だけ
    {"labels": {"good": "a", "target": "c"}},              # defect が無い
    {"defect_refs": None},                                 # n_defect_refs > 0 なのに一覧が無い
])
def test_config_rejects_inconsistent_fewshot_settings(fewshot_lab, change):
    base, _, _, cfg = fewshot_lab
    p = base / "bad.yaml"
    p.write_text(yaml.safe_dump({**cfg, **change}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError):
        RunConfig.load(p)


def test_report_writes_cache_usage(fewshot_lab):
    base, visa, pricing, _ = fewshot_lab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    execute(plan, EchoModel(), pricing, visa, base, log=lambda _: None)
    paths = build_report([plan.out_dir], base / "results", tag="t")
    cu = pd.read_csv(paths["cache_usage"])
    row = cu[cu.category == "all"].iloc[0]
    n = len(load_run(plan.out_dir))
    assert row["n_calls"] == n and row["n_cache_read_tokens_positive"] == n - 1
    assert row["n_cache_write_tokens_positive"] == 1
    assert row["total_input_tokens_mean"] == pytest.approx(1300)
    summary = pd.read_csv(paths["summary"])
    assert summary.loc[0, "cache_read_tokens_mean"] == pytest.approx(1000 * (n - 1) / n)


# ---- CI-M11：キャッシュの集計 ----------------------------------------------------------


def test_cache_stats_counts_positive_calls_and_total_input():
    df = pd.DataFrame({"input_tokens": [1200, 300, 300], "cache_read_tokens": [0, 900, 900],
                       "cache_write_tokens": [900, 0, 0]})
    s = M.cache_stats(df)
    assert s["n_calls"] == 3
    assert s["cache_read_tokens_mean"] == pytest.approx(600) and s["n_cache_read_tokens_positive"] == 2
    assert s["cache_write_tokens_mean"] == pytest.approx(300) and s["n_cache_write_tokens_positive"] == 1
    assert s["total_input_tokens_mean"] == pytest.approx((2100 + 1200 + 1200) / 3)


# ---- CI-M12：判定の変化 ----------------------------------------------------------------


def runs(spec):
    """spec: {image_id: (label, [pred, ...])} → 回ごとの行"""
    return pd.DataFrame([{"image_id": i, "category": i.split("/")[0], "label": l, "repeat": r, "pred": p}
                         for i, (l, ps) in spec.items() for r, p in enumerate(ps, 1)])


def test_changed_judgements_classifies_each_image():
    base = runs({
        "pcb1/a": ("anomaly", ["OK", "OK", "OK"]),   # → NG：直った
        "pcb1/b": ("normal", ["OK", "OK", "OK"]),    # → NG：悪くなった
        "pcb1/c": ("anomaly", ["NG", "NG", "NG"]),   # 変わらない
        "cap/d": ("normal", ["OK", "NG", "OK"]),     # 基準が割れている
        "cap/e": ("anomaly", ["NG", "NG", "NG"]),    # → 解析失敗を含む
        "cap/f": ("normal", ["NG", "NG", "NG"]),     # → OK：直った
    })
    other = runs({
        "pcb1/a": ("anomaly", ["NG", "NG"]),
        "pcb1/b": ("normal", ["NG", "NG"]),
        "pcb1/c": ("anomaly", ["NG", "NG"]),
        "cap/d": ("normal", ["OK", "OK"]),
        "cap/e": ("anomaly", ["NG", None]),
        "cap/f": ("normal", ["OK", "OK"]),
    })
    ch = M.changed_judgements(base, other).set_index("image_id")
    assert ch.loc["pcb1/a", "change"] == "OK→NG" and ch.loc["pcb1/a", "effect"] == "fixed"
    assert ch.loc["pcb1/b", "change"] == "OK→NG" and ch.loc["pcb1/b", "effect"] == "broken"
    assert ch.loc["pcb1/c", "change"] == "same"
    assert ch.loc["cap/d", "change"] == "unstable" and ch.loc["cap/e", "change"] == "unstable"
    assert ch.loc["cap/e", "other_preds"] == "NG;NA"
    assert ch.loc["cap/f", "change"] == "NG→OK" and ch.loc["cap/f", "effect"] == "fixed"
    c = M.change_counts(ch.reset_index())
    assert c == {"n_images": 6, "n_same": 1, "n_ok_to_ng": 2, "n_ng_to_ok": 1, "n_unstable": 2,
                 "n_fixed": 2, "n_broken": 1}


def test_changed_judgements_refuses_different_image_sets():
    base = runs({"a/1": ("normal", ["OK"]), "a/2": ("normal", ["OK"])})
    other = runs({"a/1": ("normal", ["OK"])})
    with pytest.raises(ValueError, match="image sets differ"):
        M.changed_judgements(base, other)


def test_build_changes_writes_rows_and_counts(fewshot_lab):
    base, visa, pricing, cfg = fewshot_lab
    plan3 = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    execute(plan3, EchoModel(), pricing, visa, base, log=lambda _: None)
    p0 = base / "c0.yaml"
    p0.write_text(yaml.safe_dump({**cfg, "name": "c0", "n_defect_refs": 0, "cache_point": False},
                                 allow_unicode=True), encoding="utf-8")
    plan0 = plan_run(RunConfig.load(p0), base, base / "runs")

    class AllOK(EchoModel):
        def invoke(self, req):
            r = super().invoke(req)
            r.raw_text = json.dumps({"judgement": "OK", "confidence": 0.9})
            return r

    execute(plan0, AllOK(), pricing, visa, base, log=lambda _: None)
    paths = build_changes(plan0.out_dir, [plan3.out_dir], base / "results")
    counts = pd.read_csv(paths["counts"])
    allrow = counts[counts.category == "all"].iloc[0]
    # 検査対象：良品 4 枚・不良品 3 枚 × 2 品目。全部 OK → 全部 NG
    assert allrow["n_ok_to_ng"] == 14 and allrow["n_fixed"] == 6 and allrow["n_broken"] == 8
    assert len(pd.read_csv(paths["changes"])) == 14
