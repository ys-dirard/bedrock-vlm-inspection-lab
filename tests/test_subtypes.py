"""異常の種類ごとの集計（docs/contracts.md の CI-ST1〜ST5）。"""

import json

import pandas as pd
import pytest

from vlmlab.subtypes import attach_subtype, build_subtype_report, subtype_rates, vlm_score


def split_df():
    rows = []
    for cat in ("box", "bag"):
        for sub, label, n in (("good", "normal", 4), ("logical_anomalies", "anomaly", 2),
                              ("structural_anomalies", "anomaly", 2)):
            for i in range(n):
                rows.append({"image_id": f"{cat}/{sub}/{i}", "category": cat, "label": label, "split": "test",
                             "path": f"{cat}/{sub}/{i}.png", "dataset": "mvtec_loco_ad", "subtype": sub})
    return pd.DataFrame(rows)


def records(preds):
    """preds: image_id -> (pred, confidence)。"""
    s = split_df()
    return pd.DataFrame([{"run_name": "r", "repeat": 1, "image_id": r.image_id, "category": r.category,
                          "label": r.label, "pred": preds.get(r.image_id, ("OK", 0.9))[0],
                          "confidence": preds.get(r.image_id, ("OK", 0.9))[1]} for r in s.itertuples()])


def test_attach_subtype_checks_ids_and_labels():
    """CI-ST1：一覧に無い image_id、正解の食い違いは止める。"""
    rec = records({})
    assert set(attach_subtype(rec, split_df()).subtype) == {"good", "logical_anomalies", "structural_anomalies"}
    with pytest.raises(ValueError, match="not in split"):
        attach_subtype(pd.concat([rec, rec.iloc[[0]].assign(image_id="zzz")]), split_df())
    with pytest.raises(ValueError, match="differs"):
        attach_subtype(rec.assign(label="normal"), split_df())
    bad = split_df()
    bad.loc[0, "subtype"] = "logical_anomalies"
    with pytest.raises(ValueError, match="exactly the normal"):
        attach_subtype(rec, bad)


def test_rates_per_subtype_and_scope():
    """CI-ST2・ST3：種類ごとの NG の割合（異常では再現率、good では誤検知率）。全品目と品目ごとの両方。"""
    preds = {"box/logical_anomalies/0": ("NG", 0.8), "box/structural_anomalies/0": ("NG", 0.9),
             "box/structural_anomalies/1": ("NG", 0.7), "bag/good/0": ("NG", 0.6)}
    df = attach_subtype(records(preds), split_df())
    df["score"] = vlm_score(df)
    t = subtype_rates(df).set_index(["scope", "subtype"])
    assert t.loc[("all", "logical_anomalies"), "ng_rate"] == 0.25
    assert t.loc[("all", "structural_anomalies"), "ng_rate"] == 0.5
    assert t.loc[("all", "good"), "ng_rate"] == 0.125
    assert t.loc[("all", "good"), "ng_rate_meaning"] == "false_positive_rate"
    assert t.loc[("box", "structural_anomalies"), "ng_rate"] == 1.0
    assert t.loc[("bag", "good"), "n_total"] == 4
    assert pd.isna(t.loc[("all", "good"), "auroc_vs_good"])


def test_parse_failures_leave_denominator():
    """CI-ST3：解析失敗は分母から除き、件数を書く（D3）。全部失敗なら割合は None。"""
    preds = {"box/logical_anomalies/0": (None, None), "box/logical_anomalies/1": (None, None),
             "bag/logical_anomalies/0": ("NG", 0.9)}
    df = attach_subtype(records(preds), split_df())
    df["score"] = vlm_score(df)
    t = subtype_rates(df).set_index(["scope", "subtype"])
    r = t.loc[("all", "logical_anomalies")]
    assert (r.n_total, r.n_valid, r.n_parse_fail, r.ng_rate) == (4, 2, 2, 0.5)
    assert pd.isna(t.loc[("box", "logical_anomalies"), "ng_rate"])


def test_auroc_against_all_good_in_scope():
    """CI-ST4：種類の異常と、同じ範囲の良品すべてで AUROC。不良らしさ = NG なら c、OK なら 1−c。"""
    preds = {f"{c}/logical_anomalies/{i}": ("NG", 0.9) for c in ("box", "bag") for i in range(2)}
    preds["bag/good/0"] = ("NG", 0.95)  # 良品1枚だけが論理的な異常より不良らしい
    df = attach_subtype(records(preds), split_df())
    df["score"] = vlm_score(df)
    t = subtype_rates(df).set_index(["scope", "subtype"])
    # 異常4枚 × 良品8枚の32組のうち、良品 0.95 が上回る4組だけ負け → 28/32
    assert t.loc[("all", "logical_anomalies"), "auroc_vs_good"] == pytest.approx(28 / 32)
    assert t.loc[("box", "logical_anomalies"), "auroc_vs_good"] == pytest.approx(1.0)
    # 構造的な異常はすべて OK 0.9（スコア 0.1）で、良品7枚と同点、1枚より下 → (0.5*7)/8
    assert t.loc[("all", "structural_anomalies"), "auroc_vs_good"] == pytest.approx(3.5 / 8)


def test_auroc_none_when_confidence_missing():
    """CI-ST4：有効な行にスコアの欠けがあれば AUROC は計算しない（理由を書く）。"""
    df = attach_subtype(records({"box/good/0": ("OK", None)}), split_df())
    df["score"] = vlm_score(df)
    r = subtype_rates(df).set_index(["scope", "subtype"]).loc[("all", "logical_anomalies")]
    assert pd.isna(r.auroc_vs_good) and r.auroc_reason == "score missing in some valid rows"


def test_build_report_from_run_dir(tmp_path):
    """CI-ST5：記録の raw_text から読み直して集計し、CSV と Markdown を出す（API は呼ばない）。"""
    s = split_df()
    split_csv = tmp_path / "split.csv"
    s.to_csv(split_csv, index=False)
    run = tmp_path / "runs" / "x"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("parse: {}\n", encoding="utf-8")
    with (run / "calls.jsonl").open("w", encoding="utf-8") as f:
        for r in s.itertuples():
            ng = r.subtype == "structural_anomalies"
            text = json.dumps({"judgement": "NG" if ng else "OK", "defect": "", "confidence": 0.8})
            f.write(json.dumps({"run_name": "x", "repeat": 1, "image_id": r.image_id, "category": r.category,
                                "label": r.label, "path": r.path, "raw_text": text, "api_error": None,
                                "cost_usd": 0.001, "latency_s": 1.0, "input_tokens": 10, "output_tokens": 5}) + "\n")
    paths = build_subtype_report([str(run)], split_csv, tmp_path / "out")
    t = pd.read_csv(paths["csv"]).set_index(["scope", "subtype"])
    assert t.loc[("all", "structural_anomalies"), "ng_rate"] == 1.0
    assert t.loc[("all", "logical_anomalies"), "ng_rate"] == 0.0
    assert t.loc[("all", "structural_anomalies"), "auroc_vs_good"] == 1.0
    assert "structural_anomalies" in paths["md"].read_text(encoding="utf-8")
