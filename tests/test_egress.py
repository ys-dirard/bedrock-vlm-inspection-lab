"""egress.py・egress_report.py・run.py の text_only（12/13 の記事。CI-A6、CI-E1〜E8）。Bedrock は呼ばない。"""

import json

import numpy as np
import pandas as pd
import pytest

from vlmlab.anomalydino import load_map, map_filename, save_map
from vlmlab.bedrock import ModelResponse
from vlmlab.cost import Price, PricingTable
from vlmlab.egress import (Calibration, calibrate, check_calib, load_calibrations, make_calib_split, measure,
                           save_calibrations, score_level3, score_rank_text, verbalize)
from vlmlab.egress_report import anomalydino_rows, model_row
from vlmlab.run import RunConfig, execute, plan_run


def _row(image_id, label="normal", split="train"):
    cat = image_id.split("/")[0]
    return {"image_id": image_id, "category": cat, "label": label, "split": split, "path": image_id + ".JPG"}


# ---------------------------------------------------------------- CI-A6

def test_map_roundtrip_and_filename(tmp_path):
    d = np.arange(6, dtype=float) / 10
    m = np.array([True, False, True, True, True, False])
    p = tmp_path / "maps" / map_filename("pcb1/normal/0001")
    assert p.name == "pcb1__normal__0001.npz"
    save_map(p, d, m, [2, 3])
    dist, mask = load_map(p)
    assert dist.shape == (2, 3) and mask.shape == (2, 3)
    np.testing.assert_allclose(dist.ravel(), d, atol=1e-7)
    assert (mask.ravel() == m).all()
    with pytest.raises(ValueError):
        save_map(p, d, m, [4, 4])
    with pytest.raises(ValueError):
        map_filename("a__b")


# ---------------------------------------------------------------- CI-E1

def test_calib_split_excludes_refs_and_rejects_overlap():
    knn = pd.DataFrame([_row("pcb1/normal/1"), _row("pcb1/normal/2"), _row("pcb1/normal/3"),
                        _row("pcb1/anomaly/9", "anomaly", "test")])
    refs = pd.DataFrame([{**_row("pcb1/normal/1"), "ref_rank": 1}])
    test = pd.DataFrame([_row("pcb1/normal/5", split="test"), _row("pcb1/anomaly/6", "anomaly", "test")])
    c = make_calib_split(knn, refs, test)
    assert c["image_id"].tolist() == ["pcb1/normal/2", "pcb1/normal/3"]
    with pytest.raises(ValueError, match="overlaps with test"):
        check_calib(pd.DataFrame([_row("pcb1/normal/5")]), refs, test)
    with pytest.raises(ValueError, match="train normals"):
        check_calib(pd.DataFrame([_row("pcb1/anomaly/7", "anomaly")]), refs, test)
    with pytest.raises(ValueError, match="no calibration images"):
        check_calib(c, refs, pd.concat([test, pd.DataFrame([_row("capsules/normal/1", split="test")])]))


# ---------------------------------------------------------------- CI-E2

def test_calibrate_uses_foreground_only():
    d1 = np.array([[0.1, 0.2], [0.3, 9.0]])
    m1 = np.array([[True, True], [True, False]])  # 9.0 は背景なので入らない
    cal = calibrate("pcb1", [(d1, m1)], [0.25], 0.5)
    assert cal.patch_threshold == pytest.approx(0.2)
    assert cal.n_images == 1 and cal.scores == (0.25,)
    with pytest.raises(ValueError):
        calibrate("pcb1", [(d1, m1)], [0.1, 0.2], 0.5)
    with pytest.raises(ValueError):
        calibrate("pcb1", [(d1, m1)], [0.1], 1.0)


# ---------------------------------------------------------------- CI-E3

def test_measure_regions_position_shape():
    dist = np.zeros((9, 12))
    mask = np.ones((9, 12), dtype=bool)
    dist[0, 0:2] = 1.0           # 左上に2パッチ
    dist[6:9, 9] = 1.0           # 右下に縦長の3パッチ（最大）
    dist[4, 6] = 1.0             # 背景にする
    mask[4, 6] = False
    m = measure(dist, mask, 0.5)
    assert m.n_regions == 2
    assert m.largest_area == 3
    assert (m.largest_row, m.largest_col) == (2, 2)
    assert m.largest_aspect == pytest.approx(3.0)
    assert m.n_foreground == 107
    assert m.total_area_frac == pytest.approx(5 / 107)
    assert m.largest_area_frac == pytest.approx(3 / 107)


def test_measure_diagonal_is_one_region_and_none():
    dist = np.zeros((6, 6))
    dist[2, 2] = dist[3, 3] = 1.0  # 8近傍でつながる
    m = measure(dist, np.ones((6, 6), dtype=bool), 0.5)
    assert m.n_regions == 1 and m.largest_area == 2 and (m.largest_row, m.largest_col) == (1, 1)
    z = measure(np.zeros((6, 6)), np.ones((6, 6), dtype=bool), 0.5)
    assert z.n_regions == 0 and z.largest_row is None and z.total_area_frac == 0.0


# ---------------------------------------------------------------- CI-E4

def _cal(n=10):
    return Calibration("pcb1", n, 0.999, 0.5, tuple(float(x) for x in np.linspace(0.1, 0.2, n)))


def test_levels_and_rank_text():
    cal = _cal()
    assert score_level3(0.12, cal) == "学習用の良品と同じくらい"
    assert score_level3(0.195, cal) == "学習用の良品の上位10%に入る高さ"
    assert score_level3(0.21, cal) == "学習用の良品のどれよりも高い"
    assert score_rank_text(0.2, cal) == "学習用の良品10枚の中で上位10%に入る高さ"   # 0.2 以上は1枚
    assert score_rank_text(0.1, cal) == "学習用の良品10枚の中で上位100%に入る高さ"
    assert "わずかに上回る" in score_rank_text(0.21, cal)
    assert "大きく上回る" in score_rank_text(0.31, cal)


def test_verbalize_has_no_raw_numbers():
    cal = _cal()
    dist = np.zeros((9, 12))
    dist[0:1, 0:4] = 0.987654
    m = measure(dist, np.ones((9, 12), dtype=bool), 0.5)
    coarse = verbalize(0.25, m, cal, "coarse")
    fine = verbalize(0.25, m, cal, "fine")
    assert coarse == "異常度: 学習用の良品のどれよりも高い\n良品と違って見える領域: あり"
    assert "1か所" in fine and "左上" in fine and "細長い" in fine and "大きさは大きい" in fine  # 4/108 = 3.7%
    for t in (coarse, fine):
        assert "0.98" not in t and "0.25" not in t
    with pytest.raises(ValueError):
        verbalize(0.25, m, cal, "medium")


def test_calibration_json_roundtrip(tmp_path):
    cals = {"pcb1": _cal()}
    save_calibrations(cals, tmp_path / "c.json")
    assert load_calibrations(tmp_path / "c.json") == cals


# ---------------------------------------------------------------- CI-E6（run.py の text_only）

class TextModel:
    def __init__(self):
        self.requests = []

    def invoke(self, req):
        self.requests.append(req)
        j = "NG" if "高い" in req.text else "OK"
        return ModelResponse(raw_text=json.dumps({"judgement": j, "confidence": 0.8}), input_tokens=200,
                             output_tokens=20, latency_s=0.5)


@pytest.fixture
def textlab(tmp_path):
    rows = [_row(f"{c}/{lab}/{i}", lab, "test") for c in ("pcb1", "capsules") for lab in ("normal", "anomaly")
            for i in range(2)]
    (tmp_path / "data").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "split.csv", index=False)
    feats = pd.DataFrame([{"image_id": r["image_id"], "category": r["category"], "label": r["label"],
                           "text_coarse": "異常度: 高い" if r["label"] == "anomaly" else "異常度: 同じくらい"}
                          for r in rows])
    feats.to_csv(tmp_path / "data" / "features.csv", index=False)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "e.txt").write_text("品目: {{item}}\n{{features}}\nJSON で", encoding="utf-8")
    cfg = {"name": "e", "split": "data/split.csv", "model_id": "m", "region": "ap-northeast-1",
           "prompt": "prompts/e.txt", "prompt_vars": {"pcb1": {"item": "基板"}, "capsules": {"item": "カプセル"}},
           "text_only": True, "features": "data/features.csv", "features_column": "text_coarse",
           "budget_usd": 1.0}
    (tmp_path / "c.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path, PricingTable([Price("m", "standard", 0.042, 0.168)])


def test_text_only_sends_no_images_and_records_text(textlab):
    base, pricing = textlab
    plan = plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    model = TextModel()
    s = execute(plan, model, pricing, base / "no_visa", base, log=lambda _: None)  # 画像の置き場は無くてよい
    assert s["called"] == 8
    assert all(r.images == [] for r in model.requests)
    recs = [json.loads(x) for x in (base / "runs" / "e" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["n_images_sent"] == 0 and r["text_only"] for r in recs)
    a = next(r for r in recs if r["image_id"] == "pcb1/anomaly/0")
    assert a["sent_text"] == "品目: 基板\n異常度: 高い\nJSON で" and a["parsed"]["judgement"] == "NG"
    assert a["cost_usd"] == pytest.approx((200 * 0.042 + 20 * 0.168) / 1e6)


def test_text_only_resume_stops_when_features_change(textlab):
    base, pricing = textlab
    cfg = RunConfig.load(base / "c.yaml")
    execute(plan_run(cfg, base, base / "runs", limit=2), TextModel(), pricing, base, base, log=lambda _: None)
    f = pd.read_csv(base / "data" / "features.csv")
    f.loc[0, "text_coarse"] = "異常度: 変えた"
    f.to_csv(base / "data" / "features.csv", index=False)
    with pytest.raises(RuntimeError, match="config differs"):
        execute(plan_run(cfg, base, base / "runs", limit=2), TextModel(), pricing, base, base, log=lambda _: None)


def test_text_only_config_and_features_checks(textlab):
    base, _ = textlab
    d = json.loads((base / "c.yaml").read_text(encoding="utf-8"))
    for bad in ({"n_refs": 1, "refs": "data/split.csv"}, {"features_column": None}):
        (base / "bad.yaml").write_text(json.dumps({**d, **bad}), encoding="utf-8")
        with pytest.raises(ValueError, match="text_only requires"):
            RunConfig.load(base / "bad.yaml")
    (base / "bad.yaml").write_text(json.dumps({**d, "text_only": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="only used with text_only"):
        RunConfig.load(base / "bad.yaml")
    f = pd.read_csv(base / "data" / "features.csv")
    f.iloc[1:].to_csv(base / "data" / "features.csv", index=False)
    with pytest.raises(ValueError, match="features missing"):
        plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")
    f.loc[0, "text_coarse"] = " "
    f.to_csv(base / "data" / "features.csv", index=False)
    with pytest.raises(ValueError, match="empty features text"):
        plan_run(RunConfig.load(base / "c.yaml"), base, base / "runs")


# ---------------------------------------------------------------- CI-E7、E8

def _features():
    rows = []
    for cat in ("pcb1", "capsules"):
        for i, s in enumerate((0.10, 0.12, 0.14)):
            rows.append({"image_id": f"{cat}/normal/{i}", "category": cat, "label": "normal", "score": s})
        for i, s in enumerate((0.13, 0.30, 0.40)):
            rows.append({"image_id": f"{cat}/anomaly/{i}", "category": cat, "label": "anomaly", "score": s})
    return pd.DataFrame(rows)


def test_model_row_uses_repeat1_and_checks_images():
    f = _features()
    calls = f[["image_id", "category", "label"]].assign(
        repeat=1, pred=np.where(f["score"] > 0.2, "NG", "OK"), confidence=0.9, cost_usd=0.00002,
        input_tokens=150, output_tokens=20, latency_s=0.4)
    extra = calls.iloc[:1].assign(repeat=2, pred="NG")
    r = model_row("x", pd.concat([calls, extra]), f, "文")
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (4, 2, 0, 6)
    assert r["cost_per_1000_usd"] == pytest.approx(0.02)
    assert r["egress"] == "文"
    with pytest.raises(ValueError, match="differ"):
        model_row("x", calls.iloc[1:], f, "文")


def test_anomalydino_rows_use_calibration_cuts():
    f = _features()
    cals = {c: Calibration(c, 3, 0.999, 0.5, (0.09, 0.11, 0.135)) for c in ("pcb1", "capsules")}
    rows = anomalydino_rows(f, cals)
    by_max = rows[0]
    # 良品の最大 0.135 を超える：不良品 0.30・0.40 と良品 0.14 が NG
    assert (by_max["tp"], by_max["fp"], by_max["fn"]) == (4, 2, 2)
    assert by_max["cost_per_1000_usd"] == 0.0 and by_max["egress"].startswith("なし")
    # 見逃し2%：しきい値は各品目 0.13（不良品の最小）→ 良品 0.10・0.12 が通過
    assert by_max["pass_rate_at_miss2"] == pytest.approx(4 / 6)
