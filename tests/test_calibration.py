"""確信度の較正（docs/contracts.md の CI-CB1〜CB8）。"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from vlmlab import metrics as M
from vlmlab.calibration import build_calibration


def calls(rows):
    """rows: (image_id, label, repeat, pred, confidence)"""
    return pd.DataFrame([
        {"image_id": i, "label": l, "repeat": r, "pred": p, "confidence": c, "category": "pcb1"}
        for i, l, r, p, c in rows
    ])


# ---- CI-CB1・CB2：帯の決め方 ----

def test_bin_edges_left_closed_and_one_in_last_bin():
    """CI-CB2：k/10 ≤ c < (k+1)/10、c = 1 は最後の帯。境界の直前・値・直後"""
    idx = M.conf_bin_index(np.array([0.0, 0.0999999, 0.1, 0.1000001, 0.6999999, 0.7, 0.8999999, 0.9, 0.95, 1.0]))
    assert idx.tolist() == [0, 0, 1, 1, 6, 7, 8, 9, 9, 9]


def test_bin_edges_survive_float_error():
    """CI-CB2：0.7 や 0.3 のように c × 10 が誤差で 7.000…1 / 2.999… になる値も、その帯に入る"""
    vals = np.array([0.1 * k for k in range(11)])  # 0.30000000000000004 などを含む
    assert M.conf_bin_index(vals).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    assert M.conf_bin_index(np.array([1 - 0.7])).tolist() == [3]


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
def test_confidence_outside_unit_interval_raises(bad):
    """CI-CB1"""
    with pytest.raises(ValueError):
        M.conf_bin_index(np.array([0.5, bad]))


# ---- CI-CB3・CB4：信頼度図の表と ECE ----

def test_reliability_bins_counts_accuracy_and_few_flag():
    """CI-CB3：帯ごとの件数・平均確信度・正解率・差。0 < n < 10 の帯に few、空の帯は欠損（CSV では空欄）で few=False"""
    conf = np.array([0.95] * 10 + [0.55] * 2)
    correct = np.array([True] * 6 + [False] * 4 + [True, False])
    b = M.reliability_bins(conf, correct)
    assert len(b) == 10 and b["n"].sum() == 12
    top, mid, empty = b.iloc[9], b.iloc[5], b.iloc[0]
    assert (top.n, top.accuracy, top.few) == (10, 0.6, False)
    assert top.mean_conf == pytest.approx(0.95) and top.gap == pytest.approx(0.6 - 0.95)
    assert (mid.n, mid.accuracy, bool(mid.few)) == (2, 0.5, True)
    assert empty.n == 0 and pd.isna(empty.accuracy) and pd.isna(empty.mean_conf) and not empty.few


def test_few_threshold_boundary():
    """CI-CB3：件数 9 は few、10 は few でない"""
    assert bool(M.reliability_bins(np.full(9, 0.95), np.ones(9, bool)).iloc[9].few) is True
    assert bool(M.reliability_bins(np.full(10, 0.95), np.ones(10, bool)).iloc[9].few) is False


def test_ece_hand_computed():
    """CI-CB4：ECE = Σ n_b/N |acc_b − conf_b|。手計算 (10/12)|0.6−0.95| + (2/12)|0.5−0.55|"""
    conf = np.array([0.95] * 10 + [0.55] * 2)
    correct = np.array([True] * 6 + [False] * 4 + [True, False])
    e = M.ece(conf, correct)
    assert e["ece"] == pytest.approx(10 / 12 * 0.35 + 2 / 12 * 0.05)
    assert e["n"] == 12 and e["n_bins_used"] == 2 and e["n_bins_few"] == 1
    assert e["accuracy"] == pytest.approx(7 / 12)
    assert e["overconfidence"] == pytest.approx(conf.mean() - 7 / 12)


def test_ece_perfect_and_worst():
    """CI-CB4：確信度が帯の正解率と同じなら 0、確信度 1 で全部外れなら 1"""
    assert M.ece(np.array([0.5, 0.5, 1.0]), np.array([True, False, True]))["ece"] == pytest.approx(0.0)
    assert M.ece(np.array([1.0, 1.0]), np.array([False, False]))["ece"] == pytest.approx(1.0)


def test_ece_is_not_pooled_mean_gap():
    """CI-CB4：帯ごとの差の絶対値を取る（全体の平均の差ではない）。過信と過小が打ち消し合わない"""
    conf = np.array([0.9] * 10 + [0.1] * 10)
    correct = np.array([False] * 10 + [True] * 10)  # 0.9 の帯は全部外れ、0.1 の帯は全部当たり
    e = M.ece(conf, correct)
    assert e["overconfidence"] == pytest.approx(0.0)
    assert e["ece"] == pytest.approx(0.9)


def test_ece_empty_is_none():
    """CI-CB4：N = 0 なら None（0 にしない）"""
    e = M.ece(np.array([]), np.array([]))
    assert e["ece"] is None and e["n"] == 0


# ---- CI-CB5：自己申告 ----

def test_self_report_points_views_and_parse_fail_excluded():
    """CI-CB5：judgement は申告値と正解か、anomaly は D1 のスコアと不良品か。解析失敗は除く"""
    df = calls([
        ("a", "anomaly", 1, "NG", 0.9),   # 正解。不良らしさ 0.9
        ("b", "anomaly", 1, "OK", 0.7),   # 不正解。不良らしさ 0.3
        ("c", "normal", 1, "OK", 1.0),    # 正解。不良らしさ 0
        ("d", "normal", 1, None, None),   # 解析失敗
    ])
    j, r = M.self_report_points(df, "judgement")
    assert r is None and j["conf"].tolist() == [0.9, 0.7, 1.0] and j["correct"].tolist() == [True, False, True]
    a, _ = M.self_report_points(df, "anomaly")
    assert a["conf"].tolist() == [0.9, 0.3, 0.0] and a["correct"].tolist() == [True, True, False]


def test_self_report_missing_confidence_gives_reason():
    """CI-CB5：有効な行に確信度が無いものがあれば計算しない（CI-M7 と同じ扱い）"""
    pts, reason = M.self_report_points(calls([("a", "anomaly", 1, "NG", None)]), "judgement")
    assert pts is None and "confidence missing" in reason


# ---- CI-CB6：一致率による確信度 ----

def _five(image_id, label, preds):
    return [(image_id, label, r + 1, p, 0.9 if p else None) for r, p in enumerate(preds)]


def test_consistency_p_ng_vote_and_conf():
    """CI-CB6：p_ng = NG の回 / 有効な回。多数決と max(p, 1−p)"""
    df = calls(_five("a", "anomaly", ["NG", "NG", "NG", "OK", "OK"])
               + _five("b", "normal", ["OK"] * 5)
               + _five("c", "normal", ["NG", "OK", "OK", "OK", "OK"]))
    img = M.consistency_by_image(df).set_index("image_id")
    assert img.loc["a", "p_ng"] == 0.6 and img.loc["a", "vote_pred"] == "NG" and img.loc["a", "vote_conf"] == 0.6
    assert img.loc["b", "p_ng"] == 0.0 and img.loc["b", "vote_pred"] == "OK" and img.loc["b", "vote_conf"] == 1.0
    assert img.loc["c", "p_ng"] == 0.2 and img.loc["c", "vote_pred"] == "OK" and img.loc["c", "vote_conf"] == 0.8


def test_consistency_parse_fail_reduces_denominator_and_tie_is_none():
    """CI-CB6：解析失敗の回は分母から外す。NG と OK が同数なら vote は None、有効な回が無ければ p_ng も None"""
    df = calls(_five("a", "anomaly", ["NG", "NG", "OK", "OK", None])
               + _five("b", "anomaly", ["NG", "NG", "NG", None, None])
               + _five("c", "normal", [None] * 5))
    img = M.consistency_by_image(df).set_index("image_id")
    assert img.loc["a", "n_valid"] == 4 and img.loc["a", "p_ng"] == 0.5
    assert pd.isna(img.loc["a", "vote_pred"]) and pd.isna(img.loc["a", "vote_conf"])
    assert img.loc["b", "p_ng"] == 1.0 and img.loc["b", "vote_pred"] == "NG"
    assert pd.isna(img.loc["c", "p_ng"]) and pd.isna(img.loc["c", "vote_pred"])

    j = M.consistency_points(M.consistency_by_image(df), "judgement")
    assert j["image_id"].tolist() == ["b"]                       # 同数・有効なしは除く
    a = M.consistency_points(M.consistency_by_image(df), "anomaly")
    assert a["image_id"].tolist() == ["a", "b"] and a["conf"].tolist() == [0.5, 1.0]


def test_consistency_rounding_of_thirds():
    """CI-CB6：3回のとき 1/3・2/3 は小数9桁に丸め、vote_conf は 2/3 と同じ値になる"""
    df = calls(_five("a", "anomaly", ["NG", "OK", "OK"]) + _five("b", "anomaly", ["NG", "NG", "OK"]))
    img = M.consistency_by_image(df).set_index("image_id")
    assert img.loc["a", "vote_conf"] == img.loc["b", "vote_conf"] == round(2 / 3, 9)


def test_consistency_label_conflict_raises():
    """CI-CB6（pre）：同じ画像で正解が回によって違えば止める"""
    with pytest.raises(ValueError):
        M.consistency_by_image(calls([("a", "anomaly", 1, "NG", 0.9), ("a", "normal", 2, "NG", 0.9)]))


# ---- CI-CB7：グレーゾーン ----

def test_gray_zone_boundaries():
    """CI-CB7：< lo は自動 OK、≥ hi は自動 NG、[lo, hi) と NaN は人。lo・hi ちょうどの値"""
    scores = np.array([0.0, 0.4, 0.6, 0.8, 1.0, np.nan, 0.2])
    is_anom = np.array([True, True, False, False, True, True, False])
    g = M.gray_zone(scores, is_anom, 0.4, 0.8)
    # 自動 OK: 0.0(不良=見逃し), 0.2(良品)。人: 0.4, 0.6, NaN。自動 NG: 0.8(良品=誤検知), 1.0
    assert (g["n_auto_ok"], g["n_human"], g["n_auto_ng"]) == (2, 3, 2)
    assert g["n_human_no_score"] == 1
    assert g["n_miss"] == 1 and g["miss_rate"] == pytest.approx(1 / 4)
    assert g["n_false_ng"] == 1 and g["false_positive_rate"] == pytest.approx(1 / 3)


def test_gray_zone_float_error_at_boundary():
    """CI-CB7：1 − 0.6 = 0.4000000000000001 も 0.4 と同じに扱う（人へ）。0.39999… は自動 OK"""
    g = M.gray_zone(np.array([1 - 0.6, 0.3999]), np.array([True, True]), 0.4, 0.8)
    assert g["n_human"] == 1 and g["n_miss"] == 1


def test_gray_zone_no_gray_and_invalid():
    """CI-CB7：lo = hi なら人に回さない。lo > hi や範囲外は ValueError"""
    g = M.gray_zone(np.array([0.2, 0.5, 0.9]), np.array([False, True, True]), 0.5, 0.5)
    assert g["n_human"] == 0 and g["n_auto_ng"] == 2
    for lo, hi in [(0.8, 0.4), (-0.1, 0.5), (0.5, 1.1)]:
        with pytest.raises(ValueError):
            M.gray_zone(np.array([0.5]), np.array([True]), lo, hi)


# ---- CI-CB8：業務指標を一致率のスコアで ----

def test_pass_rate_scores_matches_dataframe_version():
    """CI-CB8：pass_rate_at_miss_scores は、D1 のスコアを渡すと pass_rate_at_miss（CI-M6）と同じ結果"""
    rows = [("a1", "anomaly", 1, "NG", 0.9), ("a2", "anomaly", 1, "NG", 0.6), ("a3", "anomaly", 1, "OK", 0.7),
            ("n1", "normal", 1, "OK", 0.9), ("n2", "normal", 1, "NG", 0.6), ("n3", "normal", 1, "OK", 0.7)]
    df = calls(rows)
    scores = np.array([M.anomaly_score(p, c) for _, _, _, p, c in rows])
    is_anom = np.array([l == "anomaly" for _, l, _, _, _ in rows])
    for t in (0.0, 0.34, 0.5):
        assert M.pass_rate_at_miss_scores(scores, is_anom, t) == M.pass_rate_at_miss(df, t)


def test_pass_rate_scores_ties_do_not_pass():
    """CI-CB8（CI-M6）：一致率のスコアは値が少なく同点が多い。不良品と同じスコアの良品は通さない"""
    r = M.pass_rate_at_miss_scores(np.array([0.6, 1.0, 0.6, 0.0]), np.array([True, True, False, False]), 0.0)
    assert r["threshold"] == 0.6 and r["pass_rate"] == 0.5 and r["achieved_miss_rate"] == 0.0


# ---- 集計（build_calibration）----

def _write_run(d, name, rows, repeats):
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"name": name, "repeats": repeats, "parse": {}}), encoding="utf-8")
    with (d / "calls.jsonl").open("w", encoding="utf-8") as f:
        for i, l, r, p, c in rows:
            body = None if p is None else json.dumps({"judgement": p, "defect": "", "confidence": c})
            f.write(json.dumps({"run_name": name, "repeat": r, "image_id": i, "category": "pcb1", "label": l,
                                "path": f"{i}.JPG", "raw_text": body or "broken", "api_error": None,
                                "latency_s": 1.0, "cost_usd": 0.001}) + "\n")


def test_build_calibration_end_to_end(tmp_path):
    """CI-CB4〜CB8 をまとめて：自己申告（回ごと・全回）と一致率の行、信頼度図の表、グレーゾーンの平均の行"""
    rows = (_five("a", "anomaly", ["NG", "NG", "NG", "OK", "NG"]) + _five("b", "anomaly", ["OK"] * 5)
            + _five("c", "normal", ["OK", "OK", "NG", "OK", "OK"]) + _five("d", "normal", ["OK", None, "OK", "OK", "OK"]))
    _write_run(tmp_path / "runs" / "t07", "t07", rows, 5)
    p = build_calibration([tmp_path / "runs" / "t07"], [tmp_path / "runs" / "t07"], tmp_path / "out", "cal")
    s = pd.read_csv(p["summary"])
    assert set(zip(s.method, s.view)) == {("self", "judgement"), ("self", "anomaly"),
                                         ("consistency", "judgement"), ("consistency", "anomaly")}
    self_j = s[(s.method == "self") & (s.view == "judgement")].iloc[0]
    assert self_j.n == 19  # 20 回 − 解析失敗 1 回
    # 自己申告はすべて 0.9。正解 = a:4, b:0, c:4, d:4 → 12/19
    assert self_j.ece == pytest.approx(abs(12 / 19 - 0.9))
    cons_j = s[(s.method == "consistency") & (s.view == "judgement")].iloc[0]
    # 多数決：a NG(0.8) 正、b OK(1.0) 誤、c OK(0.8) 正、d OK(1.0) 正 → 帯 8: n2 acc1 conf0.8, 帯 9: n2 acc0.5 conf1
    assert cons_j.ece == pytest.approx(2 / 4 * 0.2 + 2 / 4 * 0.5)
    g = pd.read_csv(p["gray_zone"])
    assert set(g["repeat"].astype(str)) >= {"1", "5", "mean", "vote"}
    b = pd.read_csv(p["bins"])
    assert b.groupby(["method", "view"])["n"].sum().to_dict()[("consistency", "anomaly")] == 4
    imgs = pd.read_csv(p["images"])
    assert imgs.set_index("image_id").loc["d", "n_valid"] == 4
