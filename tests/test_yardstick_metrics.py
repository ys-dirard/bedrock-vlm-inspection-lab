"""12/1 の記事（評価の物差し）で足した指標の契約テスト（docs/contracts.md の CI-Y1〜Y5）。"""

import math

import pandas as pd
import pytest

from vlmlab import metrics as M


def calls(rows):
    """rows: (image_id, label, repeat, pred)"""
    return pd.DataFrame([
        {"image_id": i, "label": l, "repeat": r, "pred": p, "confidence": None if p is None else 1.0,
         "latency_s": 1.0, "cost_usd": 0.001}
        for i, l, r, p in rows
    ])


# ---------------------------------------------------------------- CI-Y1

@pytest.mark.parametrize("scores,labels", [
    ([0.1, 0.2], ["anomaly"]),                 # 長さが違う
    ([0.1, 0.2], ["anomaly", "good"]),         # 知らない label
    ([0.1, float("nan")], ["anomaly", "normal"]),  # NaN
    ([0.1, math.inf], ["anomaly", "normal"]),  # 無限大
])
def test_invalid_scores_raise(scores, labels):
    """CI-Y1"""
    with pytest.raises(ValueError):
        M.pass_rate_at_miss_scores(scores, labels, 0.05)
    with pytest.raises(ValueError):
        M.operating_points(scores, labels)


# ---------------------------------------------------------------- CI-Y2

def test_scores_threshold_is_largest_meeting_target_and_counts_passed():
    """CI-Y2：不良品 [0,1,2,3,3]、良品 [0,0,1,3]。見逃し 20% まで → t=1（スコア 0 の不良品1枚を見逃す）"""
    scores = [0, 1, 2, 3, 3, 0, 0, 1, 3]
    labels = ["anomaly"] * 5 + ["normal"] * 4
    r = M.pass_rate_at_miss_scores(scores, labels, 0.2)
    assert r["threshold"] == 1
    assert r["achieved_miss_rate"] == pytest.approx(0.2)
    assert r["n_normal_passed"] == 2  # スコア 0 の2枚。スコア 1 の良品は不良品と同点なので通さない
    assert r["pass_rate"] == pytest.approx(2 / 4)


def test_scores_boundary_just_below_target():
    """CI-Y2：見逃し率 0.2 がちょうど目標なら採り、目標 0.19 では採らない"""
    scores = [0, 1, 2, 3, 3, 0, 0, 1, 3]
    labels = ["anomaly"] * 5 + ["normal"] * 4
    below = M.pass_rate_at_miss_scores(scores, labels, 0.19)
    assert below["threshold"] == 0 and below["achieved_miss_rate"] == 0.0
    assert below["pass_rate"] == 0.0 and below["n_normal_passed"] == 0


def test_scores_ties_go_to_ng_side():
    """CI-Y2：全画像が同じスコアなら、見逃し 0 のしきい値で良品は1枚も通らない（同点は NG 側）"""
    r = M.pass_rate_at_miss_scores([0, 0, 0, 0], ["anomaly", "anomaly", "normal", "normal"], 0.05)
    assert r["threshold"] == 0 and r["pass_rate"] == 0.0


def test_scores_none_without_both_labels():
    """CI-Y2"""
    r = M.pass_rate_at_miss_scores([1, 2], ["normal", "normal"], 0.05)
    assert r["pass_rate"] is None and r["reason"]
    assert r["n_normal"] == 2 and r["n_anomaly"] == 0


def test_pass_rate_at_miss_equals_scores_version_on_d1_scores():
    """CI-Y2：pass_rate_at_miss は D1 のスコアで pass_rate_at_miss_scores と同じ結果を返す（CI-M6 を変えない）"""
    df = pd.DataFrame({
        "label": ["anomaly", "anomaly", "anomaly", "normal", "normal", "normal"],
        "pred": ["NG", "OK", "NG", "OK", "OK", "NG"],
        "confidence": [0.9, 0.7, 0.6, 0.95, 0.7, 0.55],
    })
    scores = [M.anomaly_score(p, c) for p, c in zip(df["pred"], df["confidence"])]
    for target in (0.0, 0.34, 0.67):
        assert M.pass_rate_at_miss(df, target) == M.pass_rate_at_miss_scores(scores, df["label"], target)


# ---------------------------------------------------------------- CI-Y3

def test_accuracy_excludes_parse_failures():
    """CI-Y3：TP1・FN1・TN2・FP0、解析失敗1 → 正答率 3/4"""
    df = pd.DataFrame({
        "label": ["anomaly", "anomaly", "normal", "normal", "normal"],
        "pred": ["NG", "OK", "OK", "OK", None],
    })
    m = M.binary_metrics(df)
    assert m["accuracy"] == pytest.approx(3 / 4)
    assert m["n_valid"] == 4 and m["n_parse_fail"] == 1


def test_accuracy_all_ok_model_on_rare_defects():
    """CI-Y3：不良1%で全部 OK と言うと、正答率 0.99・再現率 0"""
    df = pd.DataFrame({"label": ["anomaly"] + ["normal"] * 99, "pred": ["OK"] * 100})
    m = M.binary_metrics(df)
    assert m["accuracy"] == pytest.approx(0.99)
    assert m["recall"] == 0.0 and m["miss_rate"] == 1.0


def test_accuracy_none_when_no_valid_rows():
    """CI-Y3：有効な行が 0 件なら None（0/0 を 0 や 1 にしない）"""
    m = M.binary_metrics(pd.DataFrame({"label": ["normal"], "pred": [None]}))
    assert m["accuracy"] is None


# ---------------------------------------------------------------- CI-Y4

def test_ng_counts_counts_ng_and_excludes_images_with_parse_failure():
    """CI-Y4"""
    df = calls([
        ("a", "anomaly", 1, "NG"), ("a", "anomaly", 2, "OK"), ("a", "anomaly", 3, "NG"),
        ("b", "normal", 1, "OK"), ("b", "normal", 2, "OK"), ("b", "normal", 3, "OK"),
        ("c", "normal", 1, "OK"), ("c", "normal", 2, None), ("c", "normal", 3, "NG"),
    ])
    t, excluded = M.ng_counts(df)
    assert t.loc["a", "ng_count"] == 2 and t.loc["b", "ng_count"] == 0
    assert "c" not in t.index and excluded == ["c"]
    assert t.loc["a", "label"] == "anomaly"


@pytest.mark.parametrize("reps", [[1, 2], [1, 2, 2], [1, 2, 3, 4], [0, 1, 2]])
def test_ng_counts_requires_each_repeat_exactly_once(reps):
    """CI-Y4"""
    df = calls([("a", "normal", r, "OK") for r in reps])
    with pytest.raises(ValueError):
        M.ng_counts(df)


def test_ng_counts_rejects_label_mismatch():
    """CI-Y4"""
    df = calls([("a", "normal", 1, "OK"), ("a", "anomaly", 2, "OK"), ("a", "normal", 3, "OK")])
    with pytest.raises(ValueError):
        M.ng_counts(df)


# ---------------------------------------------------------------- CI-Y5

def test_operating_points_lists_each_threshold_in_order():
    """CI-Y5：不良品 [1,3]、良品 [0,1]"""
    op = M.operating_points([1, 3, 0, 1], ["anomaly", "anomaly", "normal", "normal"])
    assert op["threshold"].tolist() == [0, 1, 3, math.inf]
    assert op["miss_rate"].tolist() == [0.0, 0.0, 0.5, 1.0]
    assert op["pass_rate"].tolist() == [0.0, 0.5, 1.0, 1.0]


# ---------------------------------------------------------------- CI-Y6（表の行）

def test_repeated_condition_row():
    """CI-Y6：点の指標は回ごとの平均、並べ方2は画像ごとの NG 回数、画像の平均スコアは丸めて同じ値にそろえる"""
    from vlmlab.yardstick import repeated_condition

    rows = []
    # a: 不良品。3回とも NG（確信度 0.9）
    rows += [("a", "anomaly", r, "NG", 0.9) for r in (1, 2, 3)]
    # b: 不良品。1回目だけ NG（0.15）、残りは OK（1.0）→ スコア平均 0.05
    rows += [("b", "anomaly", 1, "NG", 0.15), ("b", "anomaly", 2, "OK", 1.0), ("b", "anomaly", 3, "OK", 1.0)]
    # c: 良品。3回とも OK（0.95）→ スコア 0.05 が3つ。b と同じ平均にならなければならない
    rows += [("c", "normal", r, "OK", 0.95) for r in (1, 2, 3)]
    # d: 良品。3回とも OK（1.0）
    rows += [("d", "normal", r, "OK", 1.0) for r in (1, 2, 3)]
    df = pd.DataFrame([{"image_id": i, "label": l, "repeat": r, "pred": p, "confidence": c,
                        "latency_s": 1.0, "cost_usd": 0.001} for i, l, r, p, c in rows])
    row, detail, ops = repeated_condition("x", df)
    # 回ごとの正答率：1回目 4/4、2回目 3/4、3回目 3/4 → 平均 10/12
    assert row["accuracy"] == pytest.approx(10 / 12)
    assert row["accuracy_min"] == pytest.approx(0.75) and row["accuracy_max"] == 1.0
    # NG 回数：a=3、b=1、c=0、d=0。見逃し 0 → t=1（b の値）。良品は2枚ともスコア 0 < 1 で通過
    assert row["pass_5pct_ngcount"] == 1.0
    # 画像の平均スコア：a=0.9、b=(0.15+0+0)/3=0.05、c=0.05、d=0。b と c は丸めて同じ値になる
    imgmean = ops[0]
    assert (imgmean["threshold"] == 0.05).sum() == 1
    # 見逃し 0 → t=0.05。同点の c は通さず（NG 側）、d だけ通る
    assert row["pass_5pct_conf_imgmean"] == 0.5
