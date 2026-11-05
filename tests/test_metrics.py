import pandas as pd
import pytest

from vlmlab import metrics as M


def frame(rows, repeat=1):
    """rows: (label, pred, confidence)"""
    return pd.DataFrame([
        {"image_id": f"i{k}", "label": l, "pred": p, "confidence": c, "repeat": repeat,
         "latency_s": 1.0, "cost_usd": 0.001}
        for k, (l, p, c) in enumerate(rows)
    ])


def test_counts_and_rates():
    """CI-M2, CI-M3, CI-M4"""
    df = frame([
        ("anomaly", "NG", 0.9), ("anomaly", "NG", 0.8), ("anomaly", "OK", 0.6),   # TP2 FN1
        ("normal", "NG", 0.7), ("normal", "OK", 0.9), ("normal", "OK", 0.9), ("normal", "OK", 0.9),  # FP1 TN3
    ])
    m = M.binary_metrics(df)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 4)
    assert m["miss_rate"] == pytest.approx(1 / 3)


def test_parse_failures_are_excluded_and_counted():
    """CI-M2（D3）：解析失敗は指標から除き、件数と率を出す"""
    df = frame([("anomaly", "NG", 0.9), ("anomaly", None, None), ("normal", "OK", 0.9), ("normal", None, None)])
    m = M.binary_metrics(df)
    assert m["n_total"] == 4 and m["n_valid"] == 2 and m["n_parse_fail"] == 2
    assert m["parse_fail_rate"] == 0.5
    assert m["recall"] == 1.0 and m["false_positive_rate"] == 0.0


def test_zero_denominators_give_none_not_zero_or_one():
    """CI-M3, CI-M4：0/0 は None"""
    m = M.binary_metrics(frame([("normal", "OK", 0.9), ("normal", "OK", 0.9)]))
    assert m["precision"] is None and m["recall"] is None and m["miss_rate"] is None and m["f1"] is None
    assert m["false_positive_rate"] == 0.0


def test_f1_zero_when_precision_and_recall_zero():
    """CI-M4"""
    m = M.binary_metrics(frame([("anomaly", "OK", 0.9), ("normal", "NG", 0.9)]))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


@pytest.mark.parametrize("bad", [{"label": "good"}, {"pred": "ng"}])
def test_invalid_values_raise(bad):
    """CI-M1"""
    df = frame([("anomaly", "NG", 0.9)])
    for k, v in bad.items():
        df[k] = v
    with pytest.raises(ValueError):
        M.binary_metrics(df)


def test_repeats_are_computed_separately_then_mean_min_max():
    """CI-M5（D2）"""
    r1 = frame([("anomaly", "NG", 0.9), ("normal", "OK", 0.9)], repeat=1)  # recall 1
    r2 = frame([("anomaly", "OK", 0.9), ("normal", "OK", 0.9)], repeat=2)  # recall 0
    r3 = frame([("anomaly", None, None), ("normal", "OK", 0.9)], repeat=3)  # recall None
    agg = M.aggregate_repeats(M.per_repeat(pd.concat([r1, r2, r3])))
    assert agg["n_repeats"] == 3
    assert agg["recall_mean"] == 0.5 and agg["recall_min"] == 0.0 and agg["recall_max"] == 1.0
    assert agg["recall_n"] == 2  # None の回は平均に入れない
    assert agg["n_parse_fail_sum"] == 1


def test_anomaly_score():
    """CI-M6（D1）"""
    assert M.anomaly_score("NG", 0.8) == 0.8
    assert M.anomaly_score("OK", 0.8) == pytest.approx(0.2)


def test_pass_rate_at_miss_picks_largest_threshold_meeting_target():
    """CI-M6：不良品10枚、見逃し 10% 以下 → 1枚まで見逃してよい"""
    anomalies = [("anomaly", "NG", s) for s in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]] + [("anomaly", "OK", 0.9)]  # 最後はスコア 0.1
    normals = [("normal", "OK", c) for c in [0.95, 0.9, 0.6]] + [("normal", "NG", 0.6)]  # スコア 0.05, 0.1, 0.4, 0.6
    r = M.pass_rate_at_miss(frame(anomalies + normals), 0.10)
    # しきい値 0.55：見逃しはスコア 0.1 の1枚（10%）。良品はスコア 0.05・0.1・0.4 の3枚が通過
    assert r["threshold"] == pytest.approx(0.55)
    assert r["achieved_miss_rate"] == pytest.approx(0.1)
    assert r["pass_rate"] == pytest.approx(3 / 4)


def test_pass_rate_boundary_just_below_and_above_target():
    """CI-M6：見逃し率がちょうど目標のときは採り、少しでも超えれば採らない"""
    anomalies = [("anomaly", "NG", s) for s in [0.9, 0.8, 0.7, 0.6]] + [("anomaly", "NG", 0.5)]
    normals = [("normal", "OK", 0.5)]  # スコア 0.5
    df = frame(anomalies + normals)
    at = M.pass_rate_at_miss(df, 0.2)   # 1/5 = 0.2 の見逃しを許す → t=0.6
    assert at["threshold"] == pytest.approx(0.6) and at["pass_rate"] == 1.0
    below = M.pass_rate_at_miss(df, 0.19)  # 見逃し 0 → t=0.5。スコアが同じ良品は通さない
    assert below["threshold"] == pytest.approx(0.5) and below["pass_rate"] == 0.0


def test_equal_scores_from_float_error_are_treated_as_equal():
    """CI-M6：OK・0.7 の不良品（1−0.7）と NG・0.3 の良品は同じスコア。良品は通さない"""
    df = frame([("anomaly", "OK", 0.7), ("normal", "NG", 0.3)])
    r = M.pass_rate_at_miss(df, 0.0)
    assert r["threshold"] == 0.3
    assert r["pass_rate"] == 0.0


def test_pass_rate_none_when_confidence_missing():
    """CI-M7"""
    df = frame([("anomaly", "NG", 0.9), ("normal", "OK", None)])
    r = M.pass_rate_at_miss(df, 0.05)
    assert r["pass_rate"] is None and "confidence" in r["reason"]


def test_pass_rate_none_without_both_labels():
    """CI-M7"""
    r = M.pass_rate_at_miss(frame([("normal", "OK", 0.9)]), 0.05)
    assert r["pass_rate"] is None and r["reason"]


def test_agreement_rate_counts_parse_failure_as_disagreement():
    """CI-M8"""
    df = pd.DataFrame([
        {"image_id": "a", "label": "normal", "pred": "OK", "repeat": 1},
        {"image_id": "a", "label": "normal", "pred": "OK", "repeat": 2},
        {"image_id": "b", "label": "normal", "pred": "OK", "repeat": 1},
        {"image_id": "b", "label": "normal", "pred": "NG", "repeat": 2},
        {"image_id": "c", "label": "anomaly", "pred": None, "repeat": 1},
        {"image_id": "c", "label": "anomaly", "pred": None, "repeat": 2},
        {"image_id": "d", "label": "anomaly", "pred": "NG", "repeat": 1},  # 1回だけ → 分母に入れない
    ])
    r = M.agreement_rate(df)
    assert r == {"agreement_rate": 1 / 3, "n_images_repeated": 3, "n_agree": 1}


def test_latency_percentiles_include_parse_failures():
    """CI-M9"""
    df = pd.DataFrame({"latency_s": [1.0, 2.0, 3.0, 4.0, 5.0], "pred": ["OK", None, "NG", "OK", None]})
    r = M.latency_stats(df)
    assert r["latency_p50_s"] == 3.0
    assert r["latency_p95_s"] == pytest.approx(4.8)
    assert r["n_latency"] == 5


def test_cost_stats_and_unknown_cost():
    """CI-M10"""
    df = pd.DataFrame({"cost_usd": [0.001, 0.003]})
    r = M.cost_stats(df)
    assert r["cost_per_call_usd"] == pytest.approx(0.002)
    assert r["cost_per_1000_usd"] == pytest.approx(2.0)
    assert M.cost_stats(pd.DataFrame({"cost_usd": [0.001, None]}))["cost_per_1000_usd"] is None
