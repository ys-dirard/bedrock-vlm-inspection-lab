"""prefilter.py（2段構成）の契約 CI-F1〜F7。"""

import numpy as np
import pandas as pd
import pytest

from vlmlab.prefilter import (conditions, evaluate, join, score_summary, threshold_at_miss, thresholds,
                              top_percent)


def _ad(rows):
    """rows: (image_id, category, label, score)"""
    return pd.DataFrame(rows, columns=["image_id", "category", "label", "score"])


def _nova(spec):
    """spec: {image_id: (category, label, [pred1, pred2, pred3], cost_per_call)}"""
    out = []
    for iid, (cat, lab, preds, cost) in spec.items():
        for r, p in enumerate(preds, start=1):
            out.append({"image_id": iid, "category": cat, "label": lab, "repeat": r, "pred": p,
                        "confidence": None if p is None else 0.95, "cost_usd": cost, "latency_s": 1.0})
    return pd.DataFrame(out)


def _base():
    ad = _ad([
        ("a/n1", "a", "normal", 0.1), ("a/n2", "a", "normal", 0.3),
        ("a/x1", "a", "anomaly", 0.5), ("a/x2", "a", "anomaly", 0.2),
        ("b/n1", "b", "normal", 0.9), ("b/x1", "b", "anomaly", 0.95),
    ])
    nova = _nova({
        "a/n1": ("a", "normal", ["OK", "OK", "OK"], 0.001),
        "a/n2": ("a", "normal", ["NG", "OK", "OK"], 0.001),
        "a/x1": ("a", "anomaly", ["NG", "NG", "NG"], 0.001),
        "a/x2": ("a", "anomaly", [None, "NG", "NG"], 0.001),
        "b/n1": ("b", "normal", ["OK", "OK", "OK"], 0.002),
        "b/x1": ("b", "anomaly", ["OK", "NG", "NG"], 0.002),
    })
    return ad, nova


def test_join_takes_repeat1_and_majority():
    """CI-F2：主は repeat=1（解析失敗なら None）とその費用、参考は多数決と3回の和。"""
    df = join(*_base())
    assert df.loc["a/n2", "nova1_pred"] == "NG" and df.loc["a/n2", "novamaj_pred"] == "OK"
    assert df.loc["a/x2", "nova1_pred"] is None and df.loc["a/x2", "novamaj_pred"] == "NG"
    assert df.loc["b/x1", "nova1_cost"] == pytest.approx(0.002)
    assert df.loc["b/x1", "novamaj_cost"] == pytest.approx(0.006)


def test_join_stops_on_mismatch():
    """CI-F1：画像の集合・正解の不一致、重複、NaN、Nova の回数の欠けで止める。"""
    ad, nova = _base()
    with pytest.raises(ValueError, match="image sets"):
        join(ad.iloc[1:], nova)
    bad = ad.copy()
    bad.loc[0, "label"] = "anomaly"
    with pytest.raises(ValueError, match="label"):
        join(bad, nova)
    with pytest.raises(ValueError, match="one row"):
        join(pd.concat([ad, ad.iloc[:1]]), nova)
    nan = ad.copy()
    nan.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        join(nan, nova)
    with pytest.raises(ValueError):
        join(ad, nova[~((nova.image_id == "a/n1") & (nova.repeat == 3))])


def test_threshold_at_miss_boundary():
    """CI-F3：不良品 10 枚で目標 10% なら、1枚までは見逃してよい → 2番目に低い値。"""
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05])
    y = np.array([True] * 10 + [False])
    assert threshold_at_miss(s, y, 0.10) == pytest.approx(0.2)
    assert threshold_at_miss(s, y, 0.0999) == pytest.approx(0.1)   # 境界の直前：見逃し 0
    assert threshold_at_miss(s, y, 0.20) == pytest.approx(0.3)


def test_thresholds_per_category_and_global():
    """CI-F3：品目ごとと全体で別のしきい値。"""
    df = join(*_base())
    per = thresholds(df, "category", 0.0)
    assert per["a/n1"] == pytest.approx(0.2) and per["b/n1"] == pytest.approx(0.95)
    glob = thresholds(df, "global", 0.0)
    assert set(glob.round(9)) == {0.2}


def test_top_percent_per_category_ceil_and_ties():
    """CI-F4：k = ceil(X n / 100)。同じ異常度は image_id の昇順。"""
    ad = _ad([("a/1", "a", "normal", 0.5), ("a/2", "a", "normal", 0.5), ("a/3", "a", "anomaly", 0.1),
              ("b/1", "b", "normal", 0.2), ("b/2", "b", "anomaly", 0.9)]).set_index("image_id")
    s = top_percent(ad, "category", 30)  # a: ceil(0.9)=1、b: ceil(0.6)=1
    assert sorted(s[s].index) == ["a/1", "b/2"]
    g = top_percent(ad, "global", 40)    # ceil(2.0)=2
    assert sorted(g[g].index) == ["a/1", "b/2"]
    assert not top_percent(ad, "category", 0).any()
    assert top_percent(ad, "global", 100).all()
    with pytest.raises(ValueError):
        top_percent(ad, "category", 101)


def test_top_percent_exact_when_divisible():
    """CI-F4：100枚の 10% はちょうど 10 枚（浮動小数点の誤差で 11 にしない）。"""
    ad = _ad([(f"a/{i:03d}", "a", "normal", i / 100) for i in range(100)]).set_index("image_id")
    for pct in (10, 20, 30, 50):
        assert int(top_percent(ad, "category", pct).sum()) == pct


def test_evaluate_rest_and_fallback_and_cost():
    """CI-F5・F6：回して Nova が有効なら Nova、解析失敗と回さない画像は rest。費用は回した分だけ。"""
    df = join(*_base())
    send = pd.Series([False, True, True, True, False, True], index=df.index)  # a/n1 a/n2 a/x1 a/x2 b/n1 b/x1
    rest = pd.Series("OK", index=df.index)
    r = evaluate(df, send, rest)
    # repeat=1：a/n2 NG(誤検知)、a/x1 NG、a/x2 None→OK(見逃し)、b/x1 OK(見逃し)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 1, 2, 2)
    assert r["miss_rate"] == pytest.approx(2 / 3)
    assert r["send_rate"] == pytest.approx(4 / 6)
    assert r["cost_per_1000_usd"] == pytest.approx((0.001 * 3 + 0.002) / 6 * 1000)
    # 多数決：a/n2 OK、a/x1 NG、a/x2 NG、b/x1 NG → 見逃し 0、誤検知 0
    assert r["maj_miss_rate"] == 0 and r["maj_false_positive_rate"] == 0
    assert r["maj_cost_per_1000_usd"] == pytest.approx((0.003 * 3 + 0.006) / 6 * 1000)
    rest_ng = pd.Series("NG", index=df.index)
    r2 = evaluate(df, send, rest_ng)
    assert r2["fn"] == 1  # a/x2 は rest の NG で拾える


def test_conditions_cover_main_rows():
    """CI-F6：Nova だけは全件を回し、AnomalyDINO だけは回さず費用 0。"""
    df = join(*_base())
    rows = conditions(df)
    nova = [r for r in rows if r["condition"] == "Nova だけ"][0]
    assert nova["send_rate"] == 1.0
    ad_only = [r for r in rows if r["routing"] == "none"]
    assert ad_only and all(r["cost_per_1000_usd"] == 0 and r["threshold_from_labels"] for r in ad_only)
    top_ok = [r for r in rows if r["routing"] == "top" and r["rest"] == "OK"]
    assert all(not r["threshold_from_labels"] for r in top_ok)


def test_score_summary_auroc():
    """CI-F7：品目ごとと全体の AUROC。"""
    df = join(*_base())
    s = score_summary(df).set_index("category")
    assert s.loc["a", "auroc"] == pytest.approx(0.75)      # 不良 0.5,0.2 対 良品 0.1,0.3：4組のうち3組
    assert s.loc["b", "auroc"] == pytest.approx(1.0)
    assert s.loc["全体", "auroc"] == pytest.approx(6 / 9)
    assert s.loc["a", "normal_median"] == pytest.approx(0.2)
