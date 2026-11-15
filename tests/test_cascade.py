import numpy as np
import pandas as pd
import pytest

from vlmlab import cascade as C


def _nova(rows):
    """rows: [(image_id, label, [(pred, conf), ×3])] → 1呼び出し1行。"""
    out = []
    for image_id, label, calls in rows:
        for rep, (pred, conf) in enumerate(calls, start=1):
            out.append({"image_id": image_id, "label": label, "category": "pcb1", "repeat": rep,
                        "pred": pred, "confidence": conf, "cost_usd": 0.001, "latency_s": 1.0})
    return pd.DataFrame(out)


def _claude(rows):
    """rows: [(image_id, label, pred, conf)]"""
    return pd.DataFrame([{"image_id": i, "label": lab, "category": "pcb1", "repeat": 1, "pred": p,
                          "confidence": c, "cost_usd": 0.01, "latency_s": 3.0} for i, lab, p, c in rows])


def _emb(train_normals, tests):
    """train_normals: [vector]、tests: [(image_id, label, vector)]"""
    rows = [{"image_id": f"tr{i}", "role": "train", "category": "pcb1", "label": "normal", "vector": v,
             "cost_usd": 0.00006, "latency_s": 0.5} for i, v in enumerate(train_normals)]
    rows += [{"image_id": i, "role": "test", "category": "pcb1", "label": lab, "vector": v,
              "cost_usd": 0.00006, "latency_s": 0.5} for i, lab, v in tests]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- CI-K1

def test_nova_must_have_three_repeats():
    """CI-K1：Nova の回数が3でなければ止める（欠けを黙って除かない）"""
    df = _nova([("a", "normal", [("OK", 1.0), ("OK", 1.0), ("OK", 1.0)])])
    with pytest.raises(ValueError, match="repeats"):
        C.summarize_nova(df[df["repeat"] != 2])


def test_claude_must_have_one_call_per_image():
    """CI-K1"""
    df = _claude([("a", "normal", "OK", 0.9), ("a", "normal", "OK", 0.9)])
    with pytest.raises(ValueError, match="exactly one"):
        C.summarize_claude(df)


def test_align_rejects_different_image_sets_and_labels():
    """CI-K1：画像の集合・正解が記録の間で違えば止める"""
    nova = C.summarize_nova(_nova([("a", "normal", [("OK", 1.0)] * 3), ("b", "anomaly", [("OK", 1.0)] * 3)]))
    claude = C.summarize_claude(_claude([("a", "normal", "OK", 0.9)]))
    e, _, _ = C.embedding_distances(_emb([[1, 0], [0, 1]], [("a", "normal", [1, 0]), ("b", "anomaly", [0, 1])]))
    with pytest.raises(ValueError, match="image sets differ"):
        C.align(nova, claude, e)
    claude2 = C.summarize_claude(_claude([("a", "normal", "OK", 0.9), ("b", "normal", "OK", 0.9)]))
    with pytest.raises(ValueError, match="label differs"):
        C.align(nova, claude2, e)


# ---------------------------------------------------------------- CI-K2

def test_nova_majority_split_and_rounded_conf():
    """CI-K2：多数決、割れ、確信度の平均を6桁に丸める（0.95 の平均が 0.95 未満にならない）"""
    s = C.summarize_nova(_nova([
        ("a", "anomaly", [("NG", 0.95), ("OK", 0.95), ("OK", 0.95)]),
        ("b", "normal", [("OK", 0.95), ("OK", 0.95), ("OK", 0.95)]),
    ]))
    assert s.loc["a", "nova_pred"] == "OK" and bool(s.loc["a", "split"]) is True
    assert s.loc["b", "nova_pred"] == "OK" and bool(s.loc["b", "split"]) is False
    assert s.loc["b", "nova_conf"] == 0.95 and not s.loc["b", "nova_conf"] < 0.95
    assert s.loc["b", "nova_cost"] == pytest.approx(0.003) and s.loc["b", "nova_latency"] == 3.0


def test_nova_parse_failure_tie_and_all_failed():
    """CI-K2：解析失敗の回は多数決から除き、割れたとみなす。OK と NG が同数・全滅なら None"""
    s = C.summarize_nova(_nova([
        ("tie", "normal", [("OK", 0.9), ("NG", 0.8), (None, None)]),
        ("one", "normal", [("OK", 0.9), (None, None), (None, None)]),
        ("none", "normal", [(None, None)] * 3),
    ]))
    assert pd.isna(s.loc["tie", "nova_pred"]) and bool(s.loc["tie", "split"])
    assert s.loc["one", "nova_pred"] == "OK" and bool(s.loc["one", "split"])
    assert pd.isna(s.loc["none", "nova_pred"]) and pd.isna(s.loc["none", "nova_conf"])


# ---------------------------------------------------------------- CI-K4

def test_embedding_distance_and_loo_thresholds():
    """CI-K4：テストは学習用の良品への最小コサイン距離。しきい値は良品どうしの自分を除いた最小距離の百分位"""
    tr = [[1, 0], [1, 0.0001], [0, 1]]
    e, th, index_cost = C.embedding_distances(_emb(tr, [("x", "anomaly", [-1, 0]), ("y", "normal", [2, 0])]))
    assert e.loc["y", "e_dist"] == pytest.approx(0.0, abs=1e-9)
    assert e.loc["x", "e_dist"] == pytest.approx(1.0, abs=1e-6)   # [0,1] との距離が最小（1 − 0）
    # 自分を除いた最小距離：tr0≈0、tr1≈0、tr2≈1 − 0.0001/1 ≈ 0.9999
    assert th["pcb1"][0] == pytest.approx(0.0, abs=1e-6)
    assert th["pcb1"][100] == pytest.approx(1 - 0.0001 / np.hypot(1, 0.0001), abs=1e-6)
    assert index_cost == pytest.approx(3 * 0.00006)


def test_embedding_rejects_zero_vector():
    """CI-K4"""
    with pytest.raises(ValueError, match="zero vector"):
        C.embedding_distances(_emb([[1, 0], [0, 0]], [("x", "normal", [1, 0])]))


# ---------------------------------------------------------------- CI-K3, K5〜K10

@pytest.fixture
def table():
    nova = _nova([
        ("n1", "normal", [("OK", 1.0)] * 3),                                   # 正しい OK
        ("n2", "normal", [("NG", 0.9)] * 3),                                   # 誤検知
        ("a1", "anomaly", [("OK", 1.0)] * 3),                                  # 自信満々の見逃し
        ("a2", "anomaly", [("NG", 0.6), ("OK", 0.7), ("OK", 0.7)]),            # 割れた見逃し
    ])
    claude = _claude([("n1", "normal", "NG", 0.8),     # Claude で悪くなる
                      ("n2", "normal", "OK", 0.9),     # 直る
                      ("a1", "anomaly", "NG", 0.7),    # 直る
                      ("a2", "anomaly", None, None)])  # 解析失敗
    emb = _emb([[1, 0], [1, 0.01]], [("n1", "normal", [1, 0]), ("n2", "normal", [1, 0.005]),
                                     ("a1", "anomaly", [0, 1]), ("a2", "anomaly", [1, 1])])
    e, th, _ = C.embedding_distances(emb)
    t = C.align(C.summarize_nova(nova), C.summarize_claude(claude), e)
    return t, {r.name: r for r in C.build_rules(t, th)}


def test_rules_select_expected_images(table):
    """CI-K3"""
    t, rules = table
    esc = lambda name: set(t.index[rules[name].escalate])  # noqa: E731
    assert esc("R1 割れた") == {"a2"}
    assert esc("R3 NG と言った") == {"n2"}
    assert esc("R4 割れた or NG") == {"a2", "n2"}
    assert esc("R2 確信度<0.8") == {"a2"}                    # a2 の平均 0.666667
    assert esc("R5 確信度<1.0") == {"a2", "n2"}
    assert esc("R5 確信度<0.5") == set()
    assert esc("Nova だけ") == set() and esc("Claude だけ") == set(t.index)


def test_final_pred_falls_back_to_first_stage_on_claude_failure(table):
    """CI-K5：回したが Claude が解析失敗なら1段目の判定"""
    t, rules = table
    pred = C.final_preds(t, rules["R4 割れた or NG"])
    assert pred["n2"] == "OK"         # Claude
    assert pred["a2"] == "OK"         # Claude 失敗 → Nova の多数決
    assert pred["a1"] == "OK"         # 回さない
    claude_only = C.final_preds(t, rules["Claude だけ"])
    assert claude_only["a2"] is None  # 1段目が無いので解析失敗のまま（D3 で除く）


def test_evaluate_metrics_cost_latency_and_fixed_broken(table):
    """CI-K6〜K10"""
    t, rules = table
    r = C.evaluate(t, rules["R4 割れた or NG"])
    assert r["n_escalated"] == 2 and r["escalation_rate"] == 0.5
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (0, 0, 2, 2)
    assert r["precision"] is None and r["f1"] is None  # TP+FP=0（CI-M3、M4）
    assert r["recall"] == 0.0 and r["false_positive_rate"] == 0.0
    # 費用：Nova 3回 × 0.001 × 4枚 ＋ Claude 0.01 × 2枚 → 平均 × 1,000
    assert r["cost_per_1000_usd"] == pytest.approx((4 * 0.003 + 2 * 0.01) / 4 * 1000)
    # レイテンシ：回さない画像 3.0 秒、回した画像 6.0 秒
    assert r["latency_p50_s"] == pytest.approx(4.5)
    # 直った：n2（Nova NG → Claude OK）。a2 は Claude が失敗なので数えない
    assert (r["fixed"], r["broken"], r["both_right"], r["both_wrong"]) == (1, 0, 0, 0)
    c = C.evaluate(t, rules["Claude だけ"])
    assert c["n_valid"] == 3 and c["n_parse_fail"] == 1
    assert c["cost_per_1000_usd"] == pytest.approx(10.0)
    assert (c["fixed"], c["broken"]) == (0, 0)  # 1段目が無いので数えない


def test_embedding_rule_uses_ok_as_first_stage(table):
    """CI-K4, K5, K8：R6 は回さない画像を OK とし、1段目の費用は埋め込み"""
    t, rules = table
    r = rules["R6 距離≥良品の100%点"]
    assert set(t.index[r.escalate]) == {"a1", "a2"}
    pred = C.final_preds(t, r)
    assert pred["a1"] == "NG" and pred["a2"] == "OK" and pred["n1"] == "OK"
    ev = C.evaluate(t, r)
    assert ev["cost_per_1000_usd"] == pytest.approx((4 * 0.00006 + 2 * 0.01) / 4 * 1000)
    assert (ev["fixed"], ev["broken"]) == (1, 0)  # a1：OK → NG（a2 は Claude 失敗で数えない）


def test_cost_none_when_any_cost_missing(table):
    """CI-K8：費用が不明な行があれば None"""
    t, rules = table
    t = t.copy()
    t.loc["n1", "claude_cost"] = np.nan
    assert C.evaluate(t, rules["Claude だけ"])["cost_per_1000_usd"] is None


# ---------------------------------------------------------------- CI-K11, K12

def test_confidence_bands_counts_and_empty_band(table):
    """CI-K11：件数 0 の帯の正解率は None。上限は含まず、最後の帯は 1.0 を含む"""
    t, _ = table
    b = C.confidence_bands(t).set_index("band")
    assert b.loc["0.5〜0.7", "n"] == 1 and b.loc["0.5〜0.7", "accuracy"] == 0.0   # a2 0.666667、見逃し
    assert b.loc["0.9以上", "n"] == 3 and b.loc["0.9以上", "accuracy"] == pytest.approx(1 / 3)
    assert b.loc["0.5未満", "n"] == 0 and pd.isna(b.loc["0.5未満", "accuracy"])


def test_business_scores_follow_escalation(table):
    """CI-K12：回して Claude が有効なら Claude のスコア、それ以外は1段目（R6 は 0）"""
    t, rules = table
    s = C.final_scores(t, rules["R6 距離≥良品の100%点"])
    assert s["a1"] == pytest.approx(0.7) and s["a2"] == 0.0 and s["n1"] == 0.0
    n = C.final_scores(t, rules["Nova だけ"])
    assert n["a2"] == pytest.approx(np.mean([0.6, 0.3, 0.3]))
    bm = C.business_metric(t, list(rules.values()), "R4 割れた or NG")
    assert list(bm["rule"].unique()) == ["Nova だけ", "Claude だけ", "R4 割れた or NG"]
    assert set(bm["target_miss_rate"]) == {0.02, 0.05}


def test_pick_best_excludes_baselines_and_breaks_ties_by_cost():
    """CI-K12"""
    df = pd.DataFrame([
        {"rule": "Claude だけ", "family": "baseline", "f1": 0.9, "cost_per_1000_usd": 9.0},
        {"rule": "A", "family": "nova", "f1": 0.5, "cost_per_1000_usd": 3.0},
        {"rule": "B", "family": "embedding", "f1": 0.5, "cost_per_1000_usd": 1.0},
        {"rule": "C", "family": "nova", "f1": None, "cost_per_1000_usd": 0.1},
    ])
    assert C.pick_best(df) == "B"
