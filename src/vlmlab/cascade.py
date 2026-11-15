"""段階判定：迷った画像だけ上位モデルに回す（11/26 の記事）。docs/contracts.md の CI-K1〜K12。

API は呼ばない。1段目（Nova 2 Lite の3回、または埋め込みの距離）と2段目（Claude の全件1回）の記録を組み合わせ、
回す条件ごとの F1・誤検知率・回した割合・費用・レイテンシを計算する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .report import TARGET_MISS_RATES
from .run import load_jsonl

NOVA_REPEATS = 3
R2_TAU = 0.8
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
E_PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)
CONF_DIGITS = 6
DIST_DIGITS = 9
# 確信度の帯（下限を含み、上限を含まない。最後の帯だけ 1.0 を含む）
CONF_BANDS = (("0.5未満", 0.0, 0.5), ("0.5〜0.7", 0.5, 0.7), ("0.7〜0.9", 0.7, 0.9), ("0.9以上", 0.9, None))

_miss = M._is_missing


def _none(v: Any) -> Any:
    return None if _miss(v) else v


# ---------------------------------------------------------------- 1枚ごとのまとめ

def summarize_nova(nova: pd.DataFrame) -> pd.DataFrame:
    """Nova の3回を1枚ごとにまとめる（CI-K1 の回数、CI-K2）。"""
    rows = []
    for image_id, g in nova.groupby("image_id", sort=True):
        reps = sorted(g["repeat"].tolist())
        if reps != list(range(1, NOVA_REPEATS + 1)):
            raise ValueError(f"{image_id}: Nova must have repeats 1..{NOVA_REPEATS} exactly once, got {reps}")
        preds = [p for p in g["pred"] if not _miss(p)]
        n_ok, n_ng = preds.count("OK"), preds.count("NG")
        nova_pred = None if n_ok == n_ng else ("NG" if n_ng > n_ok else "OK")
        split = not (len(preds) == NOVA_REPEATS and len(set(preds)) == 1)
        confs = [float(c) for p, c in zip(g["pred"], g["confidence"]) if not _miss(p) and not _miss(c)]
        nova_conf = round(float(np.mean(confs)), CONF_DIGITS) if confs else None
        scores = [M.anomaly_score(p, float(c)) for p, c in zip(g["pred"], g["confidence"])
                  if not _miss(p) and not _miss(c)]
        rows.append({
            "image_id": image_id,
            "label": _single(g, "label", image_id),
            "category": _single(g, "category", image_id),
            "nova_pred": nova_pred,
            "split": split,
            "nova_conf": nova_conf,
            "nova_score": float(np.mean(scores)) if scores else None,
            "nova_cost": _sum_or_nan(g["cost_usd"]),
            "nova_latency": float(g["latency_s"].sum()),
        })
    return pd.DataFrame(rows).set_index("image_id")


def summarize_claude(claude: pd.DataFrame) -> pd.DataFrame:
    """Claude の1回を1枚1行にする（CI-K1 の回数）。"""
    counts = claude.groupby("image_id").size()
    if (counts != 1).any():
        bad = counts[counts != 1].index.tolist()[:5]
        raise ValueError(f"Claude must have exactly one call per image, e.g. {bad}")
    out = claude.set_index("image_id").sort_index()
    score = [M.anomaly_score(p, float(c)) if not _miss(p) and not _miss(c) else None
             for p, c in zip(out["pred"], out["confidence"])]
    return pd.DataFrame({
        "label": out["label"], "category": out["category"],
        "claude_pred": [_none(p) for p in out["pred"]],
        "claude_score": score,
        "claude_cost": out["cost_usd"].astype(float),
        "claude_latency": out["latency_s"].astype(float),
    }, index=out.index)


def _single(g: pd.DataFrame, col: str, image_id: str) -> Any:
    vals = g[col].unique()
    if len(vals) != 1:
        raise ValueError(f"{image_id}: {col} differs between calls: {list(vals)}")
    return vals[0]


def _sum_or_nan(s: pd.Series) -> float:
    return float("nan") if s.map(_miss).any() else float(s.sum())


# ---------------------------------------------------------------- 埋め込みの距離（R6）

def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if (n == 0).any():
        raise ValueError("zero vector in embeddings")
    return x / n


def embedding_distances(emb: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[int, float]], float]:
    """テスト画像の、同じ品目の学習用の良品への最小コサイン距離と、品目ごとのしきい値（CI-K4）。

    戻り値: (テスト画像の表[e_dist, e_cost, e_latency, label, category], {品目: {百分位: しきい値}}, 見本の費用)
    """
    tests, thresholds = [], {}
    for cat, g in emb.groupby("category", sort=True):
        tr = g[(g["role"] == "train") & (g["label"] == "normal")]
        te = g[g["role"] == "test"]
        if len(tr) < 2:
            raise ValueError(f"{cat}: need at least 2 normal train embeddings, got {len(tr)}")
        n = _unit(np.array(tr["vector"].tolist(), dtype=float))
        sim_nn = n @ n.T
        np.fill_diagonal(sim_nn, -np.inf)  # 自分自身は除く
        loo = np.round(1.0 - sim_nn.max(axis=1), DIST_DIGITS)
        thresholds[cat] = {p: round(float(np.percentile(loo, p)), DIST_DIGITS) for p in E_PERCENTILES}
        if len(te):
            t = _unit(np.array(te["vector"].tolist(), dtype=float))
            d = np.round(1.0 - (t @ n.T).max(axis=1), DIST_DIGITS)
            tests.append(pd.DataFrame({
                "image_id": te["image_id"].to_numpy(), "label": te["label"].to_numpy(),
                "category": cat, "e_dist": d,
                "e_cost": te["cost_usd"].astype(float).to_numpy(),
                "e_latency": te["latency_s"].astype(float).to_numpy(),
            }))
    index_cost = _sum_or_nan(emb.loc[emb["role"] == "train", "cost_usd"])
    table = pd.concat(tests).set_index("image_id").sort_index()
    if table.index.duplicated().any():
        raise ValueError("duplicated test image in embeddings")
    return table, thresholds, index_cost


# ---------------------------------------------------------------- 突き合わせ

def align(nova_sum: pd.DataFrame, claude_sum: pd.DataFrame, e_table: pd.DataFrame) -> pd.DataFrame:
    """3つの記録の画像の集合・正解・品目が一致することを確かめて1つの表にする（CI-K1）。"""
    sets = {"nova": set(nova_sum.index), "claude": set(claude_sum.index), "embedding": set(e_table.index)}
    if len({frozenset(s) for s in sets.values()}) != 1:
        diff = {k: len(v ^ sets["nova"]) for k, v in sets.items()}
        raise ValueError(f"image sets differ between records (symmetric difference vs nova): {diff}")
    t = nova_sum.join(claude_sum.drop(columns=["label", "category"]))
    t = t.join(e_table.drop(columns=["label", "category"]))
    for name, other in (("claude", claude_sum), ("embedding", e_table)):
        for col in ("label", "category"):
            if not (nova_sum[col] == other.loc[nova_sum.index, col]).all():
                raise ValueError(f"{col} differs between nova and {name}")
    return t


# ---------------------------------------------------------------- 回す条件

@dataclass(frozen=True)
class Rule:
    name: str
    family: str  # "baseline" / "nova" / "embedding"
    param: float | None
    escalate: pd.Series  # bool
    first_pred: pd.Series  # 1段目の判定（OK/NG/None）
    first_score: pd.Series  # 1段目の業務指標のスコア
    first_cost: pd.Series
    first_latency: pd.Series


def build_rules(t: pd.DataFrame, thresholds: dict[str, dict[int, float]]) -> list[Rule]:
    """CI-K3・K4 の条件と、比較の行（Nova だけ、Claude だけ）。"""
    idx = t.index
    no = pd.Series(False, index=idx)
    conf = t["nova_conf"]
    conf_low = lambda tau: conf.map(lambda c: True if _miss(c) else c < tau)  # noqa: E731
    nova = dict(first_pred=t["nova_pred"], first_score=t["nova_score"],
                first_cost=t["nova_cost"], first_latency=t["nova_latency"])
    rules = [
        Rule("Nova だけ", "baseline", None, no, **nova),
        Rule("R1 割れた", "nova", None, t["split"].astype(bool), **nova),
        Rule(f"R2 確信度<{R2_TAU}", "nova", R2_TAU, conf_low(R2_TAU), **nova),
        Rule("R3 NG と言った", "nova", None, t["nova_pred"] == "NG", **nova),
        Rule("R4 割れた or NG", "nova", None, t["split"].astype(bool) | (t["nova_pred"] == "NG"), **nova),
    ]
    rules += [Rule(f"R5 確信度<{tau}", "nova", tau, conf_low(tau), **nova) for tau in TAUS]
    ok = pd.Series("OK", index=idx, dtype=object)
    zero = pd.Series(0.0, index=idx)
    for p in E_PERCENTILES:
        th = t["category"].map(lambda c: thresholds[c][p])
        rules.append(Rule(f"R6 距離≥良品の{p}%点", "embedding", float(p), t["e_dist"] >= th,
                          first_pred=ok, first_score=zero, first_cost=t["e_cost"], first_latency=t["e_latency"]))
    rules.append(Rule("Claude だけ", "baseline", None, pd.Series(True, index=idx),
                      first_pred=pd.Series(None, index=idx, dtype=object), first_score=pd.Series(np.nan, index=idx),
                      first_cost=zero, first_latency=zero))
    return rules


def final_preds(t: pd.DataFrame, r: Rule) -> pd.Series:
    """CI-K5。"""
    use_claude = r.escalate & t["claude_pred"].map(lambda p: not _miss(p))
    out = [c if u else _none(f) for u, c, f in zip(use_claude, t["claude_pred"], r.first_pred)]
    return pd.Series(out, index=t.index, dtype=object)


def final_scores(t: pd.DataFrame, r: Rule) -> pd.Series:
    """CI-K12。"""
    use_claude = r.escalate & t["claude_score"].map(lambda s: not _miss(s))
    return pd.Series([c if u else f for u, c, f in zip(use_claude, t["claude_score"], r.first_score)],
                     index=t.index, dtype=float)


def evaluate(t: pd.DataFrame, r: Rule) -> dict[str, Any]:
    """CI-K6〜K10。"""
    pred = final_preds(t, r)
    m = M.binary_metrics(pd.DataFrame({"label": t["label"], "pred": pred}))
    cost = r.first_cost + np.where(r.escalate, t["claude_cost"], 0.0)
    latency = r.first_latency + np.where(r.escalate, t["claude_latency"], 0.0)
    lat = M.latency_stats(pd.DataFrame({"latency_s": latency}))
    fb = fixed_broken(t, r)
    return {
        "rule": r.name, "family": r.family, "param": r.param,
        "n_images": len(t), "n_escalated": int(r.escalate.sum()), "escalation_rate": float(r.escalate.mean()),
        **{k: m[k] for k in ("n_valid", "n_parse_fail", "tp", "fp", "tn", "fn", "precision", "recall", "f1",
                             "false_positive_rate", "miss_rate")},
        "cost_per_1000_usd": None if cost.isna().any() else float(cost.mean() * 1000),
        "latency_p50_s": lat["latency_p50_s"], "latency_p95_s": lat["latency_p95_s"],
        **fb,
    }


def _correct(pred: Any, label: str) -> bool | None:
    if _miss(pred):
        return None
    return (pred == "NG") == (label == "anomaly")


def fixed_broken(t: pd.DataFrame, r: Rule) -> dict[str, int]:
    """CI-K10。回した画像のうち、1段目と Claude の両方が有効なものだけを数える。"""
    out = {"fixed": 0, "broken": 0, "both_right": 0, "both_wrong": 0}
    for esc, f, c, lab in zip(r.escalate, r.first_pred, t["claude_pred"], t["label"]):
        if not esc:
            continue
        cf, cc = _correct(f, lab), _correct(c, lab)
        if cf is None or cc is None:
            continue
        key = {(False, True): "fixed", (True, False): "broken", (True, True): "both_right",
               (False, False): "both_wrong"}[(cf, cc)]
        out[key] += 1
    return out


# ---------------------------------------------------------------- 表

def confidence_bands(t: pd.DataFrame) -> pd.DataFrame:
    """CI-K11。"""
    rows = []
    for name, lo, hi in CONF_BANDS:
        c = t["nova_conf"]
        mask = c.map(lambda v: (not _miss(v)) and v >= lo and (hi is None or v < hi))
        sub = t[mask]
        right = [x for x in (_correct(p, lab) for p, lab in zip(sub["nova_pred"], sub["label"])) if x is not None]
        rows.append({"band": name, "lower": lo, "upper": hi, "n": int(mask.sum()), "n_judged": len(right),
                     "accuracy": (sum(right) / len(right)) if right else None,
                     "n_anomaly": int((sub["label"] == "anomaly").sum())})
    n_none = int(t["nova_conf"].map(_miss).sum())
    rows.append({"band": "確信度なし", "lower": None, "upper": None, "n": n_none, "n_judged": None,
                 "accuracy": None, "n_anomaly": int((t.loc[t["nova_conf"].map(_miss), "label"] == "anomaly").sum())})
    return pd.DataFrame(rows)


def business_metric(t: pd.DataFrame, rules: list[Rule], best: str) -> pd.DataFrame:
    """CI-K12。Nova だけ、Claude だけ、F1 が最大の段階判定。"""
    by = {r.name: r for r in rules}
    rows = []
    for name in ("Nova だけ", "Claude だけ", best):
        s = final_scores(t, by[name])
        df = pd.DataFrame({"label": t["label"],
                           "pred": ["NG" if not _miss(v) else None for v in s],
                           "confidence": s})
        for target in TARGET_MISS_RATES:
            rows.append({"rule": name, **M.pass_rate_at_miss(df, target)})
    return pd.DataFrame(rows)


def pick_best(results: pd.DataFrame) -> str:
    """F1 が最大の段階判定（同じなら1,000枚あたりの費用が安いほう）。比較の行は除く。"""
    c = results[results["family"] != "baseline"].dropna(subset=["f1"])
    c = c.sort_values(["f1", "cost_per_1000_usd"], ascending=[False, True], kind="stable")
    return str(c.iloc[0]["rule"])


def image_table(t: pd.DataFrame) -> pd.DataFrame:
    """1枚ごとの表（直った・悪くなった画像を探す用）。Nova → Claude に替えたときの変化を書く。"""
    out = t[["category", "label", "nova_pred", "split", "nova_conf", "claude_pred", "e_dist"]].copy()

    def change(f: Any, c: Any, lab: str) -> str | None:
        cf, cc = _correct(f, lab), _correct(c, lab)
        if cf is None or cc is None:
            return None
        return {(False, True): "fixed", (True, False): "broken", (True, True): "both_right",
                (False, False): "both_wrong"}[(cf, cc)]

    out["nova_to_claude"] = [change(f, c, lab) for f, c, lab in zip(t["nova_pred"], t["claude_pred"], t["label"])]
    out["ok_to_claude"] = [change("OK", c, lab) for c, lab in zip(t["claude_pred"], t["label"])]
    return out


# ---------------------------------------------------------------- 入出力

def load_embeddings(path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(load_jsonl(Path(path)))
    df = df[df["api_error"].isna()].drop_duplicates(["image_id", "role"], keep="last")
    return df.reset_index(drop=True)


def run_cascade(nova: pd.DataFrame, claude: pd.DataFrame, emb: pd.DataFrame) -> dict[str, Any]:
    e_table, thresholds, index_cost = embedding_distances(emb)
    t = align(summarize_nova(nova), summarize_claude(claude), e_table)
    rules = build_rules(t, thresholds)
    results = pd.DataFrame([evaluate(t, r) for r in rules])
    best = pick_best(results)
    rule_rows = {r.name: r for r in rules}
    fb = results[["rule", "family", "param", "n_escalated", "fixed", "broken", "both_right", "both_wrong"]]
    th = pd.DataFrame([{"category": c, "percentile": p, "threshold": v}
                       for c, d in thresholds.items() for p, v in d.items()])
    return {
        "table": t, "rules": rule_rows, "results": results, "best": best,
        "fixed_broken": fb, "confidence_bands": confidence_bands(t),
        "business": business_metric(t, rules, best), "images": image_table(t),
        "e_thresholds": th, "e_index_cost_usd": index_cost,
    }


def _fmt(v: Any, digits: int = 3) -> str:
    if _miss(v):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


def to_markdown(df: pd.DataFrame, cols: list[tuple[str, str, int]]) -> str:
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[k], d) for k, _, d in cols) + " |")
    return "\n".join(lines) + "\n"


RULE_COLS = [("rule", "条件", 0), ("n_escalated", "回した枚数", 0), ("escalation_rate", "回した割合", 3),
             ("n_valid", "有効件数", 0), ("precision", "適合率", 3), ("recall", "再現率", 3), ("f1", "F1", 3),
             ("false_positive_rate", "誤検知率", 3), ("miss_rate", "見逃し率", 3),
             ("cost_per_1000_usd", "1,000枚あたり (USD)", 3), ("latency_p50_s", "p50 (秒)", 2),
             ("latency_p95_s", "p95 (秒)", 2), ("fixed", "直った", 0), ("broken", "悪くなった", 0)]


def write_outputs(res: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = {
        "cascade_rules": out / "cascade_rules.csv", "cascade_rules_md": out / "cascade_rules.md",
        "fixed_and_broken": out / "fixed_and_broken.csv", "cascade_images": out / "cascade_images.csv",
        "confidence_bands": out / "confidence_bands.csv", "business_metric": out / "business_metric.csv",
        "e_thresholds": out / "e_thresholds.csv",
    }
    res["results"].to_csv(p["cascade_rules"], index=False, encoding="utf-8")
    md = to_markdown(res["results"], RULE_COLS)
    md += (f"\n一番よかった段階判定（F1 最大、同じなら安いほう）: {res['best']}\n"
           f"埋め込みの見本（学習用の画像）の費用は1回きりなので表に含めない: {_fmt(res['e_index_cost_usd'], 4)} USD\n")
    p["cascade_rules_md"].write_text(md, encoding="utf-8")
    res["fixed_broken"].to_csv(p["fixed_and_broken"], index=False, encoding="utf-8")
    res["images"].to_csv(p["cascade_images"], encoding="utf-8")
    res["confidence_bands"].to_csv(p["confidence_bands"], index=False, encoding="utf-8")
    res["business"].to_csv(p["business_metric"], index=False, encoding="utf-8")
    res["e_thresholds"].to_csv(p["e_thresholds"], index=False, encoding="utf-8")
    return p
