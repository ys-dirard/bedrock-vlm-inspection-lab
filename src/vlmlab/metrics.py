"""評価指標（docs/contracts.md の CI-M1〜M10）。

入力は1呼び出し1行の DataFrame。必要な列:
  image_id, label ("normal"/"anomaly"), repeat, pred ("OK"/"NG"/None), confidence, latency_s, cost_usd
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

LABELS = {"normal", "anomaly"}
PREDS = {"OK", "NG"}
RATE_KEYS = ["precision", "recall", "f1", "false_positive_rate", "miss_rate"]


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def validate(df: pd.DataFrame) -> None:
    """CI-M1。"""
    bad_label = set(df["label"].unique()) - LABELS
    if bad_label:
        raise ValueError(f"unexpected label values: {sorted(map(str, bad_label))}")
    bad_pred = {p for p in df["pred"] if not _is_missing(p) and p not in PREDS}
    if bad_pred:
        raise ValueError(f"unexpected pred values: {sorted(map(str, bad_pred))}")


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def binary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """NG を陽性とした指標。解析失敗（pred が None）は除き、件数を併記する（CI-M2〜M4）。"""
    validate(df)
    n_total = len(df)
    valid_mask = ~df["pred"].map(_is_missing)
    v = df[valid_mask]
    pos = v["label"] == "anomaly"
    ng = v["pred"] == "NG"
    tp = int((pos & ng).sum())
    fn = int((pos & ~ng).sum())
    fp = int((~pos & ng).sum())
    tn = int((~pos & ~ng).sum())
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    n_fail = n_total - int(valid_mask.sum())
    return {
        "n_total": n_total,
        "n_valid": int(valid_mask.sum()),
        "n_parse_fail": n_fail,
        "parse_fail_rate": _div(n_fail, n_total),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _div(fp, fp + tn),
        "miss_rate": _div(fn, tp + fn),
    }


def per_repeat(df: pd.DataFrame) -> pd.DataFrame:
    """回ごとの指標（CI-M5）。"""
    rows = []
    for rep, g in df.groupby("repeat", sort=True):
        rows.append({"repeat": rep, **binary_metrics(g)})
    return pd.DataFrame(rows)


def aggregate_repeats(per_rep: pd.DataFrame, keys: list[str] | None = None) -> dict[str, Any]:
    """回ごとの指標の平均・最小・最大。None の回は平均に入れず、使った回数を書く（CI-M5）。"""
    keys = keys or RATE_KEYS + ["parse_fail_rate"]
    out: dict[str, Any] = {"n_repeats": len(per_rep)}
    for k in keys:
        vals = [x for x in per_rep[k].tolist() if not _is_missing(x)]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{k}_min"] = float(min(vals)) if vals else None
        out[f"{k}_max"] = float(max(vals)) if vals else None
        out[f"{k}_n"] = len(vals)
    for k in ("n_total", "n_valid", "n_parse_fail"):
        out[f"{k}_sum"] = int(per_rep[k].sum())
    return out


SCORE_DIGITS = 9  # 1 − 0.7 = 0.30000000000000004 を 0.3 と同じに扱うため丸める


def anomaly_score(pred: str, confidence: float) -> float:
    """不良らしさ。NG なら c、OK なら 1 − c（D1、CI-M6）。"""
    return round(confidence if pred == "NG" else 1.0 - confidence, SCORE_DIGITS)


def pass_rate_at_miss(df: pd.DataFrame, target_miss_rate: float) -> dict[str, Any]:
    """見逃し率を target 以下にしたときの良品の通過率（CI-M6、CI-M7）。

    スコアが t 以上を NG とみなす。見逃し率が target 以下になる最大の t を選ぶ。
    """
    validate(df)
    v = df[~df["pred"].map(_is_missing)]
    base = {"target_miss_rate": target_miss_rate, "threshold": None, "pass_rate": None,
            "achieved_miss_rate": None, "n_normal": 0, "n_anomaly": 0, "reason": None}
    if v["confidence"].map(_is_missing).any():
        return {**base, "reason": "confidence missing in some valid responses"}
    scores = np.array([anomaly_score(p, float(c)) for p, c in zip(v["pred"], v["confidence"])])
    is_anom = (v["label"] == "anomaly").to_numpy()
    s_anom, s_norm = scores[is_anom], scores[~is_anom]
    base.update(n_normal=len(s_norm), n_anomaly=len(s_anom))
    if len(s_anom) == 0 or len(s_norm) == 0:
        return {**base, "reason": "no anomaly or no normal samples"}

    # 候補のしきい値は不良品のスコアの値。t を上げるほど通過は増え、見逃しも増える。
    best = None
    for t in sorted(set(s_anom.tolist())):
        miss = float((s_anom < t).mean())
        if miss <= target_miss_rate:
            best = (t, miss)
    if best is None:  # 最小の t では見逃しが 0 なので、ここには来ない
        return {**base, "reason": "no threshold satisfies target"}
    t, miss = best
    return {**base, "threshold": t, "pass_rate": float((s_norm < t).mean()), "achieved_miss_rate": miss}


def agreement_rate(df: pd.DataFrame) -> dict[str, Any]:
    """一致率：全回で同じ判定だった画像 / 2回以上判定した画像。解析失敗の回があれば不一致（CI-M8）。"""
    validate(df)
    n_img = 0
    n_agree = 0
    for _, g in df.groupby("image_id"):
        if len(g) < 2:
            continue
        n_img += 1
        preds = g["pred"].tolist()
        if any(_is_missing(p) for p in preds):
            continue
        if len(set(preds)) == 1:
            n_agree += 1
    return {"agreement_rate": _div(n_agree, n_img), "n_images_repeated": n_img, "n_agree": n_agree}


def latency_stats(df: pd.DataFrame) -> dict[str, Any]:
    """p50・p95（線形補間）。解析失敗の呼び出しも含む（CI-M9）。"""
    lat = df["latency_s"].dropna().to_numpy(dtype=float)
    if len(lat) == 0:
        return {"latency_p50_s": None, "latency_p95_s": None, "n_latency": 0}
    return {
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "n_latency": int(len(lat)),
    }


def cost_stats(df: pd.DataFrame) -> dict[str, Any]:
    """1件あたり・1,000枚あたりの費用。不明な費用があれば None（CI-M10）。"""
    costs = df["cost_usd"].tolist()
    if not costs or any(_is_missing(c) for c in costs):
        return {"cost_total_usd": None, "cost_per_call_usd": None, "cost_per_1000_usd": None}
    total = float(sum(costs))
    per_call = total / len(costs)
    return {"cost_total_usd": total, "cost_per_call_usd": per_call, "cost_per_1000_usd": per_call * 1000}


def token_stats(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for k in ("input_tokens", "output_tokens"):
        if k in df:
            s = df[k].dropna()
            out[f"{k}_mean"] = float(s.mean()) if len(s) else None
    return out
