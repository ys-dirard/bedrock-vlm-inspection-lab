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
    accuracy = _div(tp + tn, tp + fp + tn + fn)  # CI-Y3
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
        "accuracy": accuracy,
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


def _validate_scores(scores: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    """CI-Y1。"""
    s = np.asarray(list(scores), dtype=float)
    lab = list(labels)
    if len(s) != len(lab):
        raise ValueError(f"scores and labels differ in length: {len(s)} vs {len(lab)}")
    bad = set(map(str, lab)) - LABELS
    if bad:
        raise ValueError(f"unexpected label values: {sorted(bad)}")
    if len(s) and not np.isfinite(s).all():
        raise ValueError("scores must be finite numbers")
    return s, np.array([x == "anomaly" for x in lab], dtype=bool)


def pass_rate_at_miss_scores(scores: Any, labels: Any, target_miss_rate: float) -> dict[str, Any]:
    """任意のスコアで、見逃し率を target 以下にしたときの良品の通過率（CI-Y2。規則は CI-M6 と同じ）。

    スコアが t 以上を NG とみなす。見逃し率が target 以下になる最大の t を選ぶ。スコアが t と同じ良品は通さない。
    """
    s, is_anom = _validate_scores(scores, labels)
    s_anom, s_norm = s[is_anom], s[~is_anom]
    base = {"target_miss_rate": target_miss_rate, "threshold": None, "pass_rate": None,
            "achieved_miss_rate": None, "n_normal": int(len(s_norm)), "n_anomaly": int(len(s_anom)),
            "n_normal_passed": None, "reason": None}
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
    n_pass = int((s_norm < t).sum())
    return {**base, "threshold": t, "pass_rate": n_pass / len(s_norm), "achieved_miss_rate": miss,
            "n_normal_passed": n_pass}


def pass_rate_at_miss(df: pd.DataFrame, target_miss_rate: float) -> dict[str, Any]:
    """見逃し率を target 以下にしたときの良品の通過率（CI-M6、CI-M7）。スコアは D1。"""
    validate(df)
    v = df[~df["pred"].map(_is_missing)]
    if v["confidence"].map(_is_missing).any():
        return {"target_miss_rate": target_miss_rate, "threshold": None, "pass_rate": None,
                "achieved_miss_rate": None, "n_normal": int((v["label"] == "normal").sum()),
                "n_anomaly": int((v["label"] == "anomaly").sum()), "n_normal_passed": None,
                "reason": "confidence missing in some valid responses"}
    scores = [anomaly_score(p, float(c)) for p, c in zip(v["pred"], v["confidence"])]
    return pass_rate_at_miss_scores(scores, v["label"].tolist(), target_miss_rate)


def operating_points(scores: Any, labels: Any) -> pd.DataFrame:
    """しきい値ごとの見逃し率と良品の通過率（CI-Y5）。候補は全スコアの異なる値と +∞。"""
    s, is_anom = _validate_scores(scores, labels)
    s_anom, s_norm = s[is_anom], s[~is_anom]
    rows = []
    for t in sorted(set(s.tolist())) + [math.inf]:
        rows.append({
            "threshold": t,
            "miss_rate": float((s_anom < t).mean()) if len(s_anom) else None,
            "pass_rate": float((s_norm < t).mean()) if len(s_norm) else None,
        })
    return pd.DataFrame(rows)


def ng_counts(df: pd.DataFrame, n_repeats: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """画像ごとの「NG と言った回数」（CI-Y4）。どれかの回が解析失敗の画像は除く（Y-D2）。

    戻り値: (image_id を索引にした表[label, ng_count], 除いた image_id のリスト)
    """
    validate(df)
    rows, excluded = [], []
    for image_id, g in df.groupby("image_id", sort=True):
        reps = sorted(g["repeat"].tolist())
        if reps != list(range(1, n_repeats + 1)):
            raise ValueError(f"{image_id}: repeats must be 1..{n_repeats} exactly once, got {reps}")
        labels = g["label"].unique()
        if len(labels) != 1:
            raise ValueError(f"{image_id}: label differs between calls: {list(labels)}")
        if g["pred"].map(_is_missing).any():
            excluded.append(str(image_id))
            continue
        rows.append({"image_id": image_id, "label": labels[0], "ng_count": int((g["pred"] == "NG").sum())})
    out = pd.DataFrame(rows, columns=["image_id", "label", "ng_count"]).set_index("image_id")
    return out, excluded


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
