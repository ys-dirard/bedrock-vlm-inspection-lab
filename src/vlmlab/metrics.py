"""評価指標（docs/contracts.md の CI-M1〜M10。確信度の較正は CI-CB1〜CB8）。

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
    if v["confidence"].map(_is_missing).any():
        return {"target_miss_rate": target_miss_rate, "threshold": None, "pass_rate": None,
                "achieved_miss_rate": None, "n_normal": 0, "n_anomaly": 0,
                "reason": "confidence missing in some valid responses"}
    scores = np.array([anomaly_score(p, float(c)) for p, c in zip(v["pred"], v["confidence"])])
    return pass_rate_at_miss_scores(scores, (v["label"] == "anomaly").to_numpy(), target_miss_rate)


def pass_rate_at_miss_scores(scores: np.ndarray, is_anom: np.ndarray, target_miss_rate: float) -> dict[str, Any]:
    """CI-M6 の計算を、不良らしさのスコアから直接行う（一致率のスコアにも使う。CI-CB8）。"""
    scores = np.round(np.asarray(scores, dtype=float), SCORE_DIGITS)
    is_anom = np.asarray(is_anom, dtype=bool)
    base = {"target_miss_rate": target_miss_rate, "threshold": None, "pass_rate": None,
            "achieved_miss_rate": None, "n_normal": 0, "n_anomaly": 0, "reason": None}
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


# ---- 確信度の較正（12/12 の記事。docs/contracts.md の CI-CB1〜CB8） ----

N_CONF_BINS = 10   # 帯は [0, 0.1), [0.1, 0.2), …, [0.9, 1.0]（CI-CB2）
FEW_BIN_N = 10     # この件数より少ない帯は、信頼度図で「件数が少ない」と注記する（CI-CB3）


def _check_unit_interval(x: np.ndarray, what: str) -> None:
    """CI-CB1"""
    if np.isnan(x).any() or (x < 0).any() or (x > 1).any():
        raise ValueError(f"{what} must be numbers in [0, 1]")


def conf_bin_index(conf: np.ndarray, n_bins: int = N_CONF_BINS) -> np.ndarray:
    """帯の番号。k/n_bins ≤ c < (k+1)/n_bins を k、c = 1 は最後の帯（CI-CB2）。

    0.7 × 10 = 7.000000000000001 のような誤差で帯がずれないよう、c × n_bins を小数9桁に丸めてから切り捨てる。
    """
    conf = np.asarray(conf, dtype=float)
    _check_unit_interval(conf, "confidence")
    idx = np.floor(np.round(conf * n_bins, SCORE_DIGITS)).astype(int)
    return np.minimum(idx, n_bins - 1)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_CONF_BINS,
                     few_n: int = FEW_BIN_N) -> pd.DataFrame:
    """信頼度図の表（CI-CB3）。帯ごとの件数・平均確信度・正解率・差（正解率 − 平均確信度）と、件数の少ない帯の印。"""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(conf) != len(correct):
        raise ValueError("conf and correct must have the same length")
    idx = conf_bin_index(conf, n_bins)
    rows = []
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        mean_conf = float(conf[m].mean()) if n else None
        acc = float(correct[m].mean()) if n else None
        rows.append({
            "bin": k, "lower": k / n_bins, "upper": (k + 1) / n_bins, "n": n,
            "mean_conf": mean_conf, "accuracy": acc,
            "gap": None if n == 0 else acc - mean_conf,
            "few": bool(0 < n < few_n),
        })
    return pd.DataFrame(rows)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_CONF_BINS, few_n: int = FEW_BIN_N) -> dict[str, Any]:
    """期待較正誤差 ECE = Σ_b (n_b / N) |正解率_b − 平均確信度_b|（CI-CB4）。

    あわせて全体の平均確信度・正解率と、過信の度合い（平均確信度 − 正解率）を返す。N = 0 なら None。
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(conf)
    if n == 0:
        return {"ece": None, "n": 0, "mean_conf": None, "accuracy": None, "overconfidence": None,
                "n_bins_used": 0, "n_bins_few": 0}
    bins = reliability_bins(conf, correct, n_bins, few_n)
    used = bins[bins["n"] > 0]
    value = float(sum(r.n / n * abs(r.accuracy - r.mean_conf) for r in used.itertuples()))
    mean_conf, acc = float(conf.mean()), float(correct.mean())
    return {"ece": value, "n": n, "mean_conf": mean_conf, "accuracy": acc, "overconfidence": mean_conf - acc,
            "n_bins_used": int(len(used)), "n_bins_few": int(bins["few"].sum())}


def _truth(label: str) -> str:
    return "NG" if label == "anomaly" else "OK"


VIEWS = ("judgement", "anomaly")


def self_report_points(df: pd.DataFrame, view: str = "judgement") -> tuple[pd.DataFrame | None, str | None]:
    """自己申告の確信度を、較正を測る点（conf, correct）にする（CI-CB5）。

    view="judgement"：conf = 申告した確信度、correct = 判定が正解か。
    view="anomaly"  ：conf = 不良らしさ（D1 のスコア）、correct = 不良品か。
    解析失敗の行は除く。有効な行に確信度の無いものが1つでもあれば (None, 理由) を返す。
    """
    validate(df)
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    v = df[~df["pred"].map(_is_missing)]
    if v["confidence"].map(_is_missing).any():
        return None, "confidence missing in some valid responses"
    c = v["confidence"].astype(float).to_numpy()
    _check_unit_interval(c, "confidence")
    if view == "judgement":
        conf = c
        correct = (v["pred"] == v["label"].map(_truth)).to_numpy()
    else:
        conf = np.array([anomaly_score(p, x) for p, x in zip(v["pred"], c)])
        correct = (v["label"] == "anomaly").to_numpy()
    return pd.DataFrame({"image_id": v["image_id"].to_numpy(), "repeat": v["repeat"].to_numpy(),
                         "conf": conf, "correct": correct}), None


def consistency_by_image(df: pd.DataFrame) -> pd.DataFrame:
    """一致率による確信度（CI-CB6）。画像ごとに、有効な回のうち NG と言った割合 p_ng を出す。

    vote_pred：p_ng > 0.5 なら NG、< 0.5 なら OK。ちょうど 0.5（同数）や、有効な回が無いときは None。
    vote_conf：多数決の判定の確信度 = max(p_ng, 1 − p_ng)。vote_pred が None なら None。
    """
    validate(df)
    rows = []
    for image_id, g in df.groupby("image_id", sort=True):
        labels = set(g["label"])
        if len(labels) != 1:
            raise ValueError(f"label differs between repeats for {image_id}")
        valid = [p for p in g["pred"] if not _is_missing(p)]
        n_ng = sum(p == "NG" for p in valid)
        p_ng = round(n_ng / len(valid), SCORE_DIGITS) if valid else None
        if p_ng is None or p_ng == 0.5:
            vote, vconf = None, None
        else:
            vote = "NG" if p_ng > 0.5 else "OK"
            vconf = max(p_ng, round(1 - p_ng, SCORE_DIGITS))
        rows.append({"image_id": image_id, "label": labels.pop(),
                     "category": g["category"].iloc[0] if "category" in g else None,
                     "n_runs": len(g), "n_valid": len(valid), "n_ng": n_ng, "p_ng": p_ng,
                     "vote_pred": vote, "vote_conf": vconf})
    return pd.DataFrame(rows)


def consistency_points(img: pd.DataFrame, view: str = "judgement") -> pd.DataFrame:
    """consistency_by_image の表を較正の点にする（CI-CB6）。

    view="judgement"：vote_pred が None の画像を除き、conf = vote_conf、correct = 多数決が正解か。
    view="anomaly"  ：p_ng が None の画像を除き、conf = p_ng、correct = 不良品か。
    """
    if view == "judgement":
        v = img[img["vote_pred"].notna()]
        return pd.DataFrame({"image_id": v["image_id"].to_numpy(), "conf": v["vote_conf"].astype(float).to_numpy(),
                             "correct": (v["vote_pred"] == v["label"].map(_truth)).to_numpy()})
    if view == "anomaly":
        v = img[img["p_ng"].notna()]
        return pd.DataFrame({"image_id": v["image_id"].to_numpy(), "conf": v["p_ng"].astype(float).to_numpy(),
                             "correct": (v["label"] == "anomaly").to_numpy()})
    raise ValueError(f"view must be one of {VIEWS}")


def gray_zone(scores: np.ndarray, is_anom: np.ndarray, lo: float, hi: float) -> dict[str, Any]:
    """不良らしさのスコアでグレーを決めて人に回す（CI-CB7）。

    スコア < lo は自動で OK、スコア ≥ hi は自動で NG、そのあいだ（lo ≤ スコア < hi）と、スコアが無い（NaN）画像は人へ。
    人が見た画像は正しく判定されるとみなす。見逃し = 自動で OK にした不良品。
    """
    if not (0 <= lo <= hi <= 1):
        raise ValueError("need 0 <= lo <= hi <= 1")
    s = np.asarray(scores, dtype=float)
    a = np.asarray(is_anom, dtype=bool)
    if len(s) != len(a):
        raise ValueError("scores and is_anom must have the same length")
    known = ~np.isnan(s)
    s_r = np.round(np.where(known, s, 0.0), SCORE_DIGITS)
    auto_ok = known & (s_r < round(lo, SCORE_DIGITS))
    auto_ng = known & (s_r >= round(hi, SCORE_DIGITS)) & ~auto_ok
    human = ~(auto_ok | auto_ng)
    n, n_anom, n_norm = len(s), int(a.sum()), int((~a).sum())
    n_miss = int((auto_ok & a).sum())
    n_fp = int((auto_ng & ~a).sum())
    return {"lo": lo, "hi": hi, "n": n, "n_human": int(human.sum()), "human_rate": _div(int(human.sum()), n),
            "n_human_no_score": int((~known).sum()),
            "n_auto_ok": int(auto_ok.sum()), "n_auto_ng": int(auto_ng.sum()),
            "n_miss": n_miss, "miss_rate": _div(n_miss, n_anom),
            "n_false_ng": n_fp, "false_positive_rate": _div(n_fp, n_norm)}
