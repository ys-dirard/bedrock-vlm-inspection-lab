"""確信度の較正（12/12 の記事）。記録から ECE・信頼度図の表・グレーゾーン・業務指標を出す。API は呼ばない。

- 自己申告（self）：応答の confidence。繰り返しのある実験は回ごとに ECE を計算し（D2）、全回をまとめた ECE も出す
- 一致率（consistency）：同じ画像を複数回判定したときの、NG と言った割合（CI-CB6）。1枚1点
- 見方（view）：judgement = 「判定の確信度」と「判定が正解か」、anomaly = 「不良らしさ」と「不良品か」
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .report import load_run

TARGET_MISS_RATES = (0.02, 0.05)
# グレー（人に回す）の範囲 [lo, hi)。(0.5, 0.5) はグレー無し（0.5 で切るだけ）。0.4〜0.8 は plan.md の例
GRAY_ZONES = ((0.5, 0.5), (0.4, 0.8), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9))


def _ece_row(points: pd.DataFrame | None) -> dict[str, Any]:
    if points is None:
        return M.ece(np.array([]), np.array([]))
    return M.ece(points["conf"].to_numpy(), points["correct"].to_numpy())


def _bins(points: pd.DataFrame | None) -> pd.DataFrame:
    if points is None or len(points) == 0:
        return pd.DataFrame()
    return M.reliability_bins(points["conf"].to_numpy(), points["correct"].to_numpy())


def self_report(df: pd.DataFrame, source: str) -> tuple[list[dict], list[pd.DataFrame], list[dict], list[dict]]:
    """自己申告の確信度：ECE（回ごと・全回）、信頼度図の表（全回）、業務指標とグレーゾーン（回ごと）。"""
    summary, bins, biz, gray = [], [], [], []
    n_rep = df["repeat"].nunique()
    for view in M.VIEWS:
        pts, reason = M.self_report_points(df, view)
        per_rep = []
        if pts is not None:
            for rep, g in pts.groupby("repeat", sort=True):
                per_rep.append(_ece_row(g)["ece"])
        vals = [v for v in per_rep if v is not None]
        pooled = _ece_row(pts)
        summary.append({
            "source": source, "method": "self", "view": view, "n_repeats": n_rep,
            **{k: pooled[k] for k in ("ece", "n", "mean_conf", "accuracy", "overconfidence", "n_bins_used", "n_bins_few")},
            "ece_repeat_mean": float(np.mean(vals)) if vals else None,
            "ece_repeat_min": min(vals) if vals else None, "ece_repeat_max": max(vals) if vals else None,
            "reason": reason,
        })
        b = _bins(pts)
        if len(b):
            bins.append(b.assign(source=source, method="self", view=view))

    # 業務指標・グレーゾーン：回ごとに D1 のスコアで計算し、平均も出す
    for rep, g in df.groupby("repeat", sort=True):
        for t in TARGET_MISS_RATES:
            biz.append({"source": source, "method": "self", "repeat": rep, **M.pass_rate_at_miss(g, t)})
        scores = _d1_scores(g)
        is_anom = (g["label"] == "anomaly").to_numpy()
        for lo, hi in GRAY_ZONES:
            gray.append({"source": source, "method": "self", "repeat": rep, **M.gray_zone(scores, is_anom, lo, hi)})
    return summary, bins, biz, gray


def _d1_scores(g: pd.DataFrame) -> np.ndarray:
    """解析失敗・確信度なしの行は NaN（グレーゾーンでは人に回す）。"""
    out = []
    for p, c in zip(g["pred"], g["confidence"]):
        out.append(np.nan if M._is_missing(p) or M._is_missing(c) else M.anomaly_score(p, float(c)))
    return np.array(out, dtype=float)


def consistency(df: pd.DataFrame, source: str) -> tuple[list[dict], list[pd.DataFrame], list[dict], list[dict], pd.DataFrame]:
    """一致率による確信度：画像ごとに1点。"""
    img = M.consistency_by_image(df)
    n_rep = df["repeat"].nunique()
    if n_rep < 2:
        raise ValueError(f"{source}: consistency needs 2 or more repeats")
    summary, bins = [], []
    for view in M.VIEWS:
        pts = M.consistency_points(img, view)
        e = _ece_row(pts)
        summary.append({
            "source": source, "method": "consistency", "view": view, "n_repeats": n_rep,
            **{k: e[k] for k in ("ece", "n", "mean_conf", "accuracy", "overconfidence", "n_bins_used", "n_bins_few")},
            "ece_repeat_mean": None, "ece_repeat_min": None, "ece_repeat_max": None,
            "n_images": len(img), "n_images_excluded": len(img) - len(pts),
            "reason": None,
        })
        b = _bins(pts)
        if len(b):
            bins.append(b.assign(source=source, method="consistency", view=view))
    scores = img["p_ng"].astype(float).to_numpy()  # None は NaN
    is_anom = (img["label"] == "anomaly").to_numpy()
    known = ~np.isnan(scores)
    biz = []
    for t in TARGET_MISS_RATES:
        r = M.pass_rate_at_miss_scores(scores[known], is_anom[known], t)
        biz.append({"source": source, "method": "consistency", "repeat": "vote", **r,
                    "n_excluded_no_score": int((~known).sum())})
    gray = [{"source": source, "method": "consistency", "repeat": "vote", **M.gray_zone(scores, is_anom, lo, hi)}
            for lo, hi in GRAY_ZONES]
    return summary, bins, biz, gray, img.assign(source=source)


def _with_mean(rows: list[dict], keys: list[str], group: list[str]) -> pd.DataFrame:
    """回ごとの行に、回の平均の行（repeat = "mean"）を足す。"""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df[keys] = df[keys].apply(pd.to_numeric)
    per = df[df["repeat"] != "vote"]
    if per.empty:
        return df
    means = per.groupby(group, sort=False)[keys].mean().reset_index()
    counts = per.groupby(group, sort=False)["repeat"].nunique().reset_index(name="n_repeats_in_mean")
    means = means.merge(counts, on=group).assign(repeat="mean")
    return pd.concat([df, means], ignore_index=True)


def build_calibration(self_runs: list[str | Path], consistency_runs: list[str | Path], out_dir: str | Path,
                      tag: str = "calibration") -> dict[str, Path]:
    """self_runs の自己申告と、consistency_runs の一致率を集計して CSV に出す。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, bins, biz, gray, imgs = [], [], [], [], []
    loaded: dict[str, pd.DataFrame] = {}

    def get(rd: str | Path) -> pd.DataFrame:
        key = str(rd)
        if key not in loaded:
            loaded[key] = load_run(rd)
        return loaded[key]

    for rd in self_runs:
        df = get(rd)
        s, b, z, g = self_report(df, df["run_name"].iloc[0])
        summary += s; bins += b; biz += z; gray += g
    for rd in consistency_runs:
        df = get(rd)
        s, b, z, g, img = consistency(df, df["run_name"].iloc[0])
        summary += s; bins += b; biz += z; gray += g; imgs.append(img)

    paths = {
        "summary": out / f"{tag}_summary.csv",
        "bins": out / f"{tag}_reliability_bins.csv",
        "business_metric": out / f"{tag}_business_metric.csv",
        "gray_zone": out / f"{tag}_gray_zone.csv",
        "images": out / f"{tag}_consistency_images.csv",
    }
    pd.DataFrame(summary).to_csv(paths["summary"], index=False, lineterminator="\n")
    lead = ["source", "method", "view"]
    if bins:
        b = pd.concat(bins, ignore_index=True)
        b[lead + [c for c in b.columns if c not in lead]].to_csv(paths["bins"], index=False, lineterminator="\n")
    _with_mean(biz, ["pass_rate", "achieved_miss_rate"], ["source", "method", "target_miss_rate"]).to_csv(
        paths["business_metric"], index=False, lineterminator="\n")
    _with_mean(gray, ["n_human", "human_rate", "n_miss", "miss_rate", "n_false_ng", "false_positive_rate"],
               ["source", "method", "lo", "hi"]).to_csv(paths["gray_zone"], index=False, lineterminator="\n")
    if imgs:
        pd.concat(imgs, ignore_index=True).to_csv(paths["images"], index=False, lineterminator="\n")
    else:
        paths.pop("images")
    return paths
