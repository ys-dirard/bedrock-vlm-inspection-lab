"""2段構成：手元の AnomalyDINO の異常度で絞り、上位だけ Nova 2 Lite に回す（12/10 の記事）。docs/contracts.md の CI-F1〜F8。

API は呼ばない。AnomalyDINO の記録（runs/<ad>/scores.jsonl）と、11/11 の条件 B（runs/n1_B）の Nova の記録を組み合わせる。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .cascade import summarize_nova
from .run import load_jsonl

TARGET_MISS_RATES = (0.02, 0.05)
TOP_PERCENTS = (10, 20, 30, 50)
SCOPES = ("category", "global")      # 主は品目ごと（F-D4）
RESTS = ("OK", "AD_miss2", "AD_miss5")

_miss = M._is_missing


# ---------------------------------------------------------------- 読み込みと突き合わせ（CI-F1、F2）

def load_scores(run_dir: str | Path) -> pd.DataFrame:
    rows = load_jsonl(Path(run_dir) / "scores.jsonl")
    if not rows:
        raise ValueError(f"no scores in {run_dir}")
    return pd.DataFrame(rows)


def join(ad: pd.DataFrame, nova_calls: pd.DataFrame) -> pd.DataFrame:
    """CI-F1・F2：1枚1行の表（score、nova1_pred、nova1_cost、novamaj_pred、novamaj_cost）。"""
    counts = ad.groupby("image_id").size()
    if (counts != 1).any():
        raise ValueError(f"AnomalyDINO must have one row per image, e.g. {counts[counts != 1].index[:5].tolist()}")
    scores = ad["score"].astype(float)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite AnomalyDINO score")
    a = ad.set_index("image_id")[["category", "label", "score"]].sort_index()

    maj = summarize_nova(nova_calls)  # repeat 1〜3 がちょうど1回ずつでなければ止まる（CI-K1）
    r1 = nova_calls[nova_calls["repeat"] == 1].set_index("image_id").sort_index()
    if set(a.index) != set(maj.index):
        only_a = sorted(set(a.index) - set(maj.index))[:5]
        only_n = sorted(set(maj.index) - set(a.index))[:5]
        raise ValueError(f"image sets differ: only AnomalyDINO {only_a}, only Nova {only_n}")
    for col in ("label", "category"):
        diff = a.index[a[col] != maj.loc[a.index, col]]
        if len(diff):
            raise ValueError(f"{col} differs for {diff[:5].tolist()}")
    out = a.copy()
    # object 型にする（pandas が文字列型を推定すると None が NaN になる）
    out["nova1_pred"] = pd.Series([None if _miss(p) else p for p in r1.loc[a.index, "pred"]], index=a.index, dtype=object)
    out["nova1_cost"] = r1.loc[a.index, "cost_usd"].astype(float)
    out["novamaj_pred"] = pd.Series([None if _miss(p) else p for p in maj.loc[a.index, "nova_pred"]], index=a.index,
                                    dtype=object)
    out["novamaj_cost"] = maj.loc[a.index, "nova_cost"].astype(float)
    return out


# ---------------------------------------------------------------- しきい値と回す画像（CI-F3、F4）

def threshold_at_miss(scores: np.ndarray, is_anom: np.ndarray, target: float) -> float:
    """CI-F3：異常度 < t の不良品の割合が target 以下になる最大の t（候補は不良品の異常度）。"""
    s_anom = np.asarray(scores, dtype=float)[np.asarray(is_anom, dtype=bool)]
    if len(s_anom) == 0:
        raise ValueError("no anomaly samples")
    best = None
    for t in sorted(set(s_anom.tolist())):
        if float((s_anom < t).mean()) <= target:
            best = t
    assert best is not None  # 最小の候補では見逃し 0
    return best


def thresholds(df: pd.DataFrame, scope: str, target: float) -> pd.Series:
    """画像ごとに当てるしきい値（品目ごと、または全体で1つ）。"""
    if scope == "global":
        t = threshold_at_miss(df["score"].to_numpy(), (df["label"] == "anomaly").to_numpy(), target)
        return pd.Series(t, index=df.index)
    if scope != "category":
        raise ValueError(scope)
    per = {c: threshold_at_miss(g["score"].to_numpy(), (g["label"] == "anomaly").to_numpy(), target)
           for c, g in df.groupby("category")}
    return df["category"].map(per)


def top_percent(df: pd.DataFrame, scope: str, pct: float) -> pd.Series:
    """CI-F4：範囲ごとに k = ceil(pct × n / 100) 枚。異常度の降順、同じ値は image_id の昇順。"""
    if not 0 <= pct <= 100:
        raise ValueError(pct)
    if scope not in SCOPES:
        raise ValueError(scope)
    send = pd.Series(False, index=df.index)
    groups = [("all", df)] if scope == "global" else list(df.groupby("category"))
    for _, g in groups:
        k = math.ceil(round(pct * len(g) / 100, 9))
        order = g.assign(_id=g.index).sort_values(["score", "_id"], ascending=[False, True])
        send.loc[order.index[:k]] = True
    return send


# ---------------------------------------------------------------- 最終の判定と指標（CI-F5、F6）

def final_pred(df: pd.DataFrame, send: pd.Series, rest: pd.Series, nova_col: str) -> pd.Series:
    """CI-F5：回して Nova が有効なら Nova、それ以外は rest（回さない画像の判定）。"""
    nova = df[nova_col]
    use_nova = send & nova.map(lambda p: not _miss(p))
    return pd.Series(np.where(use_nova, nova, rest), index=df.index)


def evaluate(df: pd.DataFrame, send: pd.Series, rest: pd.Series) -> dict[str, Any]:
    """CI-F6：主（repeat=1）と参考（多数決）の指標・回した割合・1,000枚あたりの費用。"""
    out: dict[str, Any] = {"n_images": len(df), "n_sent": int(send.sum()), "send_rate": float(send.mean())}
    for prefix, pred_col, cost_col in (("", "nova1_pred", "nova1_cost"), ("maj_", "novamaj_pred", "novamaj_cost")):
        pred = final_pred(df, send, rest, pred_col)
        m = M.binary_metrics(pd.DataFrame({"label": df["label"], "pred": pred}))
        keys = ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "false_positive_rate", "miss_rate") if not prefix \
            else ("f1", "false_positive_rate", "miss_rate")
        out.update({f"{prefix}{k}": m[k] for k in keys})
        costs = df.loc[send, cost_col]
        out[f"{prefix}cost_per_1000_usd"] = None if costs.map(_miss).any() else float(costs.sum()) / len(df) * 1000
    return out


def conditions(df: pd.DataFrame) -> list[dict[str, Any]]:
    """表1の全条件。主は scope=category・Nova repeat=1。"""
    none = pd.Series(False, index=df.index)
    allsend = pd.Series(True, index=df.index)
    ok = pd.Series("OK", index=df.index)
    rows = [{"condition": "Nova だけ", "routing": "all", "scope": "-", "x_pct": None, "rest": "-",
             "threshold_from_labels": False, **evaluate(df, allsend, ok)}]
    thr = {(scope, m): thresholds(df, scope, m) for scope in SCOPES for m in TARGET_MISS_RATES}
    ad_pred = {k: pd.Series(np.where(df["score"] >= t, "NG", "OK"), index=df.index) for k, t in thr.items()}
    for scope in SCOPES:
        for m in TARGET_MISS_RATES:
            tag = f"miss{round(m * 100)}"
            rows.append({"condition": f"AnomalyDINO だけ（見逃し{round(m * 100)}%のしきい値）", "routing": "none",
                         "scope": scope, "x_pct": None, "rest": f"AD_{tag}", "threshold_from_labels": True,
                         **evaluate(df, none, ad_pred[(scope, m)])})
        for m in TARGET_MISS_RATES:
            tag = f"miss{round(m * 100)}"
            send = df["score"] >= thr[(scope, m)]
            rows.append({"condition": f"2段：見逃し{round(m * 100)}%のしきい値以上を回す", "routing": "threshold",
                         "scope": scope, "x_pct": None, "rest": "OK", "threshold_from_labels": True,
                         **evaluate(df, send, ok)})
        for pct in TOP_PERCENTS:
            send = top_percent(df, scope, pct)
            for rest in RESTS:
                if rest == "OK":
                    rest_s, from_labels = ok, False
                else:
                    m = 0.02 if rest == "AD_miss2" else 0.05
                    rest_s, from_labels = ad_pred[(scope, m)], True
                rows.append({"condition": f"2段：上位{pct}%を回す", "routing": "top", "scope": scope, "x_pct": pct,
                             "rest": rest, "threshold_from_labels": from_labels, **evaluate(df, send, rest_s)})
    return rows


# ---------------------------------------------------------------- 異常度の要約と時間（CI-F7、F8）

def score_summary(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rows = []
    for cat, g in [*df.groupby("category"), ("全体", df)]:
        y = (g["label"] == "anomaly").astype(int)
        row = {"category": cat, "auroc": float(roc_auc_score(y, g["score"])) if y.nunique() == 2 else None}
        for lab in ("normal", "anomaly"):
            s = g.loc[g["label"] == lab, "score"]
            row.update({f"{lab}_n": len(s), f"{lab}_min": s.min(), f"{lab}_median": s.median(), f"{lab}_max": s.max()})
        rows.append(row)
    return pd.DataFrame(rows)


def time_summary(ad: pd.DataFrame, meta: list[dict[str, Any]]) -> dict[str, Any]:
    t = ad["time_s"].astype(float).to_numpy()
    out = {"n": len(t), "time_p50_s": float(np.percentile(t, 50)), "time_p95_s": float(np.percentile(t, 95)),
           "time_mean_s": float(t.mean()), "time_total_s": float(t.sum())}
    if meta:
        out["model_load_s"] = [m.get("model_load_s") for m in meta]
        out["bank_time_s"] = {c: v["time_s"] for m in meta for c, v in m.get("bank", {}).items()}
        out["torch_threads"] = meta[-1].get("torch_threads")
        out["peak_memory_mb"] = max((m.get("memory") or {}).get("peak_mb") or 0 for m in meta) or None
        out["started"] = meta[0].get("started")
        out["finished"] = meta[-1].get("finished")
    return out


def run_prefilter(ad: pd.DataFrame, nova_calls: pd.DataFrame) -> dict[str, Any]:
    df = join(ad, nova_calls)
    return {"table": df, "rows": conditions(df), "scores": score_summary(df)}
