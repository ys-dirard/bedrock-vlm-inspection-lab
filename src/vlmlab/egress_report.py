"""12/13 の記事の表1：画像を出さない構成（文だけ → Nova Micro）、AnomalyDINO だけ、VLM 直接を並べる。API は呼ばない。

docs/contracts.md の CI-E7、E8。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .egress import Calibration
from .prefilter import threshold_at_miss
from .report import TARGET_MISS_RATES, load_run


def _check_images(df: pd.DataFrame, features: pd.DataFrame, name: str) -> None:
    """CI-E7：比べる記録の画像の集合・正解・品目が features と同じで、1枚1行であること。"""
    if df["image_id"].duplicated().any():
        raise ValueError(f"{name}: more than one row per image")
    a = df.set_index("image_id")[["category", "label"]].sort_index()
    b = features.set_index("image_id")[["category", "label"]].sort_index()
    if not a.index.equals(b.index) or not a.equals(b):
        raise ValueError(f"{name}: images/labels/categories differ from features")


def model_row(name: str, calls: pd.DataFrame, features: pd.DataFrame, egress: str) -> dict[str, Any]:
    """CI-E8：モデルの記録（repeat=1 だけ）の指標。解析失敗は指標から除き、件数を書く。費用は全件分。"""
    df = calls[calls["repeat"] == 1].copy()
    _check_images(df, features, name)
    m = M.binary_metrics(df)
    row: dict[str, Any] = {"condition": name, **{k: m[k] for k in (
        "n_valid", "n_parse_fail", "tp", "fp", "tn", "fn", "precision", "recall", "f1",
        "false_positive_rate", "miss_rate")}}
    for t in TARGET_MISS_RATES:
        p = M.pass_rate_at_miss(df, t)
        row[f"pass_rate_at_miss{int(t * 100)}"] = p["pass_rate"]
        row[f"pass_rate_at_miss{int(t * 100)}_reason"] = p["reason"]
    row.update(M.cost_stats(df))
    row.update(M.token_stats(df))
    row.update(M.latency_stats(df))
    row["egress"] = egress
    return row


def anomalydino_rows(features: pd.DataFrame, cals: dict[str, Calibration]) -> list[dict[str, Any]]:
    """CI-E8：AnomalyDINO だけ。判定は学習用の良品で決めた区切り（品目ごと）：
    「異常度 > 良品の最大」または「異常度 > 良品の90%点」を NG。業務指標は異常度そのもので並べる（CI-F3 と同じ規則。
    しきい値はテストの正解で決まるので楽観的）。費用は 0、外に出る情報はなし。"""
    rows = []
    scores = features["score"].to_numpy(dtype=float)
    pass_rates = {}
    for t in TARGET_MISS_RATES:
        per = []
        for cat, g in features.groupby("category"):
            thr = threshold_at_miss(g["score"].to_numpy(), (g["label"] == "anomaly").to_numpy(), t)
            per.append(g.assign(_ng=g["score"] >= thr))
        dd = pd.concat(per)
        normal = dd[dd["label"] == "normal"]
        pass_rates[t] = float((~normal["_ng"]).mean())
    for rule, attr in (("学習用の良品の最大", "score_max"), ("学習用の良品の90%点", "score_q90")):
        cut = features["category"].map({c: getattr(cal, attr) for c, cal in cals.items()}).to_numpy(dtype=float)
        pred = np.where(scores > cut, "NG", "OK")
        m = M.binary_metrics(pd.DataFrame({"label": features["label"], "pred": pred}))
        row: dict[str, Any] = {"condition": f"AnomalyDINO だけ（異常度が{rule}を超えたら NG）",
                               **{k: m[k] for k in ("n_valid", "n_parse_fail", "tp", "fp", "tn", "fn", "precision",
                                                    "recall", "f1", "false_positive_rate", "miss_rate")}}
        for t in TARGET_MISS_RATES:
            row[f"pass_rate_at_miss{int(t * 100)}"] = pass_rates[t]
            row[f"pass_rate_at_miss{int(t * 100)}_reason"] = "異常度で並べ、しきい値はテストの正解で品目ごとに決めた（楽観的）"
        row.update({"cost_total_usd": 0.0, "cost_per_call_usd": 0.0, "cost_per_1000_usd": 0.0,
                    "egress": "なし（手元だけ）"})
        rows.append(row)
    return rows


def build_egress_summary(features: pd.DataFrame, cals: dict[str, Calibration],
                         text_runs: dict[str, str | Path], direct_runs: dict[str, str | Path]) -> pd.DataFrame:
    rows = []
    for gran, rd in text_runs.items():
        calls = load_run(rd)
        col = f"text_{gran}"
        chars = features[col].str.len()
        rows.append(model_row(f"文だけ → Nova Micro（{gran}）", calls, features,
                              f"品目名と文（文は平均{chars.mean():.0f}文字、最大{chars.max()}文字）。画像は0枚"))
    rows += anomalydino_rows(features, cals)
    for name, rd in direct_runs.items():
        rows.append(model_row(name, load_run(rd), features, "製品の画像と良品の見本の画像"))
    return pd.DataFrame(rows)


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = [("condition", "条件", None), ("n_valid", "有効件数", None), ("f1", "F1", 3),
            ("false_positive_rate", "誤検知率", 3), ("miss_rate", "見逃し率", 3),
            ("pass_rate_at_miss2", "見逃し2%での良品の通過率", 3), ("pass_rate_at_miss5", "見逃し5%での良品の通過率", 3),
            ("input_tokens_mean", "入力トークン", 0), ("output_tokens_mean", "出力トークン", 0),
            ("cost_per_1000_usd", "1,000枚あたり (USD)", 4), ("egress", "社外に出る情報", None)]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r.get(k), d) if d is not None else _fmt(r.get(k)) for k, _, d in cols)
                     + " |")
    return "\n".join(lines) + "\n\n各条件とも1枚1回（VLM 直接は 11/11 の記録の repeat=1）。解析失敗は指標から除き、費用は全件分。\n"
