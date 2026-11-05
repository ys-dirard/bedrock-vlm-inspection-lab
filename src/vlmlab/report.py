"""集計表（CSV と Markdown）を出す。

記録（calls.jsonl）の raw_text から読み直して集計するので、読み取りの規則を変えても API を呼び直さなくてよい。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import metrics as M
from .parse import ParseSpec, parse_response
from .run import load_jsonl

TARGET_MISS_RATES = (0.02, 0.05)


def load_run(run_dir: str | Path, parse_override: dict[str, Any] | None = None) -> pd.DataFrame:
    """1つの実験の記録を読み、(image_id, repeat) ごとに最後の成功した呼び出しだけを残す。"""
    run_dir = Path(run_dir)
    records = load_jsonl(run_dir / "calls.jsonl")
    if not records:
        raise ValueError(f"no records in {run_dir}")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    spec = ParseSpec.from_dict(parse_override if parse_override is not None else cfg.get("parse"))
    df = pd.DataFrame(records)
    ok = df[df["api_error"].isna()].drop_duplicates(["image_id", "repeat"], keep="last").copy()
    parsed = [parse_response(t, spec) for t in ok["raw_text"]]
    ok["pred"] = [p.pred for p in parsed]
    ok["confidence"] = [p.confidence for p in parsed]
    ok["defect"] = [p.defect for p in parsed]
    ok["parse_error"] = [p.parse_error for p in parsed]
    ok.attrs["n_api_error"] = int(df["api_error"].notna().sum())
    ok.attrs["cost_all_calls_usd"] = _sum_or_none(df["cost_usd"])
    return ok.reset_index(drop=True)


def _sum_or_none(s: pd.Series) -> float | None:
    return None if s.isna().any() else float(s.sum())


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    pr = M.per_repeat(df)
    out = {
        **M.aggregate_repeats(pr),
        **M.agreement_rate(df),
        **M.latency_stats(df),
        **M.cost_stats(df),
        **M.token_stats(df),
    }
    return out


def business(df: pd.DataFrame) -> list[dict[str, Any]]:
    """回ごとに業務指標を出し、平均・最小・最大にまとめる（D1、D2）。"""
    rows = []
    for target in TARGET_MISS_RATES:
        per = [M.pass_rate_at_miss(g, target) for _, g in df.groupby("repeat", sort=True)]
        vals = [p["pass_rate"] for p in per if p["pass_rate"] is not None]
        rows.append({
            "target_miss_rate": target,
            "pass_rate_mean": sum(vals) / len(vals) if vals else None,
            "pass_rate_min": min(vals) if vals else None,
            "pass_rate_max": max(vals) if vals else None,
            "n_repeats_used": len(vals),
            "thresholds": ";".join("" if p["threshold"] is None else f"{p['threshold']:.4g}" for p in per),
            "reason": "; ".join(sorted({p["reason"] for p in per if p["reason"]})) or None,
        })
    return rows


def misclassified(df: pd.DataFrame) -> pd.DataFrame:
    truth = df["label"].map({"normal": "OK", "anomaly": "NG"})
    wrong = df[(df["pred"] != truth) | df["pred"].isna()]
    cols = ["run_name", "repeat", "image_id", "category", "label", "pred", "confidence", "defect",
            "parse_error", "path"]
    return wrong[cols].sort_values(["image_id", "repeat"])


def build_report(run_dirs: list[str | Path], out_dir: str | Path, tag: str = "summary") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, cat_rows, biz_rows, wrong = [], [], [], []
    for rd in run_dirs:
        df = load_run(rd)
        name = df["run_name"].iloc[0]
        summary_rows.append({"run_name": name, "n_api_error": df.attrs["n_api_error"],
                             "cost_all_calls_usd": df.attrs["cost_all_calls_usd"], **summarize(df)})
        for cat, g in df.groupby("category", sort=True):
            cat_rows.append({"run_name": name, "category": cat, **M.aggregate_repeats(M.per_repeat(g))})
        biz_rows += [{"run_name": name, **r} for r in business(df)]
        wrong.append(misclassified(df))

    paths = {
        "summary": out / f"{tag}.csv",
        "summary_md": out / f"{tag}.md",
        "by_category": out / f"{tag}_by_category.csv",
        "business_metric": out / f"{tag}_business_metric.csv",
        "misclassified": out / f"{tag}_misclassified.csv",
    }
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths["summary"], index=False, lineterminator="\n")
    pd.DataFrame(cat_rows).to_csv(paths["by_category"], index=False, lineterminator="\n")
    pd.DataFrame(biz_rows).to_csv(paths["business_metric"], index=False, lineterminator="\n")
    pd.concat(wrong).to_csv(paths["misclassified"], index=False, lineterminator="\n")
    paths["summary_md"].write_text(to_markdown(summary), encoding="utf-8")
    return paths


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_markdown(summary: pd.DataFrame) -> str:
    cols = [
        ("run_name", "条件", None),
        ("n_valid_sum", "有効件数", None),
        ("parse_fail_rate_mean", "解析失敗率", 3),
        ("precision_mean", "適合率", 3),
        ("recall_mean", "再現率", 3),
        ("f1_mean", "F1", 3),
        ("f1_min", "F1 最小", 3),
        ("f1_max", "F1 最大", 3),
        ("false_positive_rate_mean", "誤検知率", 3),
        ("miss_rate_mean", "見逃し率", 3),
        ("agreement_rate", "一致率", 3),
        ("latency_p50_s", "p50 (秒)", 2),
        ("latency_p95_s", "p95 (秒)", 2),
        ("input_tokens_mean", "入力トークン", 0),
        ("output_tokens_mean", "出力トークン", 0),
        ("cost_per_1000_usd", "1,000枚あたり (USD)", 3),
    ]
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(_fmt(r.get(k), d) if d is not None else _fmt(r.get(k)) for k, _, d in cols) + " |")
    note = "\n指標は回ごとに計算した平均（F1 は最小・最大も）。解析失敗は指標から除き、有効件数と解析失敗率を併記。\n"
    return "\n".join(lines) + "\n" + note
