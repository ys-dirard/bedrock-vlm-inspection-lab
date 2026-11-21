"""記録から集計表を作る（CI-IM1〜IM7）。API は呼ばない。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .run import load_records
from .score import FIELDS, parse_invoice, score, values_for_agreement


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den  # 0/0 を 0 や 1 にしない


def latest_per_doc(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """doc_id ごとに最後の成功、成功が無ければ最後の失敗を使う（CI-IM1）。"""
    by: dict[str, dict[str, Any]] = {}
    for r in records:
        prev = by.get(r["doc_id"])
        if prev is None or not r.get("api_error") or prev.get("api_error"):
            by[r["doc_id"]] = r
    return list(by.values())


def scored_frame(records: list[dict[str, Any]], truths: dict[str, dict[str, Any]], allow_fence: bool) -> pd.DataFrame:
    """1件1行の採点表（CI-IM2）。allow_fence=False なら ```json の囲みも解析失敗として扱う。"""
    rows = []
    for r in latest_per_doc(records):
        base = {k: r.get(k) for k in ("run_name", "condition", "model_id", "repeat", "doc_id", "invoice_id", "variant",
                                      "input_tokens", "output_tokens", "latency_s", "cost_usd", "stop_reason", "seq")}
        base["api_error"] = bool(r.get("api_error"))
        if r.get("api_error"):
            rows.append(base)
            continue
        p = parse_invoice(r.get("raw_text"))
        parsed = p["parsed"] if (p["how"] == "strict" or (allow_fence and p["how"] == "fenced")) else None
        base.update({"parse_how": p["how"], "parse_ok": parsed is not None,
                     "schema_ok": parsed is not None and not p["schema_errors"],
                     "agree_key": json.dumps(values_for_agreement(parsed), ensure_ascii=False)
                     if parsed is not None else None})
        base.update(score(parsed, truths[r["invoice_id"]]))
        rows.append(base)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """1つの条件（と画像の種類）の指標（CI-IM3〜IM6）。API エラーの件は分母から外し、件数を出す。"""
    ok = df[~df["api_error"].astype(bool)].copy() if len(df) else df
    for c in ["parse_ok", "schema_ok", "all_fields", "checkable_all", "flagged", *[f"ok_{f}" for f in FIELDS]]:
        if c in ok:
            ok[c] = ok[c].astype(bool)  # API エラーの行が混ざると object 型になり、~ が整数の反転になるため
    n = len(ok)
    out: dict[str, Any] = {"n_calls": len(df), "n_api_error": int(df["api_error"].sum()) if len(df) else 0, "n": n}
    if n == 0:
        return out
    out["parse_fail_rate"] = _rate(int((~ok["parse_ok"]).sum()), n)
    out["schema_fail_rate"] = _rate(int((~ok["schema_ok"]).sum()), n)
    out["fenced_rate"] = _rate(int((ok["parse_how"] == "fenced").sum()), n)
    for f in FIELDS:
        out[f"acc_{f}"] = float(ok[f"ok_{f}"].mean())
    out["acc_all_fields"] = float(ok["all_fields"].mean())
    out["line_item_match_rate"] = _rate(int(ok["n_items_matched"].sum()), int(ok["n_items_truth"].sum()))
    # 整合チェック（CI-IM4）：誤りあり＝all_fields が False。検出率と、誤りでないのに引っかかった割合
    wrong, right = ok[~ok["all_fields"]], ok[ok["all_fields"]]
    out["n_wrong"] = len(wrong)
    out["detect_rate"] = _rate(int(wrong["flagged"].sum()), len(wrong))
    out["false_alarm_rate"] = _rate(int(right["flagged"].sum()), len(right))
    wrong_c = ok[~ok["checkable_all"]]
    out["n_wrong_checkable"] = len(wrong_c)
    out["detect_rate_checkable"] = _rate(int(wrong_c["flagged"].sum()), len(wrong_c))
    out["false_alarm_rate_checkable"] = _rate(int(ok[ok["checkable_all"]]["flagged"].sum()), int(ok["checkable_all"].sum()))
    # トークン・レイテンシ・費用（CI-IM5）
    out["input_tokens_mean"] = float(ok["input_tokens"].mean())
    out["output_tokens_mean"] = float(ok["output_tokens"].mean())
    lat = ok["latency_s"].dropna().to_numpy(dtype=float)
    out["latency_p50_s"] = float(np.percentile(lat, 50)) if len(lat) else None
    out["latency_p95_s"] = float(np.percentile(lat, 95)) if len(lat) else None
    first = ok.sort_values("seq").iloc[0]
    out["first_call_latency_s"] = float(first["latency_s"]) if first["latency_s"] is not None else None
    costs = df["cost_usd"]
    out["cost_total_usd"] = None if costs.isna().any() else float(costs.sum())
    out["cost_per_1000_usd"] = None if costs.isna().any() else float(costs.mean() * 1000)
    out["stop_max_tokens"] = int((ok["stop_reason"] == "max_tokens").sum())
    return out


def agreement(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, Any]:
    """一致率：両方の回で読めた doc のうち、10項目の値がすべて同じだった割合。どちらかが読めなければ一致としない（CI-IM6）。"""
    a = df1[~df1["api_error"]].set_index("doc_id")["agree_key"]
    b = df2[~df2["api_error"]].set_index("doc_id")["agree_key"]
    common = a.index.intersection(b.index)
    same = sum(1 for d in common if a[d] is not None and a[d] == b[d])
    return {"n_common": len(common), "n_same": same, "agreement": _rate(same, len(common))}


def build_report(run_dirs: list[Path], truths: dict[str, dict[str, Any]], out_dir: Path, allow_fence: bool = True) -> pd.DataFrame:
    """条件×画像の種類ごとの表（CSV と Markdown）と、1件1行の採点表を出す（CI-IM7）。"""
    frames = [scored_frame(load_records(d / "calls.jsonl"), truths, allow_fence) for d in run_dirs]
    allrows = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows.to_csv(out_dir / "scored_calls.csv", index=False, encoding="utf-8", lineterminator="\n")
    summ = []
    for (run_name, cond), g in allrows.groupby(["run_name", "condition"], sort=False):
        for variant, gv in [("all", g), *g.groupby("variant")]:
            summ.append({"run_name": run_name, "condition": cond, "variant": variant, **summarize(gv)})
    table = pd.DataFrame(summ)
    table.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8", lineterminator="\n")
    (out_dir / "summary.md").write_text(to_markdown(table), encoding="utf-8", newline="\n")
    return table


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def to_markdown(table: pd.DataFrame) -> str:
    """指標を行、条件×画像の種類を列にした Markdown の表。"""
    heads = [f"{r.condition} / {r.variant}" for r in table.itertuples()]
    metrics = [c for c in table.columns if c not in ("run_name", "condition", "variant")]
    lines = ["| 指標 | " + " | ".join(heads) + " |", "| --- |" + " --- |" * len(heads)]
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(_fmt(v) for v in table[m].tolist()) + " |")
    return "\n".join(lines) + "\n"
