"""MMAD の集計表を出す（CI-MS1〜MS6 の式を使う）。

出すもの（out_dir に）
- accuracy_by_subtask.csv: モデル × 数え方（strict / script）× 課題の正答率と平均（4データセットの単純平均）
- accuracy_by_source.csv: モデル × 数え方 × データセットごとの7課題の正答率と平均
- per_question.csv: 1行1問の正誤（信頼区間や、モデル間で正誤が分かれた問題を調べるため）
- run_summary.csv: 問題数、API エラー、記号が取れなかった割合、トークン数の平均、費用
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .score import LETTERS, accuracy_table, unparsed_rate


def load_calls(run_dir: str | Path) -> pd.DataFrame:
    """calls.jsonl を読み、record_id ごとに1行にする（CI-MR4）。

    成功した記録があれば最後の成功を、無ければ最後の失敗を使う。
    """
    path = Path(run_dir) / "calls.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no records in {path}")
    df = pd.DataFrame(rows)
    df["_ok"] = df["api_error"].isna()
    df["_order"] = range(len(df))
    df = df.sort_values(["record_id", "_ok", "_order"]).groupby("record_id", as_index=False).tail(1)
    return df.drop(columns=["_ok", "_order"]).sort_values("record_id").reset_index(drop=True)


def _sum_or_none(s: pd.Series) -> float | None:
    return None if s.isna().any() else float(s.sum())


def summarize_run(df: pd.DataFrame) -> dict[str, Any]:
    answered = df[df["api_error"].isna()]
    total_cost = _sum_or_none(df["cost_usd"])
    n = len(df)
    return {
        "run_name": df["run_name"].iloc[0], "model_id": df["model_id"].iloc[0], "mode": df["mode"].iloc[0],
        "n_shots": int(df["n_shots"].iloc[0]), "n_questions": n,
        "n_api_error": int(df["api_error"].notna().sum()),
        "unparsed_rate": unparsed_rate(answered),
        "mean_input_tokens": float(answered["input_tokens"].mean()) if len(answered) else None,
        "mean_output_tokens": float(answered["output_tokens"].mean()) if len(answered) else None,
        "cost_usd_total": total_cost,
        "cost_usd_per_1000": None if total_cost is None or n == 0 else total_cost / n * 1000,
    }


def build_report(run_dirs: list[str | Path], out_dir: str | Path, subset: set[str] | None = None) -> dict[str, Path]:
    """subset を渡すと、その record_id だけで集計する（CI-MR5。問題数の違うモデルを同じ問題で比べるため）。

    subset にある record_id が記録に無ければ ValueError（そろわないまま比べない）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subtask, source, per_q, summary = [], [], [], []
    for rd in run_dirs:
        df = load_calls(rd)
        if subset is not None:
            if missing := subset - set(df["record_id"]):
                raise ValueError(f"{rd} lacks {len(missing)} record_ids of the subset, e.g. {sorted(missing)[:3]}")
            df = df[df["record_id"].isin(subset)].reset_index(drop=True)
        info = summarize_run(df)
        summary.append(info)
        answered = df[df["api_error"].isna()]  # API エラーは集計から除き、件数を run_summary に出す
        for mode in ("strict", "script"):
            table = accuracy_table(answered, mode)
            avg = table.loc["Average"].to_dict()
            subtask.append({"run_name": info["run_name"], "model_id": info["model_id"], "count": mode, **avg})
            for ds, row in table.drop(index="Average").iterrows():
                source.append({"run_name": info["run_name"], "model_id": info["model_id"], "count": mode,
                               "dataset": ds, **row.to_dict()})
        q = df[["record_id", "qid", "dataset", "task", "is_normal", "answer", "pred", "api_error"]].copy()
        q.insert(0, "run_name", info["run_name"])
        q["correct"] = q["pred"].isin(LETTERS) & (q["pred"] == q["answer"])
        per_q.append(q)
    paths = {
        "accuracy_by_subtask": out / "accuracy_by_subtask.csv",
        "accuracy_by_source": out / "accuracy_by_source.csv",
        "per_question": out / "per_question.csv",
        "run_summary": out / "run_summary.csv",
    }
    pd.DataFrame(subtask).to_csv(paths["accuracy_by_subtask"], index=False, lineterminator="\n", float_format="%.2f")
    pd.DataFrame(source).to_csv(paths["accuracy_by_source"], index=False, lineterminator="\n", float_format="%.2f")
    pd.concat(per_q).to_csv(paths["per_question"], index=False, lineterminator="\n")
    pd.DataFrame(summary).to_csv(paths["run_summary"], index=False, lineterminator="\n")
    return paths
