"""答えの読み取りと、summary.py と同じ式の集計（CI-MS1〜MS6）。"""

from __future__ import annotations

import math
import re
from typing import Iterable

import pandas as pd

from .data import DATASETS, TASKS

# gpt4o.py の parse_answer と同じ正規表現。最後に見つかった記号を答えにする（generate_answer の gpt_answer[-1]）
_LETTER = re.compile(r"\b([A-E])\b")
LETTERS = ("A", "B", "C", "D", "E")
MODES = ("strict", "script")  # strict: 読めない答えは不正解 / script: summary.py と同じく分母から外す


def extract_answer(text: str | None) -> str | None:
    """応答の本文から答えの記号を取る。見つからなければ None（選択肢の文で合わせ直すことはしない）。"""
    if not text:
        return None
    found = _LETTER.findall(text)
    return found[-1] if found else None


def _acc(correct: Iterable[bool]) -> float | None:
    values = list(correct)
    if not values:
        return None
    return 100.0 * sum(values) / len(values)


def _mean(values: list[float | None]) -> float | None:
    if any(v is None for v in values) or not values:
        return None
    return sum(values) / len(values)  # type: ignore[arg-type]


def accuracy_table(df: pd.DataFrame, mode: str = "strict") -> pd.DataFrame:
    """データセット × 課題の正答率（%）と、その平均を返す。

    df の列: dataset, task, is_normal, answer, pred（記号か None）。API エラーで答えが無い行は呼ぶ側で除く。
    - mode="strict": pred が None の行は不正解として数える
    - mode="script": pred が A〜E 以外の行を分母から外す（summary.py と同じ）
    - 異常の判別は、データセットごとに良品と不良品の正答率の平均（summary.py と論文 Table 2 の注）
    - 各データセットの "Average" はその行の7課題の平均。最後の "Average" 行は、各列のデータセット間の平均
    - 分母が 0 のセルは None。None を含む平均は None（0 や除外で埋めない）
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    required = {"dataset", "task", "is_normal", "answer", "pred"}
    if missing := required - set(df.columns):
        raise ValueError(f"missing columns {sorted(missing)}")
    if bad := set(df["answer"]) - set(LETTERS):
        raise ValueError(f"answer must be A-E, got {sorted(bad)}")
    work = df.copy()
    valid = work["pred"].isin(LETTERS)
    if mode == "script":
        work = work[valid]
    work["correct"] = work["pred"] == work["answer"]

    datasets = [d for d in DATASETS if d in set(df["dataset"])]
    rows = {}
    for ds in datasets:
        sub = work[work["dataset"] == ds]
        row: dict[str, float | None] = {}
        for task in TASKS:
            cell = sub[sub["task"] == task]
            if task == "Anomaly Detection":
                normal = _acc(cell.loc[cell["is_normal"], "correct"])
                anomaly = _acc(cell.loc[~cell["is_normal"], "correct"])
                row[task] = None if normal is None or anomaly is None else (normal + anomaly) / 2
            else:
                row[task] = _acc(cell["correct"])
        row["Average"] = _mean([row[t] for t in TASKS])
        rows[ds] = row
    columns = list(TASKS) + ["Average"]
    rows["Average"] = {c: _mean([rows[ds][c] for ds in datasets]) for c in columns}
    return pd.DataFrame.from_dict(rows, orient="index")[columns]


def unparsed_rate(df: pd.DataFrame) -> float | None:
    """記号が取れなかった割合（API エラーの行は呼ぶ側で除く）。"""
    if len(df) == 0:
        return None
    return float((~df["pred"].isin(LETTERS)).mean())


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """正答数 k / 問題数 n の 95% 信頼区間（Wilson、割合で返す）。n=0 なら None。"""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)
