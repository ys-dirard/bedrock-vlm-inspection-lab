"""分析エージェントのツールの SQL（読み取りだけ）。docs/contracts.md の CI-PL-Q1〜Q5。

値は SQL に埋め込まず、Athena の ExecutionParameters（? の位置引数）で渡す。
列の名前は決まった集合からだけ選ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COLUMNS = ("run_name", "model_id", "category", "label", "pred")  # CI-PL-Q1
MAX_LIMIT = 50


@dataclass(frozen=True)
class Query:
    sql: str
    params: tuple[str, ...]  # Athena の ExecutionParameters（文字列のリテラルとして渡す）


def _column(name: str) -> str:
    if name not in COLUMNS:
        raise ValueError(f"column must be one of {COLUMNS}: {name!r}")
    return name


def _where(table: str, filters: dict[str, str] | None) -> tuple[str, list[str]]:
    conds, params = [], []
    for k, v in sorted((filters or {}).items()):
        col = _column(k)
        if not isinstance(v, str):
            raise ValueError(f"filter value must be a string: {k}={v!r}")
        conds.append(f"{col} = ?")
        params.append(_quote(v))
    return ("WHERE " + " AND ".join(conds)) if conds else "", params


def _quote(v: str) -> str:
    # ExecutionParameters は SQL のリテラルとして解釈されるので、文字列は '...' で囲み、' を '' にする
    return "'" + v.replace("'", "''") + "'"


# 11月の CI-M2・M3 と同じ定義（NG を陽性、pred が null の行は除く、分母 0 は null）
_METRICS = """
  count(*) AS n_calls,
  count_if(pred IS NOT NULL) AS n_valid,
  count_if(pred IS NULL) AS n_parse_fail,
  count_if(label = 'anomaly' AND pred = 'NG') AS tp,
  count_if(label = 'normal' AND pred = 'NG') AS fp,
  count_if(label = 'normal' AND pred = 'OK') AS tn,
  count_if(label = 'anomaly' AND pred = 'OK') AS fn,
  CAST(count_if(label = 'anomaly' AND pred = 'NG') AS double)
    / nullif(count_if(label = 'anomaly' AND pred IS NOT NULL), 0) AS recall,
  CAST(count_if(label = 'normal' AND pred = 'NG') AS double)
    / nullif(count_if(label = 'normal' AND pred IS NOT NULL), 0) AS false_positive_rate,
  sum(cost_usd) AS cost_usd,
  approx_percentile(latency_s, 0.5) AS latency_p50_s"""


def count_by(table: str, group_by: list[str], filters: dict[str, str] | None = None) -> Query:
    """軸ごとの件数と、再現率・誤検知率・費用。"""
    if not group_by:
        raise ValueError("group_by must not be empty")
    cols = [_column(c) for c in group_by]
    where, params = _where(table, filters)
    keys = ", ".join(cols)
    sql = f"SELECT {keys},{_METRICS}\nFROM {table}\n{where}\nGROUP BY {keys}\nORDER BY {keys}"
    return Query(sql=_tidy(sql), params=tuple(params))


def list_images(table: str, filters: dict[str, str] | None = None, limit: int = 20) -> Query:
    """条件に合う画像の一覧（判定・確信度・欠陥・費用）。"""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer in 1..{MAX_LIMIT}: {limit!r}")  # CI-PL-Q3
    where, params = _where(table, filters)
    sql = (f"SELECT run_name, image_id, category, label, pred, confidence, defect, cost_usd\n"
           f"FROM {table}\n{where}\nORDER BY run_name, image_id\nLIMIT {limit}")
    return Query(sql=_tidy(sql), params=tuple(params))


def compare_models(table: str, run_names: list[str]) -> Query:
    """実験（run_name）ごとの指標を並べる。"""
    if not 2 <= len(run_names) <= 5 or not all(isinstance(r, str) for r in run_names):
        raise ValueError("run_names must be 2..5 strings")
    marks = ", ".join("?" for _ in run_names)
    sql = (f"SELECT run_name, model_id,{_METRICS}\nFROM {table}\nWHERE run_name IN ({marks})\n"
           f"GROUP BY run_name, model_id\nORDER BY run_name")
    return Query(sql=_tidy(sql), params=tuple(_quote(r) for r in run_names))


def _tidy(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if line.strip())


def build(tool: str, table: str, args: dict[str, Any]) -> Query:
    """ツールの名前と入力から SQL を作る（Lambda の入口から呼ぶ）。"""
    if tool == "count_by":
        return count_by(table, list(args.get("group_by") or []), args.get("filters"))
    if tool == "list_images":
        return list_images(table, args.get("filters"), args.get("limit", 20))
    if tool == "compare_models":
        return compare_models(table, list(args.get("run_names") or []))
    raise ValueError(f"unknown tool: {tool!r}")
