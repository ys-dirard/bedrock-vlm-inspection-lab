"""12/18：評価を区切って流し直すときの決まり（docs/contracts.md の CI-AC-S5・S6）。"""

from __future__ import annotations

from typing import Any


def select_targets(records: list[dict[str, Any]], run_id: str | None) -> list[dict[str, Any]]:
    """CI-AC-S5。評価する記録：run_id で絞り（指定があれば）、Runtime のセッション ID があり、api_error が無いもの。
    同じ image_id が2つあれば止める（同じ run の中で重なるのは、流し直しの誤り）。"""
    rows = [r for r in records if (run_id is None or r.get("run_id") == run_id)
            and r.get("runtime_session_id") and r.get("api_error") is None]
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicated image_id in the selected records; pass --run-id")
    return rows


def evaluated(results: list[dict[str, Any]], evaluator: str) -> tuple[set[str], float, float]:
    """CI-AC-S6。この評価器で、エラーなく評価できたセッション、これまでの費用の合計と最大。"""
    mine = [r for r in results if r.get("evaluator") == evaluator]
    done = {r["session_id"] for r in mine if not r.get("error")}
    costs = [r.get("cost_usd") or 0.0 for r in mine]
    return done, sum(costs), max(costs, default=0.0)
