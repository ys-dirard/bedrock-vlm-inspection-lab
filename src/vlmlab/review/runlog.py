"""DynamoDB の記録と正解ラベルから run_log.csv の行を作る（CI-HR12・HR13）。AWS は呼ばない。

正解（label）は状態機械に渡さない（判定に混ざらないよう、ローカルの一覧とだけ突き合わせる）。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..cost import Price, call_cost

# 経路ごとに入る状態の数（Step Functions の状態遷移の課金の見積りに使う。実測ではない）
STATES_BY_STATUS = {
    "auto_decided": 3,   # Judge → Route → RecordAuto
    "human_decided": 5,  # Judge → Route → AskHuman → ValidDecision → RecordHuman
    "timed_out": 4,      # Judge → Route → AskHuman → RecordTimeout
    "judge_failed": 3,   # Judge → RecordJudgeError → JudgeFailed
}
FINAL_STATUSES = tuple(STATES_BY_STATUS)

COLUMNS = [
    "review_id", "image_id", "category", "label", "status", "route", "route_reason",
    "vlm_judgement", "vlm_confidence", "anomaly_score", "parse_error", "final_judgement", "decided_by",
    "correct", "missed", "false_ng",
    "started_at", "requested_at", "decided_at", "human_wait_s", "total_s",
    "input_tokens", "output_tokens", "bedrock_cost_usd", "est_state_transitions", "judge_latency_s",
]


def _s(item: dict[str, Any], name: str) -> str | None:
    v = item.get(name)
    return None if v is None else v.get("S")


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _secs(a: str | None, b: str | None) -> float | None:
    ta, tb = _ts(a), _ts(b)
    return None if ta is None or tb is None else round((tb - ta).total_seconds(), 3)


def build_rows(items: list[dict[str, Any]], labels: dict[str, dict[str, str]], price: Price) -> list[dict[str, Any]]:
    """CI-HR12：1件1行。labels は image_id → {"category", "label"}。

    - 正解の無い image_id、終わっていない状態（pending・submitting・expired）があれば ValueError（数え漏れを黙って出さない）
    - correct は final_judgement があるときだけ（NG を陽性。label anomaly ⇔ NG）。missed は不良品を OK で確定、
      false_ng は良品を NG で確定。確定していなければ None
    """
    rows = []
    for it in items:
        image_id = _s(it, "image_id")
        if image_id not in labels:
            raise ValueError(f"no label for image_id {image_id!r}")
        status = _s(it, "status")
        if status not in FINAL_STATUSES:
            raise ValueError(f"review {_s(it, 'review_id')!r} is not finished (status={status!r})")
        vlm = json.loads(_s(it, "vlm_result") or "{}")
        lab = labels[image_id]
        final = _s(it, "final_judgement")
        if final is not None and final not in ("OK", "NG"):
            raise ValueError(f"unexpected final_judgement {final!r}")
        truth_ng = lab["label"] == "anomaly"
        tin, tout = vlm.get("input_tokens"), vlm.get("output_tokens")
        cost = call_cost(price, int(tin), int(tout)) if tin is not None and tout is not None else None
        rows.append({
            "review_id": _s(it, "review_id"), "image_id": image_id, "category": lab["category"], "label": lab["label"],
            "status": status, "route": vlm.get("route"), "route_reason": vlm.get("route_reason"),
            "vlm_judgement": vlm.get("judgement"), "vlm_confidence": vlm.get("confidence"),
            "anomaly_score": vlm.get("anomaly_score"), "parse_error": vlm.get("parse_error"),
            "final_judgement": final, "decided_by": _s(it, "decided_by"),
            "correct": None if final is None else (final == "NG") == truth_ng,
            "missed": None if final is None else (truth_ng and final == "OK"),
            "false_ng": None if final is None else ((not truth_ng) and final == "NG"),
            "started_at": _s(it, "started_at"), "requested_at": _s(it, "requested_at"),
            "decided_at": _s(it, "decided_at"),
            "human_wait_s": _secs(_s(it, "requested_at"), _s(it, "decided_at")) if status == "human_decided" else None,
            "total_s": _secs(_s(it, "started_at"), _s(it, "decided_at")),
            "input_tokens": tin, "output_tokens": tout, "bedrock_cost_usd": cost,
            "est_state_transitions": STATES_BY_STATUS[status], "judge_latency_s": vlm.get("latency_s"),
        })
    return sorted(rows, key=lambda r: (r["started_at"] or "", r["image_id"]))


def summarize(rows: list[dict[str, Any]], transition_usd: float) -> dict[str, Any]:
    """CI-HR13：経路ごとの件数、見逃し・誤検知、費用。0 で割らない（None にする）。"""
    n = len(rows)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    n_anom = sum(r["label"] == "anomaly" for r in rows)
    n_norm = n - n_anom
    missed = sum(bool(r["missed"]) for r in rows)
    false_ng = sum(bool(r["false_ng"]) for r in rows)
    costs = [r["bedrock_cost_usd"] for r in rows]
    bedrock = None if any(c is None for c in costs) else sum(costs)
    waits = sorted(r["human_wait_s"] for r in rows if r["human_wait_s"] is not None)
    transitions = sum(r["est_state_transitions"] for r in rows)
    return {
        "n": n, "by_status": by_status,
        "n_human": by_status.get("human_decided", 0) + by_status.get("timed_out", 0),
        "missed": missed, "miss_rate": None if n_anom == 0 else missed / n_anom,
        "false_ng": false_ng, "false_ng_rate": None if n_norm == 0 else false_ng / n_norm,
        "undecided": sum(r["final_judgement"] is None for r in rows),
        "human_wait_s_median": None if not waits else _median(waits),
        "bedrock_cost_usd": bedrock,
        "est_state_transitions": transitions,
        "est_sfn_cost_usd_without_free_tier": transitions * transition_usd,
    }


def _median(xs: list[float]) -> float:
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2
