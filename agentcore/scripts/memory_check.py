"""12/17：戦略を持たない Memory に、自分で入れた記録を意味で検索できるかを少数で確かめる（記事の sources.md の U1）。

作るもの（12/17 の results/resources_plan.md の段階1。2026-11-21 書き手が承認）:
- AgentCore Memory vlmlabCaseMemory（戦略なし、実行ロールなし、eventExpiryDuration 7、indexedKeys: category・judgement）

確かめる記録は、確認用の短い文5件（データの事実ではない）を名前空間 /cases/check/ に入れる。31日で消える設定にする
（recordExpiryDurationDays の下限は 31。2026-11-21 に 7 で ParamValidationError になったため直した）。
費用の見込み: 保存 5件 × 0.00075 USD／件・月 未満、取り出し 3回 × 0.0005 USD = 0.0015 USD

  uv run python scripts/memory_check.py create     # Memory を作り、ACTIVE まで待つ
  uv run python scripts/memory_check.py check      # 記録を5件入れ、3回検索する。結果を runs/agentcore/memory_check.json に書く
  uv run python scripts/memory_check.py delete
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentlab.memstore import batch_create_request, retrieve_request  # noqa: E402

REGION = "ap-northeast-1"
NAME = "vlmlabCaseMemory"
NS = "/cases/check/"
TAGS = {"project": "advent-2026", "article": "2026-12-17"}
OUT = Path(__file__).resolve().parents[1] / "runs" / "agentcore" / "memory_check.json"

CHECK_RECORDS = [  # 確認用の文（12/13 の特徴の文の形を真似たもの。実データの記録ではない）
    ("c1", "pcb1", "NG", "品目 pcb1。異常の領域が1つ、位置は右上、面積は小さい、縦横比は細長い。確定した判定は NG。"),
    ("c2", "pcb1", "OK", "品目 pcb1。異常の領域は無い。確定した判定は OK。"),
    ("c3", "capsules", "NG", "品目 capsules。異常の領域が2つ、位置は中央、面積は中くらい。確定した判定は NG。"),
    ("c4", "capsules", "OK", "品目 capsules。異常の領域が1つ、位置は左下、面積はごく小さい。確定した判定は OK。"),
    ("c5", "pcb1", "NG", "品目 pcb1。異常の領域が1つ、位置は中央、面積は大きい。確定した判定は NG。"),
]
QUERIES = [
    ("pcb1 の右上に小さな細長い領域", None),
    ("capsules の中央に領域が2つ", None),
    ("pcb1 の右上に小さな細長い領域", "pcb1"),  # メタデータで品目を絞る
]


def ctl():
    return boto3.client("bedrock-agentcore-control", region_name=REGION)


def find(c) -> dict | None:
    for page in c.get_paginator("list_memories").paginate():
        for m in page.get("memories", []):
            if m["id"].startswith(f"{NAME}-"):
                return m
    return None


def create() -> None:
    c = ctl()
    if find(c):
        print("memory exists")
    else:
        r = c.create_memory(name=NAME, eventExpiryDuration=7, description="12/17 records search check (no strategy)",
                            indexedKeys=[{"key": "category", "type": "STRING"}, {"key": "judgement", "type": "STRING"}],
                            tags=TAGS)
        print("created", r["memory"]["id"], r["memory"]["status"])
    for _ in range(60):
        m = find(c)
        print(m["status"])
        if m["status"] in ("ACTIVE", "FAILED"):
            return
        time.sleep(10)


def check() -> None:
    m = find(ctl())
    if not m or m["status"] != "ACTIVE":
        sys.exit("memory is not ACTIVE")
    dp = boto3.client("bedrock-agentcore", region_name=REGION)
    now = datetime.now(timezone.utc)
    req = batch_create_request(m["id"], NS, [{"id": rid, "text": text, "metadata": {"category": cat, "judgement": j}}
                                             for rid, cat, j, text in CHECK_RECORDS], now)
    created = dp.batch_create_memory_records(**req)
    result = {"memory_id": m["id"], "batch_create": {k: v for k, v in created.items() if k != "ResponseMetadata"},
              "retrievals": []}
    time.sleep(20)  # 索引に入るまで待つ
    listed = dp.list_memory_records(memoryId=m["id"], namespace=NS)
    result["listed"] = len(listed.get("memoryRecordSummaries", []))
    for q, cat in QUERIES:
        try:
            r = dp.retrieve_memory_records(**retrieve_request(m["id"], NS, q, 3, cat))
            hits = [{"text": h["content"].get("text"), "score": h.get("score"), "metadata": h.get("metadata")}
                    for h in r.get("memoryRecordSummaries", [])]
            result["retrievals"].append({"query": q, "category_filter": cat, "hits": hits})
        except Exception as e:  # noqa: BLE001  検索できないこと自体が確かめたい結果
            result["retrievals"].append({"query": q, "category_filter": cat, "error": f"{type(e).__name__}: {e}"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))


def delete() -> None:
    c = ctl()
    m = find(c)
    if m:
        c.delete_memory(memoryId=m["id"])
        print("deleted", m["id"])
    else:
        print("no memory")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["create", "check", "delete"])
    {"create": create, "check": check, "delete": delete}[ap.parse_args().action]()
