"""12/17：記憶の事例を Memory（vlmlabCaseMemory、戦略なし）に入れる。20件版は /cases/v20/、100件版は /cases/v100/。

- 本文は memory_records.csv の record_text（品目・12/13 の細かい文・確定した判定。CI-AC-M4）
- メタデータは category・judgement・image_id（category で絞って検索する）
- 記録は31日で消える（recordExpiryDurationDays。下限 31）
- その名前空間にすでに記録があれば、重ねて入れずに止める（--force で入れる）
- 入れたあと、一覧（ListMemoryRecords）の件数が入れた数になるまで待って確かめる

  uv run python scripts/ingest_memory.py [--records runs/agentcore/memory_features/memory_records.csv]
費用: 保存 120件 × 0.25〜0.75 USD／1,000件・月（31日で消える）。取り込みそのものの単価は料金ページに無い
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentlab.memcases import NAMESPACES, memory_records  # noqa: E402
from agentlab.memstore import batch_create_request  # noqa: E402

REGION = "ap-northeast-1"
NAME = "vlmlabCaseMemory"


def memory_id() -> str:
    c = boto3.client("bedrock-agentcore-control", region_name=REGION)
    for page in c.get_paginator("list_memories").paginate():
        for m in page.get("memories", []):
            if m["id"].startswith(f"{NAME}-") and m["status"] == "ACTIVE":
                return m["id"]
    sys.exit(f"{NAME} が ACTIVE でない")


def count(dp, mid: str, ns: str) -> int:
    n, token = 0, None
    while True:
        kw = {"memoryId": mid, "namespace": ns, "maxResults": 100}
        if token:
            kw["nextToken"] = token
        r = dp.list_memory_records(**kw)
        n += len(r.get("memoryRecordSummaries", []))
        token = r.get("nextToken")
        if not token:
            return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(ROOT / "runs" / "agentcore" / "memory_features" / "memory_records.csv"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cases = pd.read_csv(a.records)
    mid = memory_id()
    dp = boto3.client("bedrock-agentcore", region_name=REGION)
    for version, ns in NAMESPACES.items():
        recs = memory_records(cases, version)
        have = count(dp, mid, ns)
        if have and not a.force:
            print(f"{ns}: already {have} records, skip (--force to add)")
            continue
        r = dp.batch_create_memory_records(**batch_create_request(mid, ns, recs, datetime.now(timezone.utc)))
        ok, ng = len(r.get("successfulRecords", [])), r.get("failedRecords", [])
        print(f"{ns}: created {ok}, failed {len(ng)}")
        if ng:
            sys.exit(f"failed records: {ng[:3]}")
        for _ in range(12):
            n = count(dp, mid, ns)
            if n >= have + len(recs):
                break
            time.sleep(5)
        print(f"{ns}: listed {n} (expected {have + len(recs)})")


if __name__ == "__main__":
    main()
