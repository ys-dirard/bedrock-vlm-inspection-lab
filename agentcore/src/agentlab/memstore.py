"""12/17：Memory に記録を入れる・取り出す要求の形（docs/contracts.md の CI-AC-M6・M7）。AWS は呼ばない。

BatchCreateMemoryRecords の recordExpirySpecification は Union で、指定できるのは recordExpiryDurationDays だけ
（REMOVE の action は作るときには効かない）。指定しなければ記録は期限切れにならない。
https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_BatchCreateMemoryRecords.html （2026-11-21 確認）
日数の下限 31 は boto3（botocore 1.44.7 のモデル）の検証の値（2026-11-21 の memory_check で ParamValidationError）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

MIN_EXPIRY_DAYS = 31
MAX_BATCH = 100  # 1回の BatchCreateMemoryRecords の records の上限（API リファレンス）


def batch_create_request(memory_id: str, namespace: str, records: list[dict[str, Any]], now: datetime,
                         expiry_days: int | None = MIN_EXPIRY_DAYS) -> dict[str, Any]:
    """CI-AC-M6。records は {"id", "text", "metadata": {key: str}} の並び。"""
    if not records or len(records) > MAX_BATCH:
        raise ValueError(f"records must be 1..{MAX_BATCH}: {len(records)}")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicated record id")
    req: dict[str, Any] = {"memoryId": memory_id, "records": [
        {"requestIdentifier": r["id"], "namespaces": [namespace], "content": {"text": r["text"]}, "timestamp": now,
         "metadata": {k: {"stringValue": v} for k, v in r.get("metadata", {}).items()}}
        for r in records]}
    if expiry_days is not None:
        if expiry_days < MIN_EXPIRY_DAYS:
            raise ValueError(f"expiry_days must be >= {MIN_EXPIRY_DAYS} (or None for no expiry)")
        req["recordExpirySpecification"] = {"recordExpiryDurationDays": expiry_days}
    return req


def retrieve_request(memory_id: str, namespace: str, query: str, top_k: int,
                     category: str | None = None) -> dict[str, Any]:
    """CI-AC-M7。category を渡したときだけ、メタデータの category で絞る。"""
    if top_k < 1 or not query.strip():
        raise ValueError("top_k must be >= 1 and query must not be empty")
    crit: dict[str, Any] = {"searchQuery": query, "topK": top_k}
    if category is not None:
        crit["metadataFilters"] = [{"left": {"metadataKey": "category"}, "operator": "EQUALS_TO",
                                    "right": {"metadataValue": {"stringValue": category}}}]
    return {"memoryId": memory_id, "namespace": namespace, "searchCriteria": crit}
