"""Memory の要求の形（CI-AC-M6・M7）。boto3 の検証（スタブ）にも通す。AWS は呼ばない。"""

from datetime import datetime, timezone

import boto3
import pytest
from botocore.stub import Stubber

from agentlab.memstore import batch_create_request, retrieve_request

NOW = datetime(2026, 11, 21, tzinfo=timezone.utc)
RECS = [{"id": "c1", "text": "品目: pcb1", "metadata": {"category": "pcb1", "judgement": "NG"}},
        {"id": "c2", "text": "品目: capsules", "metadata": {"category": "capsules", "judgement": "OK"}}]
MEM = "vlmlabCaseMemory-abcdefghij"


def client():
    return boto3.client("bedrock-agentcore", region_name="ap-northeast-1", aws_access_key_id="x",
                        aws_secret_access_key="x")


def test_batch_create_passes_boto3_validation():  # CI-AC-M6（2026-11-21 の ParamValidationError の再発防止）
    c = client()
    req = batch_create_request(MEM, "/cases/check/", RECS, NOW)
    assert req["recordExpirySpecification"] == {"recordExpiryDurationDays": 31}
    with Stubber(c) as st:
        st.add_response("batch_create_memory_records", {"successfulRecords": [], "failedRecords": []}, req)
        c.batch_create_memory_records(**req)
    no_expiry = batch_create_request(MEM, "/cases/check/", RECS, NOW, expiry_days=None)
    assert "recordExpirySpecification" not in no_expiry


def test_batch_create_rejects_bad_input():
    for kw in ({"expiry_days": 7}, {"records": []}, {"records": RECS[:1] * 2}):
        args = {"records": RECS, **kw}
        recs = args.pop("records")
        with pytest.raises(ValueError):
            batch_create_request(MEM, "/cases/check/", recs, NOW, **args)


def test_retrieve_request_passes_boto3_validation():  # CI-AC-M7
    c = client()
    for cat in (None, "pcb1"):
        req = retrieve_request(MEM, "/cases/check/", "pcb1 の右上", 3, cat)
        assert ("metadataFilters" in req["searchCriteria"]) == (cat is not None)
        with Stubber(c) as st:
            st.add_response("retrieve_memory_records", {"memoryRecordSummaries": []}, req)
            c.retrieve_memory_records(**req)
    with pytest.raises(ValueError):
        retrieve_request(MEM, "/cases/check/", " ", 3)
