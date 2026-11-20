"""分析エージェントのツール（AgentCore Gateway の Lambda ターゲット）。読み取りだけ（CI-PL-Q1〜Q5）。

Gateway は、ツールの入力を event に、ツールの名前を context.client_context.custom["bedrockAgentCoreToolName"]
（"<ターゲット名>___<ツール名>"）に入れて呼ぶ（https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-lambda.html 、2026-11-20 確認）。
"""

from __future__ import annotations

import os
import time
from typing import Any

import boto3

from platformlab import sqlbuild

ATHENA = boto3.client("athena")
TABLE = os.environ.get("RESULTS_TABLE", "vlmlab_platform.results")
WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "vlmlab-platform")
TIMEOUT_S = 25
MAX_ROWS = 200


def tool_name(context: Any) -> str:
    custom = getattr(getattr(context, "client_context", None), "custom", None) or {}
    full = custom.get("bedrockAgentCoreToolName", "")
    return full.split("___", 1)[1] if "___" in full else full


def run_query(q: sqlbuild.Query) -> dict[str, Any]:
    args: dict[str, Any] = {"QueryString": q.sql, "WorkGroup": WORKGROUP}
    if q.params:
        args["ExecutionParameters"] = list(q.params)
    qid = ATHENA.start_query_execution(**args)["QueryExecutionId"]
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        st = ATHENA.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = st["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if time.monotonic() > deadline:
            ATHENA.stop_query_execution(QueryExecutionId=qid)
            return {"error": f"query timed out after {TIMEOUT_S}s", "query_id": qid}
        time.sleep(0.5)
    if state != "SUCCEEDED":
        return {"error": st["Status"].get("StateChangeReason", state), "query_id": qid}
    rows = ATHENA.get_query_results(QueryExecutionId=qid, MaxResults=MAX_ROWS + 1)["ResultSet"]["Rows"]
    header = [c.get("VarCharValue") for c in rows[0]["Data"]] if rows else []
    data = [dict(zip(header, [c.get("VarCharValue") for c in r["Data"]])) for r in rows[1:]]
    scanned = st.get("Statistics", {}).get("DataScannedInBytes")
    return {"rows": data, "n_rows": len(data), "data_scanned_bytes": scanned, "query_id": qid}


def handler(event: dict, context: Any) -> dict:
    name = tool_name(context)
    try:
        q = sqlbuild.build(name, TABLE, event or {})
    except ValueError as e:
        return {"error": f"invalid input: {e}"}
    return run_query(q)
