"""検査エージェントの AgentCore Runtime の入口。

Step Functions から SDK 統合（invokeAgentRuntime）で呼ばれる。
入力: {"bucket": ..., "key": "inbox/..."}。画像の素性は S3 のメタデータ（runname・imageid・category・label・runtime・stopsession）。
出力: {"results_key": ..., "record_line": "<1行の JSON>", "summary": {...}}。S3 と DynamoDB への書き込みは Step Functions がする。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

from judge import run_judge
from platformlab.record import ImageMeta, build_record, results_key, to_line
from platformlab.runcfg import PlatformRunConfig

JST = timezone(timedelta(hours=9))
REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
app = BedrockAgentCoreApp()
s3 = boto3.client("s3", region_name=REGION)


def _read(bucket: str, key: str) -> tuple[bytes, dict[str, str]]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read(), obj.get("Metadata", {})


def handle(payload: dict, session_id: str | None) -> dict:
    bucket, key = payload["bucket"], payload["key"]
    target_raw, metadata = _read(bucket, key)
    meta = ImageMeta.from_s3_metadata(key, metadata)
    cfg_raw, _ = _read(bucket, f"run-configs/{meta.run_name}.json")
    cfg = PlatformRunConfig.from_json(cfg_raw.decode("utf-8"))
    ref_raw = None
    if cfg.n_refs > 0:
        ref_raw, _ = _read(bucket, f"refs/{meta.category}.img")
    called_at = datetime.now(JST)
    out = run_judge(cfg, target_raw, ref_raw, lambda: CodeInterpreter(REGION))
    rec = build_record(cfg, meta, out, session_id, called_at.isoformat(timespec="seconds"))
    return {
        "results_key": results_key(called_at.strftime("%Y-%m-%d"), cfg.run_name, meta.image_id),
        "record_line": to_line(rec),
        "summary": {k: rec[k] for k in ("run_name", "image_id", "label", "pred", "cost_usd", "api_error",
                                        "input_tokens", "output_tokens", "n_tool_calls", "latency_s")},
    }


@app.entrypoint
def invoke(payload, context):
    return handle(payload, getattr(context, "session_id", None))


if __name__ == "__main__":
    app.run()
