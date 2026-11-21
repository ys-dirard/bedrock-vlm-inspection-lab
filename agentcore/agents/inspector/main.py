"""検査エージェントの AgentCore Runtime の入口（12/16〜）。

手元のスクリプト（scripts/invoke_runtime.py）が InvokeAgentRuntime を直接呼ぶ。ペイロードの形は agentlab/payload.py。
出力: {"record_line": "<1行の JSON>", "summary": {...}} か、ペイロードが正しくなければ {"error": "..."}。
Runtime は S3・DynamoDB などに何も書かない（記録は呼んだ側が保存する）。
判定の本体（judge.py・preprocess.py）は 11/30 の platform/agents/inspector から変えずに持ってきた。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

from agentlab.payload import parse_payload
from agentlab.record import build_record, to_line
from judge import run_judge

JST = timezone(timedelta(hours=9))
REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
SUMMARY_KEYS = ("run_name", "image_id", "label", "pred", "parse_error", "api_error", "input_tokens", "output_tokens",
                "cache_read_tokens", "cache_write_tokens", "cost_usd", "n_cycles", "n_tool_calls", "latency_s")
app = BedrockAgentCoreApp()


def handle(payload: dict, session_id: str | None, sandbox_factory=None) -> dict:
    try:
        req = parse_payload(payload)
    except ValueError as e:
        return {"error": f"bad payload: {e}"}
    factory = sandbox_factory or (lambda: CodeInterpreter(REGION))
    called_at = datetime.now(JST).isoformat(timespec="seconds")
    out = run_judge(req.cfg, req.target, req.ref, factory)
    rec = build_record(req.cfg, req.meta, out, session_id, called_at)
    return {"record_line": to_line(rec), "summary": {k: rec[k] for k in SUMMARY_KEYS}}


@app.entrypoint
def invoke(payload, context):
    return handle(payload, getattr(context, "session_id", None))


if __name__ == "__main__":
    app.run()
