"""分析エージェント（AgentCore Runtime）。自然文の質問に、Gateway の向こうの決まった集計ツールだけで答える。

- ツールは Gateway（受信の認証は IAM＝SigV4）の MCP で取る。Cognito は使わない
- 会話の文脈は AgentCore Memory の短期記憶（セッション ID ごと）
入力: {"prompt": "...", "session_id": "..."（省略時は Runtime のセッション ID）}
出力: {"answer": "...", "input_tokens": .., "output_tokens": .., "n_tool_calls": .., "latency_s": .., "cost_usd": ..}
"""

from __future__ import annotations

import os
import time

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient

from vlmlab.cost import Price, call_cost

REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
GATEWAY_URL = os.environ["GATEWAY_URL"]
MEMORY_ID = os.environ["MEMORY_ID"]
MODEL_ID = os.environ.get("MODEL_ID", "jp.amazon.nova-2-lite-v1:0")
PRICE = Price(MODEL_ID, "standard", float(os.environ.get("PRICE_INPUT_PER_MTOK", "0.396")),
              float(os.environ.get("PRICE_OUTPUT_PER_MTOK", "3.311")))

SYSTEM = """あなたは外観検査の記録を集計するアシスタントです。
数字は必ずツールの結果から答えてください。ツールで分からないことは、分からないと答えてください。
記録の列: run_name（実験の名前）、model_id、category（品目）、label（正解 normal/anomaly）、pred（判定 OK/NG、読めなければ null）。
再現率は不良品を NG と言えた割合、誤検知率は良品を NG と言った割合です。"""

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload, context):
    session_id = payload.get("session_id") or getattr(context, "session_id", None) or "default-session"
    mcp = MCPClient(lambda: aws_iam_streamablehttp_client(GATEWAY_URL, "bedrock-agentcore", REGION))
    memory = AgentCoreMemorySessionManager(
        AgentCoreMemoryConfig(memory_id=MEMORY_ID, session_id=session_id, actor_id="analyst"), region_name=REGION)
    started = time.perf_counter()
    with mcp:
        agent = Agent(model=BedrockModel(model_id=MODEL_ID, region_name=REGION, temperature=0.0, max_tokens=1024,
                                         streaming=False),
                      tools=mcp.list_tools_sync(), system_prompt=SYSTEM, session_manager=memory,
                      callback_handler=None)
        result = agent(payload["prompt"])
    u = result.metrics.accumulated_usage
    n_in, n_out = int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))
    return {
        "answer": "".join(b.get("text", "") for b in result.message.get("content", [])),
        "input_tokens": n_in,
        "output_tokens": n_out,
        "n_cycles": result.metrics.cycle_count,
        "n_tool_calls": sum(m.call_count for m in result.metrics.tool_metrics.values()),
        "latency_s": round(time.perf_counter() - started, 3),
        "cost_usd": call_cost(PRICE, n_in, n_out),
    }


if __name__ == "__main__":
    app.run()
