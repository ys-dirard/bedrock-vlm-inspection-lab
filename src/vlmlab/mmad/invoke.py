"""Claude を Messages API の本文（InvokeModel）で呼ぶ。バッチ推論の出力の読み取りも同じ関数で行う（CI-MB1）。

Nova 2 Lite は既存の vlmlab.bedrock.BedrockConverse（Converse）で呼ぶ。
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from ..bedrock import RETRYABLE, ModelResponse


def response_from_anthropic(body: dict[str, Any], latency_s: float | None = None, attempts: int = 1) -> ModelResponse:
    """Messages API の応答（InvokeModel の本文、またはバッチ出力の modelOutput）を ModelResponse にする。

    text ブロックだけをつなぐ（thinking ブロックは raw_text に入れず、extra に数を残す）。
    """
    blocks = body.get("content") or []
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    usage = body.get("usage") or {}
    return ModelResponse(
        raw_text="".join(texts) if texts else None,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        latency_s=latency_s,
        stop_reason=body.get("stop_reason"),
        attempts=attempts,
        extra={"n_thinking_blocks": sum(1 for b in blocks if b.get("type") in ("thinking", "redacted_thinking"))},
    )


class BedrockInvokeAnthropic:
    """bedrock-runtime の invoke_model で、Messages API の本文をそのまま送る。"""

    def __init__(self, region: str, max_attempts: int = 6, base_delay_s: float = 2.0, client: Any = None):
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=120),
            )
        self.client = client
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s

    def invoke(self, model_id: str, body: dict[str, Any]) -> ModelResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        payload = json.dumps(body)
        last_error = None
        attempt = 0
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.invoke_model(
                    modelId=model_id, body=payload, contentType="application/json", accept="application/json",
                )
                out = json.loads(resp["body"].read())
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "ClientError")
                last_error = f"{code}: {e}"
                retryable = code in RETRYABLE
            except BotoCoreError as e:
                last_error = f"{type(e).__name__}: {e}"
                retryable = True
            else:
                return response_from_anthropic(out, time.perf_counter() - t0, attempt)
            if not retryable or attempt == self.max_attempts:
                break
            time.sleep(self.base_delay_s * 2 ** (attempt - 1) * (0.5 + random.random()))
        return ModelResponse(raw_text=None, attempts=attempt, api_error=last_error)
