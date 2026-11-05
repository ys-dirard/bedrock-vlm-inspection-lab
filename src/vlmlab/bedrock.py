"""Converse の呼び出し、再試行、usage とレイテンシの記録。

run.py は `VisionModel`（invoke を持つもの）だけを知る。テストでは偽物に差し替える。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .data import EncodedImage

# 設定ファイルの層の名前 → Converse の serviceTier.type
# https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html（2026-11-05 確認）
SERVICE_TIERS = {"standard": "default", "flex": "flex", "priority": "priority"}

RETRYABLE = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
}


@dataclass(frozen=True)
class ModelRequest:
    model_id: str
    images: list[EncodedImage]  # 並べた順に渡す（見本 → 検査対象）
    text: str  # 画像のあとに置く指示
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 300
    service_tier: str = "standard"
    additional_fields: dict[str, Any] | None = None


@dataclass
class ModelResponse:
    raw_text: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float | None = None  # 成功した1回の往復時間（再試行の待ちは含まない）
    stop_reason: str | None = None
    service_tier_resolved: str | None = None
    attempts: int = 1
    api_error: str | None = None  # 再試行しても失敗したときの例外名と内容
    extra: dict[str, Any] = field(default_factory=dict)


class VisionModel(Protocol):
    def invoke(self, request: ModelRequest) -> ModelResponse: ...


def build_converse_kwargs(req: ModelRequest) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"image": {"format": img.format, "source": {"bytes": img.data}}} for img in req.images
    ]
    content.append({"text": req.text})
    kwargs: dict[str, Any] = {
        "modelId": req.model_id,
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"temperature": req.temperature, "maxTokens": req.max_tokens},
    }
    if req.system:
        kwargs["system"] = [{"text": req.system}]
    if req.service_tier != "standard":
        if req.service_tier not in SERVICE_TIERS:
            raise ValueError(f"unknown service_tier {req.service_tier!r}")
        kwargs["serviceTier"] = {"type": SERVICE_TIERS[req.service_tier]}
    if req.additional_fields:
        kwargs["additionalModelRequestFields"] = req.additional_fields
    return kwargs


def response_from_converse(resp: dict[str, Any], latency_s: float, attempts: int) -> ModelResponse:
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    texts = [b["text"] for b in blocks if "text" in b]
    usage = resp.get("usage", {})
    return ModelResponse(
        raw_text="".join(texts) if texts else None,
        input_tokens=int(usage.get("inputTokens", 0)),
        output_tokens=int(usage.get("outputTokens", 0)),
        cache_read_tokens=int(usage.get("cacheReadInputTokens", 0)),
        cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0)),
        latency_s=latency_s,
        stop_reason=resp.get("stopReason"),
        service_tier_resolved=(resp.get("serviceTier") or {}).get("type"),
        attempts=attempts,
        extra={"server_latency_ms": resp.get("metrics", {}).get("latencyMs")},
    )


class BedrockConverse:
    """Amazon Bedrock の Converse API で呼ぶ。"""

    def __init__(self, region: str, max_attempts: int = 6, base_delay_s: float = 2.0, client: Any = None):
        if client is None:
            import boto3
            from botocore.config import Config

            # 再試行はこちらで行い、レイテンシを1回ごとに測る
            client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=120),
            )
        self.client = client
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s

    def invoke(self, request: ModelRequest) -> ModelResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        kwargs = build_converse_kwargs(request)
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.converse(**kwargs)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "ClientError")
                last_error = f"{code}: {e}"
                retryable = code in RETRYABLE
            except BotoCoreError as e:  # 読み取りのタイムアウト、接続の失敗など
                last_error = f"{type(e).__name__}: {e}"
                retryable = True
            else:
                return response_from_converse(resp, time.perf_counter() - t0, attempt)
            if not retryable or attempt == self.max_attempts:
                break
            time.sleep(self.base_delay_s * 2 ** (attempt - 1) * (0.5 + random.random()))
        return ModelResponse(raw_text=None, attempts=attempt, api_error=last_error)
