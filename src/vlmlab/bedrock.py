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
# 応答の serviceTier.type → 設定ファイルの層の名前（費用の単価を引くときに使う）
RESOLVED_TO_TIER = {v: k for k, v in SERVICE_TIERS.items()}

# 画像と文の並べ方
# - refs_first:   [見本…, 検査対象, 指示]（この基盤の既定）
# - target_first: [検査対象, 指示, ref_note, 見本…]（AWS サンプル sample-generative-visual-inspection の並び。
#   サンプルは InvokeModel で見本を2通目の user メッセージとして渡すが、この基盤は Converse で
#   1通の user メッセージを送る作りなので、同じ順のまま1通にまとめる。2通に分けたときの挙動は確かめていない）
LAYOUTS = ("refs_first", "target_first")

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
    images: list[EncodedImage]  # 見本 → 検査対象の順（最後が検査対象）。渡す順は layout で決まる
    text: str  # 指示
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 300
    top_p: float | None = None
    layout: str = "refs_first"
    ref_note: str | None = None  # target_first のとき、見本の前に置く文
    service_tier: str = "standard"
    service_tier_explicit: bool = False  # True なら standard でも serviceTier={"type": "default"} を送る（CI-B3）
    additional_fields: dict[str, Any] | None = None


@dataclass
class ModelResponse:
    raw_text: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float | None = None  # 成功した1回の往復時間（再試行の待ちは含まない）
    latency_total_s: float | None = None  # 最初の試行の開始から戻るまで（失敗した試行と待ちを含む。CI-B4）
    stop_reason: str | None = None
    service_tier_resolved: str | None = None
    attempts: int = 1
    api_error: str | None = None  # 再試行しても失敗したときの例外名と内容
    retry_errors: list[str] = field(default_factory=list)  # 失敗した試行ごとのエラーコード（CI-B4）
    extra: dict[str, Any] = field(default_factory=dict)


class VisionModel(Protocol):
    def invoke(self, request: ModelRequest) -> ModelResponse: ...


def _image_block(img: EncodedImage) -> dict[str, Any]:
    return {"image": {"format": img.format, "source": {"bytes": img.data}}}


def build_content(req: ModelRequest) -> list[dict[str, Any]]:
    if req.layout == "refs_first":
        if req.ref_note:
            raise ValueError("ref_note is only used with layout 'target_first'")
        return [_image_block(i) for i in req.images] + [{"text": req.text}]
    if req.layout == "target_first":
        if not req.images:
            raise ValueError("layout 'target_first' requires the target image")
        *refs, target = req.images
        content: list[dict[str, Any]] = [_image_block(target), {"text": req.text}]
        if refs:
            if req.ref_note:
                content.append({"text": req.ref_note})
            content.extend(_image_block(i) for i in refs)
        return content
    raise ValueError(f"unknown layout {req.layout!r}; expected one of {LAYOUTS}")


def service_tier_sent(service_tier: str, explicit: bool = False) -> str | None:
    """Converse に送る serviceTier.type。standard は explicit のときだけ "default" を送り、ほかは送らない（CI-B3）。"""
    if service_tier not in SERVICE_TIERS:
        raise ValueError(f"unknown service_tier {service_tier!r}; expected one of {sorted(SERVICE_TIERS)}")
    if service_tier == "standard" and not explicit:
        return None
    return SERVICE_TIERS[service_tier]


def build_converse_kwargs(req: ModelRequest) -> dict[str, Any]:
    inference: dict[str, Any] = {"temperature": req.temperature, "maxTokens": req.max_tokens}
    if req.top_p is not None:
        inference["topP"] = req.top_p
    kwargs: dict[str, Any] = {
        "modelId": req.model_id,
        "messages": [{"role": "user", "content": build_content(req)}],
        "inferenceConfig": inference,
    }
    if req.system:
        kwargs["system"] = [{"text": req.system}]
    sent = service_tier_sent(req.service_tier, req.service_tier_explicit)
    if sent is not None:
        kwargs["serviceTier"] = {"type": sent}
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

    def __init__(self, region: str, max_attempts: int = 6, base_delay_s: float = 2.0, client: Any = None,
                 read_timeout_s: int = 120):
        if client is None:
            import boto3
            from botocore.config import Config

            # 再試行はこちらで行い、レイテンシを1回ごとに測る
            client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=read_timeout_s),
            )
        self.client = client
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s

    def invoke(self, request: ModelRequest) -> ModelResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        kwargs = build_converse_kwargs(request)
        last_error = None
        retry_errors: list[str] = []
        t_start = time.perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.converse(**kwargs)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "ClientError")
                last_error = f"{code}: {e}"
                retryable = code in RETRYABLE
            except BotoCoreError as e:  # 読み取りのタイムアウト、接続の失敗など
                code = type(e).__name__
                last_error = f"{code}: {e}"
                retryable = True
            else:
                t_end = time.perf_counter()
                out = response_from_converse(resp, t_end - t0, attempt)
                out.latency_total_s = t_end - t_start
                out.retry_errors = retry_errors
                return out
            retry_errors.append(code)
            if not retryable or attempt == self.max_attempts:
                break
            time.sleep(self.base_delay_s * 2 ** (attempt - 1) * (0.5 + random.random()))
        return ModelResponse(raw_text=None, attempts=attempt, api_error=last_error,
                             latency_total_s=time.perf_counter() - t_start, retry_errors=retry_errors)
