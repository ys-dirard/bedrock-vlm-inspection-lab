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
    temperature: float | None = 0.0  # None なら送らない（推論 high のとき。CI-B5）
    max_tokens: int | None = 300  # None なら送らない（同上）
    top_p: float | None = None
    layout: str = "refs_first"
    ref_note: str | None = None  # target_first のとき、見本の前に置く文
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


def reasoning_effort(additional_fields: dict[str, Any] | None) -> str | None:
    """additionalModelRequestFields の推論の深さ。推論を有効にしていなければ None。

    Nova 2 の書き方（reasoningConfig.type が "enabled" のときの maxReasoningEffort）。
    https://docs.aws.amazon.com/nova/latest/nova2-userguide/extended-thinking.html（2026-11-16 確認）
    """
    rc = (additional_fields or {}).get("reasoningConfig")
    if not isinstance(rc, dict) or rc.get("type") != "enabled":
        return None
    return rc.get("maxReasoningEffort")


def check_reasoning_constraints(
    temperature: float | None, max_tokens: int | None, top_p: float | None,
    additional_fields: dict[str, Any] | None,
) -> None:
    """推論 high では temperature・topP・topK・maxTokens を送れない（CI-B5）。呼ぶ前に止める。

    ユーザーガイド：「when using "high", temp, topP, and maxToken must be unset」
    「Temperature, topP and topK cannot be used with maxReasoningEffort set to high」
    """
    if reasoning_effort(additional_fields) != "high":
        return
    bad = [name for name, v in (("temperature", temperature), ("maxTokens", max_tokens), ("topP", top_p)) if v is not None]
    if "topK" in (additional_fields or {}):
        bad.append("topK")
    if bad:
        raise ValueError(f"reasoning effort 'high' requires {bad} to be unset (null)")


def build_converse_kwargs(req: ModelRequest) -> dict[str, Any]:
    check_reasoning_constraints(req.temperature, req.max_tokens, req.top_p, req.additional_fields)
    inference: dict[str, Any] = {}
    if req.temperature is not None:
        inference["temperature"] = req.temperature
    if req.max_tokens is not None:
        inference["maxTokens"] = req.max_tokens
    if req.top_p is not None:
        inference["topP"] = req.top_p
    kwargs: dict[str, Any] = {
        "modelId": req.model_id,
        "messages": [{"role": "user", "content": build_content(req)}],
    }
    if inference:
        kwargs["inferenceConfig"] = inference
    if req.system:
        kwargs["system"] = [{"text": req.system}]
    if req.service_tier != "standard":
        if req.service_tier not in SERVICE_TIERS:
            raise ValueError(f"unknown service_tier {req.service_tier!r}")
        kwargs["serviceTier"] = {"type": SERVICE_TIERS[req.service_tier]}
    if req.additional_fields:
        kwargs["additionalModelRequestFields"] = req.additional_fields
    return kwargs


def split_content(blocks: list[dict[str, Any]]) -> tuple[str | None, int, str | None]:
    """応答の content を、本文と推論に分ける（CI-B3、CI-B4）。

    - 本文: 最上位に "text" を持つブロックだけを、出てきた順につなぐ。無ければ None。
      推論のブロック（{"reasoningContent": {...}}）や toolUse は本文に入れない
    - 推論: reasoningContent のブロックの数と、その reasoningText.text をつないだもの
      （Nova 2 では "[REDACTED]"）。redactedContent（バイト列）は "<redactedContent>" と書く
    """
    texts: list[str] = []
    reasoning: list[str] = []
    n_reasoning = 0
    for b in blocks:
        if "text" in b and isinstance(b["text"], str):
            texts.append(b["text"])
        elif "reasoningContent" in b:
            n_reasoning += 1
            rc = b["reasoningContent"] or {}
            if "reasoningText" in rc:
                reasoning.append(str((rc["reasoningText"] or {}).get("text", "")))
            elif "redactedContent" in rc:
                reasoning.append("<redactedContent>")
    return ("".join(texts) if texts else None), n_reasoning, ("".join(reasoning) if reasoning else None)


def response_from_converse(resp: dict[str, Any], latency_s: float, attempts: int) -> ModelResponse:
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    text, n_reasoning, reasoning_text = split_content(blocks)
    usage = resp.get("usage", {})
    return ModelResponse(
        raw_text=text,
        input_tokens=int(usage.get("inputTokens", 0)),
        output_tokens=int(usage.get("outputTokens", 0)),
        cache_read_tokens=int(usage.get("cacheReadInputTokens", 0)),
        cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0)),
        latency_s=latency_s,
        stop_reason=resp.get("stopReason"),
        service_tier_resolved=(resp.get("serviceTier") or {}).get("type"),
        attempts=attempts,
        extra={"server_latency_ms": resp.get("metrics", {}).get("latencyMs"),
               "reasoning_blocks": n_reasoning, "reasoning_text": reasoning_text},
    )


class BedrockConverse:
    """Amazon Bedrock の Converse API で呼ぶ。"""

    def __init__(self, region: str, max_attempts: int = 6, base_delay_s: float = 2.0, client: Any = None,
                 read_timeout_s: int = 120):
        if client is None:
            import boto3
            from botocore.config import Config

            # 再試行はこちらで行い、レイテンシを1回ごとに測る。
            # 推論を有効にすると応答が長くなるので、設定ファイルの read_timeout_s で延ばせる
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
