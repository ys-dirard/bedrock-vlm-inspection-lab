"""判定の Lambda（S3 の画像を Bedrock の Converse で判定し、人に回すかを決める）。

入力（Step Functions から）: {"image_id": str, "bucket": str, "key": str, "ref_keys": [str, ...]}
出力: judge_result()（CI-HR6・HR7）。Bedrock の呼び出しが失敗したら例外を投げる（自動で OK にしない）。

環境変数
- MODEL_ID（例 jp.amazon.nova-2-lite-v1:0）、PROMPT_TEXT（プロンプトの本文）
- INFERENCE_JSON（例 {"temperature": 0, "maxTokens": 300}）、PARSE_SPEC_JSON（vlmlab.parse.ParseSpec）
- ROUTE_POLICY_JSON（vlmlab.review.route.RoutePolicy）
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..parse import ParseSpec, parse_response
from .route import RoutePolicy, decide

_FORMATS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}
RAW_TEXT_LIMIT = 4000  # 状態の入出力（256 KiB）を超えないよう、応答の本文はここで切る


def image_format(key: str) -> str:
    ext = os.path.splitext(key)[1].lower()
    if ext not in _FORMATS:
        raise ValueError(f"unsupported image extension {ext!r} for {key}")
    return _FORMATS[ext]


def validate_event(event: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    """CI-HR6（pre）：image_id・bucket・key は空でない文字列、ref_keys は文字列の配列。"""
    for k in ("image_id", "bucket", "key"):
        if not isinstance(event.get(k), str) or not event[k]:
            raise ValueError(f"event.{k} must be a non-empty string")
    refs = event.get("ref_keys", [])
    if not isinstance(refs, list) or not all(isinstance(r, str) and r for r in refs):
        raise ValueError("event.ref_keys must be a list of non-empty strings")
    return event["image_id"], event["bucket"], event["key"], refs


def build_messages(images: list[tuple[bytes, str]], prompt_text: str) -> list[dict[str, Any]]:
    """見本 → 検査対象 → 指示の順（検証基盤の layout: refs_first と同じ）。"""
    content: list[dict[str, Any]] = [{"image": {"format": f, "source": {"bytes": b}}} for b, f in images]
    content.append({"text": prompt_text})
    return [{"role": "user", "content": content}]


def judge_result(image_id: str, model_id: str, resp: dict[str, Any], latency_s: float,
                 spec: ParseSpec, policy: RoutePolicy) -> dict[str, Any]:
    """CI-HR7：Converse の応答から、読み取り・回し方・usage をまとめる（JSON にできる値だけ）。"""
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    texts = [b["text"] for b in blocks if "text" in b]
    raw_text = "".join(texts) if texts else None
    pr = parse_response(raw_text, spec)
    d = decide(policy, image_id, pr.pred, pr.confidence)
    usage = resp.get("usage", {})
    return {
        "image_id": image_id,
        "model_id": model_id,
        "judgement": pr.pred,
        "confidence": pr.confidence,
        "defect": pr.defect,
        "parse_error": pr.parse_error,
        "anomaly_score": d.anomaly_score,
        "route": d.route,
        "route_reason": d.reason,
        "auto_judgement": d.auto_judgement,
        "input_tokens": int(usage.get("inputTokens", 0)),
        "output_tokens": int(usage.get("outputTokens", 0)),
        "stop_reason": resp.get("stopReason"),
        "latency_s": round(latency_s, 3),
        "raw_text": None if raw_text is None else raw_text[:RAW_TEXT_LIMIT],
        "raw_text_truncated": raw_text is not None and len(raw_text) > RAW_TEXT_LIMIT,
    }


class Judge:
    def __init__(self, s3: Any, bedrock: Any, model_id: str, prompt_text: str, inference: dict[str, Any],
                 spec: ParseSpec, policy: RoutePolicy):
        self.s3, self.bedrock = s3, bedrock
        self.model_id, self.prompt_text, self.inference = model_id, prompt_text, inference
        self.spec, self.policy = spec, policy

    @classmethod
    def from_env(cls, s3: Any = None, bedrock: Any = None) -> "Judge":
        if s3 is None or bedrock is None:
            import boto3
            from botocore.config import Config

            cfg = Config(retries={"max_attempts": 4, "mode": "standard"}, read_timeout=60)
            s3 = s3 or boto3.client("s3")
            bedrock = bedrock or boto3.client("bedrock-runtime", config=cfg)
        return cls(
            s3, bedrock,
            model_id=os.environ["MODEL_ID"],
            prompt_text=os.environ["PROMPT_TEXT"],
            inference=json.loads(os.environ["INFERENCE_JSON"]),
            spec=ParseSpec.from_dict(json.loads(os.environ.get("PARSE_SPEC_JSON", "{}"))),
            policy=RoutePolicy.from_json(os.environ["ROUTE_POLICY_JSON"]),
        )

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        image_id, bucket, key, refs = validate_event(event)
        images = []
        for k in [*refs, key]:
            body = self.s3.get_object(Bucket=bucket, Key=k)["Body"].read()
            images.append((body, image_format(k)))
        t0 = time.perf_counter()
        resp = self.bedrock.converse(modelId=self.model_id, messages=build_messages(images, self.prompt_text),
                                     inferenceConfig=self.inference)
        return judge_result(image_id, self.model_id, resp, time.perf_counter() - t0, self.spec, self.policy)


_judge: Judge | None = None


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _judge
    if _judge is None:
        _judge = Judge.from_env()
    return _judge(event)
