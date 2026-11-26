"""Nova Multimodal Embeddings で、画像・文書のページ画像・テキストを埋め込む（12/20・12/21・12/24 の記事）。

store.py は `Embedder`（embed を持つもの）だけを知る。テストでは偽物に差し替える。
依頼と応答の形: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html（2026-11-21 確認）
- embeddingPurpose: 索引は GENERIC_INDEX。検索のクエリは、索引の中身に合わせて
  TEXT_RETRIEVAL（テキストだけ）/ IMAGE_RETRIEVAL（STANDARD_IMAGE の画像だけ）/
  DOCUMENT_RETRIEVAL（DOCUMENT_IMAGE の画像だけ）/ GENERIC_RETRIEVAL（混在）
- テキストは 8,192 文字まで。インラインの入力は Base64 にしたあとで 25 MB まで
"""

from __future__ import annotations

import base64
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..bedrock import RETRYABLE
from ..embed import EmbedResult, tokens_from_headers

PURPOSES = ("GENERIC_INDEX", "GENERIC_RETRIEVAL", "TEXT_RETRIEVAL", "IMAGE_RETRIEVAL",
            "DOCUMENT_RETRIEVAL", "CLASSIFICATION", "CLUSTERING")
DIMENSIONS = (256, 384, 1024, 3072)
DETAILS = ("STANDARD_IMAGE", "DOCUMENT_IMAGE")
MAX_TEXT_CHARS = 8192
MAX_INLINE_BYTES = 25 * 1024 * 1024  # Base64 にしたあと


@dataclass(frozen=True)
class EmbedInput:
    """埋め込む1件。modality が image なら data/format/detail、text なら text を使う。"""

    modality: str  # "image" | "text"
    text: str | None = None
    data: bytes | None = None
    format: str | None = None  # png / jpeg / gif / webp
    detail: str | None = None  # STANDARD_IMAGE / DOCUMENT_IMAGE

    @property
    def expected_type(self) -> str:
        return "IMAGE" if self.modality == "image" else "TEXT"


def build_body(inp: EmbedInput, purpose: str, dimension: int) -> dict[str, Any]:
    """InvokeModel の本文（CI-S1）。形が仕様の外なら、呼ぶ前に ValueError。"""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown embeddingPurpose {purpose!r}")
    if dimension not in DIMENSIONS:
        raise ValueError(f"embeddingDimension must be one of {DIMENSIONS}")
    params: dict[str, Any] = {"embeddingPurpose": purpose, "embeddingDimension": dimension}
    if inp.modality == "text":
        if not inp.text or not inp.text.strip():
            raise ValueError("text is empty")
        if len(inp.text) > MAX_TEXT_CHARS:
            raise ValueError(f"text has {len(inp.text)} chars; max is {MAX_TEXT_CHARS}")
        # 長すぎる文は切らずに失敗させる（黙って切ると、何を埋め込んだか分からなくなる）
        params["text"] = {"truncationMode": "NONE", "value": inp.text}
    elif inp.modality == "image":
        if inp.detail not in DETAILS:
            raise ValueError(f"detailLevel must be one of {DETAILS}")
        if inp.format not in ("png", "jpeg", "gif", "webp") or not inp.data:
            raise ValueError("image needs data and a format of png/jpeg/gif/webp")
        b64 = base64.b64encode(inp.data).decode("ascii")
        if len(b64) > MAX_INLINE_BYTES:
            raise ValueError(f"inline image is {len(b64)} bytes after Base64; max is {MAX_INLINE_BYTES}")
        params["image"] = {"detailLevel": inp.detail, "format": inp.format, "source": {"bytes": b64}}
    else:
        raise ValueError(f"unknown modality {inp.modality!r}")
    return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}


def parse_body(body: dict[str, Any], dimension: int, expected_type: str) -> list[float]:
    """応答から埋め込みを1つ取り出す（CI-S2）。形が違えば ValueError。"""
    embs = body.get("embeddings")
    if not isinstance(embs, list) or len(embs) != 1 or not isinstance(embs[0], dict):
        raise ValueError("expected exactly one embedding")
    e = embs[0]
    if e.get("embeddingType") != expected_type:
        raise ValueError(f"embeddingType is {e.get('embeddingType')!r}, not {expected_type!r}")
    if "truncatedCharLength" in e:
        raise ValueError(f"text was truncated at {e['truncatedCharLength']} chars")
    vec = e.get("embedding")
    if not isinstance(vec, list) or len(vec) != dimension:
        raise ValueError(f"embedding length is {len(vec) if isinstance(vec, list) else '-'}, expected {dimension}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vec):
        raise ValueError("embedding has a non-finite or non-numeric value")
    if not any(v != 0 for v in vec):
        raise ValueError("embedding is all zeros (cosine similarity undefined)")
    return [float(v) for v in vec]


class Embedder(Protocol):
    def embed(self, inp: EmbedInput) -> EmbedResult: ...


class NovaMultiEmbedder:
    """InvokeModel で Nova Multimodal Embeddings を呼ぶ。再試行の規則は bedrock.BedrockConverse と同じ（CI-B1）。"""

    def __init__(self, model_id: str, region: str, purpose: str, dimension: int,
                 max_attempts: int = 6, base_delay_s: float = 2.0, client: Any = None):
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("bedrock-runtime", region_name=region,
                                  config=Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=60))
        self.client, self.model_id = client, model_id
        self.purpose, self.dimension = purpose, dimension
        self.max_attempts, self.base_delay_s = max_attempts, base_delay_s

    def embed(self, inp: EmbedInput) -> EmbedResult:
        from botocore.exceptions import BotoCoreError, ClientError

        body = json.dumps(build_body(inp, self.purpose, self.dimension))
        last_error, attempt = None, 0
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.invoke_model(modelId=self.model_id, body=body,
                                                contentType="application/json", accept="application/json")
                payload = json.loads(resp["body"].read())
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "ClientError")
                last_error, retryable = f"{code}: {e}", code in RETRYABLE
            except BotoCoreError as e:
                last_error, retryable = f"{type(e).__name__}: {e}", True
            else:
                latency = time.perf_counter() - t0
                headers = resp.get("ResponseMetadata", {}).get("HTTPHeaders", {})
                tokens = tokens_from_headers(headers)
                try:
                    vec = parse_body(payload, self.dimension, inp.expected_type)
                except ValueError as e:  # 課金された呼び出しなので、トークン数は残す
                    return EmbedResult(None, tokens, latency, attempt, f"BadEmbedding: {e}", sorted(headers))
                return EmbedResult(vec, tokens, latency, attempt, None, sorted(headers))
            if not retryable or attempt == self.max_attempts:
                break
            time.sleep(self.base_delay_s * 2 ** (attempt - 1) * (0.5 + random.random()))
        return EmbedResult(None, None, None, attempt, last_error)
