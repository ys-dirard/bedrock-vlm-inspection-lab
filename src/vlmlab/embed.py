"""Nova Multimodal Embeddings の呼び出しと、k近傍法での判定（条件 E）。

embed_run.py は `ImageEmbedder`（embed を持つもの）だけを知る。テストでは偽物に差し替える。
依頼と応答の形: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html（2026-11-05 確認）
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .bedrock import RETRYABLE
from .data import EncodedImage

PURPOSES = ("GENERIC_INDEX", "CLASSIFICATION", "CLUSTERING", "IMAGE_RETRIEVAL", "GENERIC_RETRIEVAL")
DIMENSIONS = (256, 384, 1024, 3072)
# 応答の本文にトークン数は無い。InvokeModel の HTTP ヘッダーにあればそれを使う（無ければ None）
TOKEN_HEADER = "x-amzn-bedrock-input-token-count"


def build_embed_body(img: EncodedImage, purpose: str, dimension: int) -> dict[str, Any]:
    if purpose not in PURPOSES:
        raise ValueError(f"unknown embeddingPurpose {purpose!r}")
    if dimension not in DIMENSIONS:
        raise ValueError(f"embeddingDimension must be one of {DIMENSIONS}")
    import base64
    return {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": purpose,
            "embeddingDimension": dimension,
            # 単価（画像1枚あたり）は detailLevel で変わるので、既定の STANDARD_IMAGE を明示する
            "image": {"detailLevel": "STANDARD_IMAGE", "format": img.format, "source": {"bytes": base64.b64encode(img.data).decode("ascii")}},
        },
    }


def parse_embed_body(body: dict[str, Any], dimension: int) -> list[float]:
    """応答から画像の埋め込みを1つ取り出す（CI-E1）。形が違えば ValueError。"""
    embs = body.get("embeddings")
    if not isinstance(embs, list) or len(embs) != 1 or not isinstance(embs[0], dict):
        raise ValueError(f"expected exactly one embedding, got {type(embs).__name__} of len "
                         f"{len(embs) if isinstance(embs, list) else '-'}")
    e = embs[0]
    if e.get("embeddingType") != "IMAGE":
        raise ValueError(f"embeddingType is {e.get('embeddingType')!r}, not 'IMAGE'")
    vec = e.get("embedding")
    if not isinstance(vec, list) or len(vec) != dimension:
        raise ValueError(f"embedding length is {len(vec) if isinstance(vec, list) else '-'}, expected {dimension}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vec):
        raise ValueError("embedding has a non-finite or non-numeric value")
    if not any(v != 0 for v in vec):
        raise ValueError("embedding is all zeros (cosine distance undefined)")
    return [float(v) for v in vec]


def tokens_from_headers(headers: dict[str, str] | None) -> int | None:
    """HTTP ヘッダーの入力トークン数（CI-E2）。無い・数でない・負なら None（0 にしない）。"""
    if not headers:
        return None
    lower = {k.lower(): v for k, v in headers.items()}
    v = lower.get(TOKEN_HEADER)
    try:
        n = int(v) if v is not None else None
    except ValueError:
        return None
    return n if n is not None and n >= 0 else None


@dataclass
class EmbedResult:
    vector: list[float] | None
    input_tokens: int | None
    latency_s: float | None = None
    attempts: int = 1
    api_error: str | None = None
    header_names: list[str] | None = None  # 試運転で、トークン数のヘッダーがあるかを確かめるため


class ImageEmbedder(Protocol):
    def embed(self, img: EncodedImage) -> EmbedResult: ...


class NovaEmbedder:
    """InvokeModel で Nova Multimodal Embeddings を呼ぶ。再試行は bedrock.BedrockConverse と同じ規則（CI-B1）。"""

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

    def embed(self, img: EncodedImage) -> EmbedResult:
        from botocore.exceptions import BotoCoreError, ClientError

        body = json.dumps(build_embed_body(img, self.purpose, self.dimension))
        last_error = None
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
                try:
                    vec = parse_embed_body(payload, self.dimension)
                except ValueError as e:  # 課金された呼び出しなので、トークン数は残す
                    return EmbedResult(None, tokens_from_headers(headers), latency, attempt,
                                       f"BadEmbedding: {e}", sorted(headers))
                return EmbedResult(vec, tokens_from_headers(headers), latency, attempt, None, sorted(headers))
            if not retryable or attempt == self.max_attempts:
                break
            time.sleep(self.base_delay_s * 2 ** (attempt - 1) * (0.5 + random.random()))
        return EmbedResult(None, None, None, attempt, last_error)


def knn_predict(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray, k: int) -> list[tuple[str, float]]:
    """k近傍法（コサイン距離）。各テストの (判定 OK/NG, 近傍のうち不良品の割合) を返す（CI-E3）。

    判定は多数決（不良品の割合 > 0.5 なら NG）。k は奇数に限る（同票を作らない）。
    距離が同じ近傍は、学習データの並び順で先のものを取る（scikit-learn の brute と同じ）。
    """
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd number")
    if len(train_x) != len(train_y) or len(train_x) < k:
        raise ValueError("need at least k labelled training vectors")
    bad = set(train_y) - {"normal", "anomaly"}
    if bad:
        raise ValueError(f"train labels must be normal/anomaly, got {sorted(bad)}")
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    clf.fit(np.asarray(train_x, dtype=float), np.array(train_y))
    proba = clf.predict_proba(np.asarray(test_x, dtype=float))
    classes = list(clf.classes_)
    p_anom = proba[:, classes.index("anomaly")] if "anomaly" in classes else np.zeros(len(test_x))
    return [("NG" if p > 0.5 else "OK", float(round(p, 9))) for p in p_anom]
