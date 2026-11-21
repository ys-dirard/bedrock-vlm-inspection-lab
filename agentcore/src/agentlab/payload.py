"""Runtime の入口に渡すペイロード（12/16〜。docs/contracts.md の CI-AC-P1〜P5）。

11/30 の入口は S3 のメタデータと Step Functions が前提だった。12/16 からは、手元のスクリプトが画像を Base64 で
ペイロードに入れて InvokeAgentRuntime を直接呼ぶ。正解（label）は記録にだけ使い、判定の本体には渡さない（CI-PL-R5）。
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

from .record import ImageMeta
from .runcfg import PlatformRunConfig

REQUIRED = frozenset({"run_config", "image_id", "category", "label", "image_b64"})
OPTIONAL = frozenset({"ref_b64"})


@dataclass(frozen=True)
class JudgeRequest:
    cfg: PlatformRunConfig
    meta: ImageMeta
    target: bytes
    ref: bytes | None


def _b64(name: str, value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty base64 string")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"{name} is not valid base64: {e}") from e
    if not data:
        raise ValueError(f"{name} decodes to empty bytes")
    return data


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def parse_payload(payload: Any) -> JudgeRequest:
    """ペイロードを検証する。条件を満たさなければ ValueError（モデルは呼ばない）。"""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")  # CI-AC-P1
    keys = set(payload)
    missing, unknown = REQUIRED - keys, keys - REQUIRED - OPTIONAL
    if missing or unknown:
        raise ValueError(f"payload keys: missing {sorted(missing)}, unknown {sorted(unknown)}")  # CI-AC-P1
    if not isinstance(payload["run_config"], dict):
        raise ValueError("run_config must be a JSON object")
    cfg = PlatformRunConfig.from_dict(payload["run_config"])  # CI-AC-P2（CI-PL-G1・G2）
    label = payload["label"]
    if label not in ("normal", "anomaly"):
        raise ValueError(f"label must be normal/anomaly: {label!r}")  # CI-AC-P3
    image_id = _text("image_id", payload["image_id"])
    meta = ImageMeta(cfg.run_name, image_id, _text("category", payload["category"]), label, f"payload:{image_id}")
    target = _b64("image_b64", payload["image_b64"])  # CI-AC-P4
    ref_raw = payload.get("ref_b64")
    if cfg.n_refs > 0:
        ref = _b64("ref_b64", ref_raw)  # CI-AC-P5
    elif ref_raw is not None:
        raise ValueError("ref_b64 given but run_config.n_refs is 0")  # CI-AC-P5
    else:
        ref = None
    return JudgeRequest(cfg, meta, target, ref)


def build_payload(cfg: PlatformRunConfig, image_id: str, category: str, label: str,
                  target: bytes, ref: bytes | None) -> dict[str, Any]:
    """手元のスクリプトがペイロードを作る（parse_payload の逆）。"""
    from dataclasses import asdict

    p: dict[str, Any] = {"run_config": asdict(cfg), "image_id": image_id, "category": category, "label": label,
                         "image_b64": base64.b64encode(target).decode("ascii")}
    if ref is not None:
        p["ref_b64"] = base64.b64encode(ref).decode("ascii")
    return p
