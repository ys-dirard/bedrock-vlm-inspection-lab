"""判定の記録（11月の JSONL と同じキー）。docs/contracts.md の CI-PL-R1〜R6。

エージェントの中で作り、Step Functions が S3 に書く。集計は `vlmlab report` をそのまま使う。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vlmlab.cost import Price, call_cost
from vlmlab.parse import ParseSpec, parse_response

from .runcfg import PlatformRunConfig


@dataclass(frozen=True)
class ImageMeta:
    """S3 のオブジェクトのメタデータから読む、画像の素性。label はモデルに渡さない（CI-PL-R5）。"""

    run_name: str
    image_id: str
    category: str
    label: str
    key: str

    @classmethod
    def from_s3_metadata(cls, key: str, metadata: dict[str, str]) -> "ImageMeta":
        need = ("runname", "imageid", "category", "label")
        missing = [k for k in need if k not in metadata]
        if missing:
            raise ValueError(f"S3 object {key!r} lacks metadata {missing}")
        if metadata["label"] not in ("normal", "anomaly"):
            raise ValueError(f"label must be normal/anomaly: {metadata['label']!r}")
        return cls(metadata["runname"], metadata["imageid"], metadata["category"], metadata["label"], key)


@dataclass(frozen=True)
class AgentOutcome:
    """エージェントを1回走らせた結果（ターンの合計）。"""

    raw_text: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_s: float
    stop_reason: str | None
    n_cycles: int
    n_tool_calls: int
    used_code_interpreter: bool
    api_error: str | None = None


def price_of(cfg: PlatformRunConfig) -> Price:
    return Price(
        model_id=cfg.model_id,
        service_tier=cfg.service_tier,
        input_per_mtok=cfg.input_per_mtok,
        output_per_mtok=cfg.output_per_mtok,
        cache_read_per_mtok=cfg.cache_read_per_mtok,
        cache_write_per_mtok=cfg.cache_write_per_mtok,
        source_url=cfg.price_source,
        checked_on=cfg.price_checked_on,
    )


def build_record(cfg: PlatformRunConfig, meta: ImageMeta, out: AgentOutcome,
                 runtime_session_id: str | None, called_at: str) -> dict[str, Any]:
    if meta.run_name != cfg.run_name:
        raise ValueError(f"image run_name {meta.run_name!r} != config {cfg.run_name!r}")
    spec = ParseSpec.from_dict(cfg.parse)
    pr = parse_response(out.raw_text, spec) if out.api_error is None else None  # CI-PL-R4
    return {
        "run_name": cfg.run_name,
        "run_id": cfg.run_id,
        "repeat": 1,
        "image_id": meta.image_id,
        "category": meta.category,
        "label": meta.label,
        "path": meta.key,
        **cfg.identity(),
        "service_tier_resolved": None,
        "raw_text": out.raw_text if out.api_error is None else None,
        "parsed": pr.parsed if pr else None,
        "parse_error": pr.parse_error if pr else None,
        "input_tokens": out.input_tokens,
        "output_tokens": out.output_tokens,
        "cache_read_tokens": out.cache_read_tokens,
        "cache_write_tokens": out.cache_write_tokens,
        "latency_s": out.latency_s,
        "stop_reason": out.stop_reason,
        "cost_usd": call_cost(price_of(cfg), out.input_tokens, out.output_tokens,
                              out.cache_read_tokens, out.cache_write_tokens),  # CI-PL-R3
        "attempts": 1,
        "api_error": out.api_error,
        "server_latency_ms": None,
        "timestamp": called_at,
        # CI-PL-R2: 足すキー
        "pred": pr.pred if pr else None,
        "confidence": pr.confidence if pr else None,
        "defect": pr.defect if pr else None,
        "n_cycles": out.n_cycles,
        "n_tool_calls": out.n_tool_calls,
        "used_code_interpreter": out.used_code_interpreter,
        "tools_enabled": cfg.tools_enabled,
        "prompt_cache": cfg.prompt_cache,
        "runtime_session_id": runtime_session_id,
        "platform_condition": cfg.platform_condition,
    }


def to_line(record: dict[str, Any]) -> str:
    """1行の JSON（CI-PL-R6）。parsed は入れ子のまま（Athena では文字列の列として読む）。"""
    rec = dict(record)
    if rec.get("parsed") is not None:
        rec["parsed"] = json.dumps(rec["parsed"], ensure_ascii=False)
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


def results_key(called_date: str, run_name: str, image_id: str) -> str:
    """S3 の記録の置き場所。called_date は YYYY-MM-DD（JST）。パーティション（dt=）の直下に置く。"""
    return f"results/dt={called_date}/{run_name}__{image_id.replace('/', '__')}.json"
