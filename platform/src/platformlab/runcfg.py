"""実験の設定（S3 の run-configs/<run_name>.json）。docs/contracts.md の CI-PL-G1〜G4。"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))
_RUN_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def prompt_sha(text: str) -> str:
    """CI-PL-G4。11月の run.call_identity と同じ作り方（system なし、prompt_vars なし）。"""
    payload = json.dumps([text, None, {}], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class PlatformRunConfig:
    run_name: str
    run_id: str
    model_id: str
    prompt_version: str
    prompt: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    price_source: str | None = None
    price_checked_on: str | None = None
    tools_enabled: bool = True
    platform_condition: str = ""
    region: str = "ap-northeast-1"
    service_tier: str = "standard"
    temperature: float = 0.0
    max_tokens: int = 1024
    n_refs: int = 1
    max_tool_calls: int = 3
    image_long_edge: int | None = None
    parse: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_sha(self) -> str:
        return prompt_sha(self.prompt)

    def identity(self) -> dict[str, Any]:
        """記録に入れる設定のキー（11月の run.py の identity と同じ名前）。"""
        return {
            "model_id": self.model_id,
            "service_tier_requested": self.service_tier,
            "prompt_version": self.prompt_version,
            "prompt_sha": self.prompt_sha,
            "n_refs": self.n_refs,
            "image_long_edge": self.image_long_edge,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": None,
            "layout": "refs_first",
            "ref_note": None,
            "additional_fields": None,
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlatformRunConfig":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown run config keys: {sorted(unknown)}")  # CI-PL-G1
        cfg = cls(**d)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, text: str) -> "PlatformRunConfig":
        return cls.from_dict(json.loads(text))

    def validate(self) -> None:
        if not _RUN_NAME.match(self.run_name):
            raise ValueError(f"run_name must match {_RUN_NAME.pattern}: {self.run_name!r}")  # CI-PL-G1
        if self.input_per_mtok is None or self.output_per_mtok is None:
            raise ValueError("input/output price must not be null (CI-C5)")  # CI-PL-G2
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be >= 0")
        if self.image_long_edge is not None and self.image_long_edge < 64:
            raise ValueError("image_long_edge must be >= 64 or null")


def new_run_config(**kwargs: Any) -> PlatformRunConfig:
    """run_id を1回だけ決めて設定を作る（CI-PL-G3）。"""
    if "run_id" in kwargs:
        raise ValueError("run_id is generated here; do not pass it")
    stamp = datetime.now(JST).strftime("%Y%m%d-%H%M%S")
    cfg = PlatformRunConfig(run_id=f"{stamp}-{uuid.uuid4().hex[:8]}", **kwargs)
    cfg.validate()
    return cfg
