"""途中から流し直すときの決まり（docs/contracts.md の CI-AC-R1〜R4）。

2026-11-21 の 12/16 の本番で、メモリ不足で止められたあとに流し直し、終わっていた35枚を呼び直した（重複 0.69 USD）。
その再発を防ぐ。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .runcfg import PlatformRunConfig

EXTRA_KEYS = ("tools_enabled", "prompt_cache", "platform_condition")


def identity_of(cfg: PlatformRunConfig) -> dict[str, Any]:
    """結果を左右する設定（記録にあるキーだけ）。"""
    return {**cfg.identity(), **{k: getattr(cfg, k) for k in EXTRA_KEYS}}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def resume(cfg: PlatformRunConfig, records: list[dict[str, Any]]) -> tuple[PlatformRunConfig, set[tuple[str, int]]]:
    """CI-AC-R1〜R4。前の記録があれば、設定が同じことを確かめ、最後の run_id を引き継ぎ、成功した (image_id, repeat) を返す。"""
    if not records:
        return cfg, set()
    want = identity_of(cfg)
    for r in records:
        got = {k: r.get(k) for k in want}
        if got != want:
            diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
            raise ValueError(f"settings differ from the previous records (CI-AC-R2): {diff}")
        if r.get("run_name") != cfg.run_name:
            raise ValueError(f"run_name differs: {r.get('run_name')!r} != {cfg.run_name!r}")
    done = {(r["image_id"], int(r.get("repeat", 1))) for r in records if r.get("api_error") is None}
    return dataclasses.replace(cfg, run_id=records[-1]["run_id"]), done
