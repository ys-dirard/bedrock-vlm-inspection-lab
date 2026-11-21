"""12/18：評価に渡すスパンから画像を除き、除けたことを確かめる（docs/contracts.md の CI-AC-S1〜S4）。

11/30 の本番で、Runtime のログの約95%が ADOT の gen_ai.user.message（送った画像の Base64）だった。
画像が文字のまま評価器に入ると、1回の入力が上限の20万トークンに近づく（記事の results/estimate.md）。
on-demand の評価では、スパンをこちらで渡すので、渡す前に画像を除く。
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

PLACEHOLDER = "[image removed]"
MIN_B64 = 200  # これより短い Base64 らしい文字列は画像とみなさない（ID やハッシュを消さないため）
_B64CH = r"(?:[A-Za-z0-9+=]|\\?/)"  # JSON の中で / が \/ になっていても拾う
# 画像の形式ごとの Base64 の先頭（JPEG /9j/、PNG iVBORw0KGgo、GIF R0lGOD、WEBP UklGR）
_IMAGE_B64 = re.compile(rf"(?:\\?/9j\\?/|iVBORw0KGgo|R0lGOD|UklGR){_B64CH}{{{MIN_B64},}}")
_IMAGE_KEYS = ("bytes",)


def _strip_str(s: str) -> tuple[str, int]:
    out, n = _IMAGE_B64.subn(PLACEHOLDER, s)
    return out, n


def _walk(obj: Any, counter: list[int]) -> Any:
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if k in _IMAGE_KEYS and isinstance(v, str) and len(v) >= MIN_B64:
                new[k] = PLACEHOLDER  # CI-AC-S1：image ブロックの bytes は形式によらず除く
                counter[0] += 1
            else:
                new[k] = _walk(v, counter)
        return new
    if isinstance(obj, list):
        return [_walk(v, counter) for v in obj]
    if isinstance(obj, str):
        s, n = _strip_str(obj)
        counter[0] += n
        return s
    if isinstance(obj, (bytes, bytearray)):
        counter[0] += 1
        return PLACEHOLDER
    return obj


def strip_images(spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """CI-AC-S1・S2。画像の Base64 を PLACEHOLDER に置き換えた写しと、置き換えた数を返す。元のスパンは変えない。"""
    counter = [0]
    out = _walk(copy.deepcopy(spans), counter)
    return out, counter[0]


def find_images(obj: Any, path: str = "$") -> list[str]:
    """CI-AC-S3。画像らしい中身が残っている場所（JSON のパス）の一覧。"""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if k in _IMAGE_KEYS and isinstance(v, str) and len(v) >= MIN_B64:
                found.append(p)
            else:
                found += find_images(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += find_images(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if _IMAGE_B64.search(obj):
            found.append(path)
    elif isinstance(obj, (bytes, bytearray)):
        found.append(path)
    return found


def assert_text_only(spans: list[dict[str, Any]]) -> int:
    """CI-AC-S3。画像が残っていれば ValueError（評価を呼ばない）。渡す JSON の大きさ（バイト）を返す。"""
    left = find_images(spans)
    if left:
        raise ValueError(f"image-like content remains at {left[:5]} ({len(left)} places)")
    return len(json.dumps(spans, ensure_ascii=False).encode("utf-8"))


STRANDS_SCOPE = "strands.telemetry.tracer"


def missing_events(spans: list[dict[str, Any]]) -> list[str]:
    """CI-AC-S7。Strands のスパン（chat・execute_tool・execute_event_loop_cycle・invoke_agent）のうち、
    同じ spanId のイベント（本文 body のある記録）が無いものの名前。1つでもあれば評価は
    「Session span data is incomplete」で失敗する（2026-11-22 の 12/18 の本番で、29件中10件）。"""
    with_events = {s.get("spanId") for s in spans if "body" in s}
    return [s["name"] for s in spans
            if "name" in s and (s.get("scope") or {}).get("name") == STRANDS_SCOPE
            and s.get("spanId") not in with_events]
