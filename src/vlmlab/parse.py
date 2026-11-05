"""応答の JSON を読み取る。救済はしない（docs/contracts.md の CI-P1〜P5）。

JSON として読めなかった応答は「解析失敗」として数える。正規表現で JSON を探し出すことはしない。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

_FENCE_OPEN = "```"


@dataclass(frozen=True)
class ParseSpec:
    """応答の形。設定ファイルの `parse:` から作る。

    AWS サンプルの形（`qc` が OK/NOK）なら judgement_key="qc", ng_values=("NOK",) とする。
    """

    judgement_key: str = "judgement"
    ok_values: tuple[str, ...] = ("OK",)
    ng_values: tuple[str, ...] = ("NG",)
    confidence_key: str | None = "confidence"
    defect_key: str | None = "defect"
    allow_code_fence: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ParseSpec":
        d = dict(d or {})
        for k in ("ok_values", "ng_values"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass(frozen=True)
class ParseResult:
    parsed: dict[str, Any] | None  # 読み取った JSON（読めなければ None）
    pred: str | None  # "OK" / "NG"（読めなければ None）
    confidence: float | None
    defect: str | None
    parse_error: str | None

    @property
    def ok(self) -> bool:
        return self.parse_error is None


def _fail(reason: str, parsed: dict[str, Any] | None = None) -> ParseResult:
    return ParseResult(parsed=parsed, pred=None, confidence=None, defect=None, parse_error=reason)


def _strip_fence(text: str) -> str | None:
    """```json ... ``` の囲みが本文全体を包んでいるときだけ中身を返す。それ以外は None。"""
    if not (text.startswith(_FENCE_OPEN) and text.endswith(_FENCE_OPEN) and len(text) >= 6):
        return None
    first_nl = text.find("\n")
    if first_nl == -1:
        return None
    info = text[3:first_nl].strip()
    if info not in ("", "json", "JSON"):
        return None
    return text[first_nl + 1 : -3].strip()


def parse_object(raw_text: str | None, allow_code_fence: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    """本文全体が JSON のオブジェクトなら (obj, None)、そうでなければ (None, 理由)。CI-P1〜P3。"""
    if raw_text is None:
        return None, "empty: no text"
    text = raw_text.strip()
    if not text:
        return None, "empty: blank text"
    if allow_code_fence and text.startswith(_FENCE_OPEN):
        inner = _strip_fence(text)
        if inner is None:
            return None, "json: malformed code fence"
        text = inner
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"json: {e.msg} (line {e.lineno} col {e.colno})"
    if not isinstance(obj, dict):
        return None, f"json: top level is {type(obj).__name__}, not object"
    return obj, None


def parse_response(raw_text: str | None, spec: ParseSpec | None = None) -> ParseResult:
    """応答を読み取り、判定（OK/NG）と確信度に直す。例外は投げない（CI-P2）。"""
    spec = spec or ParseSpec()
    obj, err = parse_object(raw_text, spec.allow_code_fence)
    if obj is None:
        return _fail(err or "unknown")

    if spec.judgement_key not in obj:
        return _fail(f"schema: missing key '{spec.judgement_key}'", obj)
    value = obj[spec.judgement_key]
    if isinstance(value, str) and value in spec.ok_values:
        pred = "OK"
    elif isinstance(value, str) and value in spec.ng_values:
        pred = "NG"
    else:
        return _fail(f"schema: '{spec.judgement_key}' has unexpected value {value!r}", obj)

    confidence: float | None = None
    if spec.confidence_key and spec.confidence_key in obj:
        c = obj[spec.confidence_key]
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
            return _fail(f"schema: '{spec.confidence_key}' must be a number in [0, 1], got {c!r}", obj)
        confidence = float(c)

    defect: str | None = None
    if spec.defect_key and spec.defect_key in obj:
        d = obj[spec.defect_key]
        defect = d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)

    return ParseResult(parsed=obj, pred=pred, confidence=confidence, defect=defect, parse_error=None)
