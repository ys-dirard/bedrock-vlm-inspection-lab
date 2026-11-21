"""人に回すかどうかを決める（docs/contracts.md の CI-HR1〜HR5）。AWS は呼ばない。

回し方（policy）は設定（Lambda の環境変数 ROUTE_POLICY の JSON）で切り替える。
- {"kind": "confidence_band", "lo": 0.4, "hi": 0.8}: 不良らしさが [lo, hi) なら人へ（12/12 の CI-CB7 と同じ）
- {"kind": "sample", "rate": 0.5, "seed": "ac14"}: NG と解析失敗は人へ、OK は rate の割合を人へ（確信度を使わない）
- {"kind": "all"}: すべて人へ
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

HUMAN = "human"
AUTO = "auto"
_KINDS = ("confidence_band", "sample", "all")


@dataclass(frozen=True)
class RoutePolicy:
    kind: str
    lo: float | None = None
    hi: float | None = None
    rate: float | None = None
    seed: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RoutePolicy":
        """CI-HR4：形が違えば ValueError。"""
        if not isinstance(d, dict):
            raise ValueError("route policy must be a JSON object")
        kind = d.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"unknown route policy kind {kind!r}; expected one of {_KINDS}")
        allowed = {"confidence_band": {"kind", "lo", "hi"}, "sample": {"kind", "rate", "seed"}, "all": {"kind"}}[kind]
        extra = set(d) - allowed
        if extra:
            raise ValueError(f"unexpected keys for {kind}: {sorted(extra)}")
        if kind == "confidence_band":
            lo, hi = _num(d.get("lo"), "lo"), _num(d.get("hi"), "hi")
            if not (0 <= lo <= hi <= 1):
                raise ValueError("confidence_band requires 0 <= lo <= hi <= 1")
            return cls(kind, lo=lo, hi=hi)
        if kind == "sample":
            rate = _num(d.get("rate"), "rate")
            if not (0 <= rate <= 1):
                raise ValueError("sample requires 0 <= rate <= 1")
            seed = d.get("seed", "")
            if not isinstance(seed, str):
                raise ValueError("seed must be a string")
            return cls(kind, rate=rate, seed=seed)
        return cls(kind)

    @classmethod
    def from_json(cls, text: str) -> "RoutePolicy":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class RouteDecision:
    route: str  # "auto" / "human"
    auto_judgement: str | None  # route が auto のときの確定の判定（OK / NG）。human なら None
    reason: str
    anomaly_score: float | None


def _num(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return float(v)


def anomaly_score(judgement: str | None, confidence: float | None) -> float | None:
    """CI-HR1：NG なら c、OK なら 1 − c を小数9桁に丸める。判定か確信度が無ければ None。"""
    if judgement not in ("OK", "NG") or confidence is None:
        return None
    c = float(confidence)
    return round(c if judgement == "NG" else 1.0 - c, 9)


def sample_unit(seed: str, image_id: str) -> float:
    """CI-HR3：seed と画像 ID だけで決まる [0, 1) の値（SHA-256 の先頭8バイト / 2^64）。"""
    h = hashlib.sha256(f"{seed}\x00{image_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def decide(policy: RoutePolicy, image_id: str, judgement: str | None, confidence: float | None) -> RouteDecision:
    """CI-HR2・HR3・HR5。judgement が None は解析失敗（API のエラーはここに来ない）。"""
    score = anomaly_score(judgement, confidence)
    if policy.kind == "all":
        return RouteDecision(HUMAN, None, "policy_all", score)
    if judgement not in ("OK", "NG"):
        return RouteDecision(HUMAN, None, "parse_failed", score)
    if policy.kind == "confidence_band":
        if score is None:
            return RouteDecision(HUMAN, None, "no_confidence", score)
        if score < round(policy.lo, 9):
            return RouteDecision(AUTO, "OK", "below_band", score)
        if score >= round(policy.hi, 9):
            return RouteDecision(AUTO, "NG", "above_band", score)
        return RouteDecision(HUMAN, None, "in_band", score)
    # sample
    if judgement == "NG":
        return RouteDecision(HUMAN, None, "judged_ng", score)
    if sample_unit(policy.seed, image_id) < policy.rate:
        return RouteDecision(HUMAN, None, "sampled_ok", score)
    return RouteDecision(AUTO, "OK", "not_sampled_ok", score)
