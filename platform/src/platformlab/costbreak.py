"""サービスごとの費用（使用量 × 単価）。docs/contracts.md の CI-PL-C1〜C5。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# AgentCore Runtime の単価（東京、facts.md の F9、2026-11-20 確認）
RUNTIME_PRICES = {
    "V1": {"vcpu_hour": 0.0895, "gb_hour": 0.00945},
    "V2": {"vcpu_hour": 0.1276, "gb_hour": 0.0169},
}


@dataclass(frozen=True)
class CostLine:
    service: str
    usage_type: str
    quantity: float
    unit: str
    unit_price: float | None
    source: str  # measured | estimated
    basis: str = ""

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(f"quantity must be >= 0: {self.service}/{self.usage_type}={self.quantity}")  # CI-PL-C3
        if self.source not in ("measured", "estimated"):
            raise ValueError(f"source must be measured or estimated: {self.source!r}")

    @property
    def cost_usd(self) -> float | None:
        if self.quantity == 0:
            return 0.0  # CI-PL-C2
        if self.unit_price is None:
            return None
        return self.quantity * self.unit_price  # CI-PL-C1

    def to_row(self) -> dict[str, Any]:
        return {**asdict(self), "cost_usd": self.cost_usd}


def runtime_lines(name: str, vcpu_hours: float, gb_hours: float, platform_version: str,
                  source: str = "measured", basis: str = "") -> list[CostLine]:
    """Runtime の CPU とメモリを別の行に（CI-PL-C5）。"""
    if platform_version not in RUNTIME_PRICES:
        raise ValueError(f"unknown platform version: {platform_version!r}")
    p = RUNTIME_PRICES[platform_version]
    return [
        CostLine(f"AgentCore Runtime ({name})", f"vCPU-{platform_version}", vcpu_hours, "vCPU-hour",
                 p["vcpu_hour"], source, basis),
        CostLine(f"AgentCore Runtime ({name})", f"Memory-{platform_version}", gb_hours, "GB-hour",
                 p["gb_hour"], source, basis),
    ]


def total(lines: list[CostLine]) -> float | None:
    costs = [ln.cost_usd for ln in lines]
    if any(c is None for c in costs):
        return None  # CI-PL-C4
    return sum(costs)  # type: ignore[arg-type]


def per_1000(lines: list[CostLine], n_images: int) -> float | None:
    t = total(lines)
    if t is None or n_images <= 0:
        return None
    return t / n_images * 1000
