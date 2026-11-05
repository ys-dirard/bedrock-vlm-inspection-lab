"""単価表と費用の計算、累計費用での停止（docs/contracts.md の CI-C1〜C6）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PER = 1_000_000  # 単価は100万トークンあたりの USD


@dataclass(frozen=True)
class Price:
    model_id: str
    service_tier: str
    input_per_mtok: float | None
    output_per_mtok: float | None
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    source_url: str | None = None
    checked_on: str | None = None

    @property
    def runnable(self) -> bool:
        """入力と出力の単価が分かっているか（CI-C5）。"""
        return self.input_per_mtok is not None and self.output_per_mtok is not None


class PricingTable:
    def __init__(self, prices: list[Price]):
        self._prices = {(p.model_id, p.service_tier): p for p in prices}

    @classmethod
    def load(cls, path: str | Path) -> "PricingTable":
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        fields = Price.__dataclass_fields__
        prices = [Price(**{k: v for k, v in m.items() if k in fields}) for m in doc.get("models", [])]
        return cls(prices)

    def get(self, model_id: str, service_tier: str = "standard") -> Price:
        try:
            return self._prices[(model_id, service_tier)]
        except KeyError:
            raise KeyError(f"pricing.yaml has no entry for model_id={model_id!r}, service_tier={service_tier!r}") from None


def _term(tokens: int, unit_price: float | None) -> float | None:
    if tokens < 0:
        raise ValueError(f"token count must be >= 0, got {tokens}")
    if tokens == 0:
        return 0.0  # CI-C2: 使っていない項目は単価が null でも 0
    if unit_price is None:
        return None
    return tokens * unit_price


def call_cost(
    price: Price,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """1呼び出しの費用（USD）。必要な単価が null なら None（CI-C1〜C3）。"""
    terms = [
        _term(input_tokens, price.input_per_mtok),
        _term(output_tokens, price.output_per_mtok),
        _term(cache_read_tokens, price.cache_read_per_mtok),
        _term(cache_write_tokens, price.cache_write_per_mtok),
    ]
    if any(t is None for t in terms):
        return None
    return sum(terms) / PER  # type: ignore[arg-type]


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    """累計費用の見張り。超える前に止める（CI-C4、CI-C6）。"""

    def __init__(self, budget_usd: float, spent_usd: float = 0.0, max_call_usd: float = 0.0):
        if budget_usd <= 0:
            raise ValueError("budget_usd must be > 0")
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self.max_call_usd = max_call_usd

    @classmethod
    def resume(cls, budget_usd: float, previous_costs: list[float | None]) -> "BudgetGuard":
        known = [c for c in previous_costs if c is not None]
        if len(known) != len(previous_costs):
            raise ValueError("previous records include unknown cost; cannot resume budget tracking")
        return cls(budget_usd, sum(known), max(known, default=0.0))

    def check_before_call(self) -> None:
        if self.spent_usd + self.max_call_usd > self.budget_usd:
            raise BudgetExceeded(
                f"budget {self.budget_usd:.4f} USD would be exceeded: spent {self.spent_usd:.4f} + "
                f"next call up to {self.max_call_usd:.4f}"
            )

    def add(self, cost_usd: float | None) -> None:
        if cost_usd is None:
            raise ValueError("cost is unknown; refuse to continue without a price")
        if cost_usd < 0:
            raise ValueError("cost must be >= 0")
        self.spent_usd += cost_usd
        self.max_call_usd = max(self.max_call_usd, cost_usd)


def summarize_prices(table: PricingTable, keys: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """pricing_used.md 用に、使った単価を出典つきで並べる。"""
    out = []
    for model_id, tier in keys:
        p = table.get(model_id, tier)
        out.append(p.__dict__.copy())
    return out
