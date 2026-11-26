"""画像1枚分の入力トークン（実測）から、画像1,000枚あたりの「画像の分の入力費用」を出す。API は呼ばない。

目的: 記事の「縮めると安くなるか」を、実測のトークン × 呼び方ごとの東京の入力単価で示す
入力: results/token_check_summary.csv（画像あり − 画像なし）、token_check.PRICES（results/pricing_used.md・cheatsheet.csv の値）
出力: results/image_cost_per1000.csv
含めないもの: テキスト（プロンプト）と出力の費用
使い方: python src/image_cost_per1000.py（記事フォルダで）
"""

from __future__ import annotations

import csv
from pathlib import Path

from token_check import PRICES

RES = Path(__file__).resolve().parent.parent / "results"


def image_cost_per1000(image_tokens: int, input_per_mtok: float) -> float:
    """画像1枚の入力トークンと入力単価（USD / 100万トークン）から、1,000枚あたりの費用（USD）を出す。"""
    if image_tokens < 0 or input_per_mtok < 0:
        raise ValueError("tokens and price must be non-negative")
    return image_tokens * input_per_mtok * 1000 / 1_000_000


def build(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_model: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["variant"] in ("original", "long768") and r["image_tokens_measured"]:
            by_model.setdefault(r["model_id"], {})[r["variant"]] = int(r["image_tokens_measured"])
    out = []
    for m, v in by_model.items():
        if set(v) != {"original", "long768"}:
            continue
        price = PRICES[m][0]
        co, cl = image_cost_per1000(v["original"], price), image_cost_per1000(v["long768"], price)
        out.append({"model_id": m, "input_usd_per_mtok": price,
                    "image_tokens_original": v["original"], "image_tokens_long768": v["long768"],
                    "usd_per1000_original": round(co, 3), "usd_per1000_long768": round(cl, 3),
                    "long768_over_original": round(cl / co, 2) if co else ""})
    return out


def main() -> None:
    with (RES / "token_check_summary.csv").open(encoding="utf-8-sig") as f:
        out = build(list(csv.DictReader(f)))
    with (RES / "image_cost_per1000.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"{len(out)} models -> results/image_cost_per1000.csv")


if __name__ == "__main__":
    main()
