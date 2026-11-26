"""12/11 の記事の図を2枚作る。

- results/cost_by_article.png: 記事ごとの費用の記録（cost_by_article.csv の cost_usd を記事ごとに足す）と、上限 6 USD の線
- results/bill_vs_records.png: 11月の請求 20.37 USD（Cost Explorer、2026-11-20 時点）の内訳と、予算 20 USD の線
  値は results/cost_by_article.md の「記録との突き合わせ」の表から写す（下の BILL）

使い方（記事フォルダで）: uv run --no-project --with matplotlib python src/plot_costs.py
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"

# dataviz の参照パレット（ライト）
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAYFILL = "#b9b8b2"
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"

LIMIT_PER_ARTICLE = 6.0
BUDGET_MONTH = 20.0

# cost_by_article.md「記録との突き合わせ」の表（USD）。合計 20.37
BILL = [
    ("記事の検証の記録（11/16 までの実行）", 14.09, BLUE),
    ("LT のエージェント（Haiku 4.5）", 2.47, ORANGE),
    ("Haiku 4.5（何の分か未確認）", 0.72, YELLOW),
    ("使っていない公開 IPv4 アドレス", 1.09, MAGENTA),
    ("そのほか", 0.14, GRAYFILL),
    ("税", 1.86, AQUA),
]


def set_font() -> None:
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return
    raise SystemExit("日本語フォントが見つからない")


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def per_article() -> "OrderedDict[str, float]":
    totals: OrderedDict[str, float] = OrderedDict()
    with (RES / "cost_by_article.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["article_date"].replace("-", "/").lstrip("0").replace("/0", "/")
            totals[key] = totals.get(key, 0.0) + float(row["cost_usd"])
    return totals


def plot_per_article() -> None:
    totals = per_article()
    # 12/10 までの記事だけ（12/12 は参考なので除く）。12/7 は記録が無い
    items = [(k, v) for k, v in totals.items() if k != "12/12"]
    labels = [k + ("（記録なし）" if k == "12/7" else "") for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    y = list(range(len(items)))[::-1]
    ax.barh(y, values, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        if v > 5.0:  # 上限の線と重ならないように棒の中に書く
            ax.text(v - 0.08, yi, f"{v:.2f}", va="center", ha="right", color="#ffffff", fontsize=9)
        else:
            ax.text(v + 0.08, yi, f"{v:.2f}", va="center", ha="left", color=INK, fontsize=9)
    ax.axvline(LIMIT_PER_ARTICLE, color=INK2, lw=1.5, ls="--")
    ax.text(LIMIT_PER_ARTICLE - 0.05, len(items) - 0.4, "1記事の上限 6 USD", ha="right", va="bottom",
            color=INK2, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_xlim(0, 7)
    ax.set_ylim(-0.6, len(items) + 0.2)
    ax.set_xlabel("記録の費用（USD、税抜き）", color=INK2)
    ax.set_title(f"記事ごとの費用の記録（11/11〜12/10 の記事、合計 {total:.2f} USD）", color=INK, loc="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "cost_by_article.png", facecolor=SURFACE)
    plt.close(fig)


def plot_bill() -> None:
    total = sum(v for _, v, _ in BILL)
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    left = 0.0
    for label, v, color in BILL:
        ax.barh(0, v, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2, label=f"{label} {v:.2f}")
        if v >= 1.0:
            ax.text(left + v / 2, 0, f"{v:.2f}", ha="center", va="center",
                    color="#ffffff" if color in (BLUE,) else INK, fontsize=9)
        left += v
    ax.axvline(BUDGET_MONTH, color=INK, lw=1.5, ls="--")
    ax.text(BUDGET_MONTH - 0.1, 0.33, "月の予算 20 USD", ha="right", va="bottom", color=INK, fontsize=9)
    ax.text(total + 0.1, 0, f"{total:.2f}", ha="left", va="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 22.5)
    ax.set_ylim(-0.5, 0.6)
    ax.set_xlabel("USD", color=INK2)
    ax.set_title("11月の請求（Cost Explorer、2026-11-20 時点、税込み）の内訳", color=INK, loc="left", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False, fontsize=9,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RES / "bill_vs_records.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    set_font()
    plot_per_article()
    plot_bill()
    print("wrote", RES / "cost_by_article.png", RES / "bill_vs_records.png")
