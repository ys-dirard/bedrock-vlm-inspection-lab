"""請求書の画像を描く・劣化させる・上下に分ける（docs/contracts.md の CI-IR1〜IR4）。

A4（1240×1754 ピクセル、150dpi 相当）。フォントは Noto Sans JP（SIL Open Font License 1.1）。
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH, HEIGHT = 1240, 1754
MARGIN = 90
FOOTER = "この請求書は検証用に作成した架空のものです。実在の事業者とは関係ありません。"

# フォントを探す順（--font で明示しなければ最初に見つかったもの）
FONT_CANDIDATES = [
    "data/fonts/NotoSansJP-VF.ttf",
    "data/fonts/NotoSansJP-Regular.ttf",
    "C:/Windows/Fonts/NotoSansJP-VF.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def find_font(explicit: str | None = None, base_dir: Path | None = None) -> Path:
    cands = [explicit] if explicit else FONT_CANDIDATES
    for c in cands:
        p = Path(c)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        if p.exists():
            return p
    raise FileNotFoundError(f"Japanese font not found; tried {cands}. Pass --font (Noto Sans JP, OFL 1.1)")


class Fonts:
    def __init__(self, path: Path):
        self.path = path
        self._cache: dict[tuple[int, int], ImageFont.FreeTypeFont] = {}

    def get(self, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
        key = (size, weight)
        if key not in self._cache:
            f = ImageFont.truetype(str(self.path), size)
            try:  # 可変フォントなら太さを指定する。可変でなければそのまま
                f.set_variation_by_axes([weight])
            except (OSError, ValueError):
                pass
            self._cache[key] = f
        return self._cache[key]


def yen(v: int) -> str:
    return f"¥{v:,}"


def format_date(iso: str, style: str) -> str:
    """seireki: 2026年3月14日 / wareki: 令和8年3月14日（CI-IR1。令和元年 = 2019年）。"""
    y, m, d = (int(x) for x in iso.split("-"))
    if style == "seireki":
        return f"{y}年{m}月{d}日"
    if style == "wareki":
        if y < 2019:
            raise ValueError("wareki formatting supports Reiwa (2019-) only")
        ry = y - 2018
        return f"令和{'元' if ry == 1 else ry}年{m}月{d}日"
    raise ValueError(f"unknown date style {style!r}")


def render(truth: dict[str, Any], fonts: Fonts) -> Image.Image:
    """正解の値をそのまま描く（CI-IR2）。軽減税率の明細の品名には「※」を付ける。"""
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    dr = ImageDraw.Draw(img)
    f_title, f_big, f_mid, f_body, f_small = fonts.get(56, 700), fonts.get(34, 700), fonts.get(26), fonts.get(23), fonts.get(19)
    black, gray = (20, 20, 20), (110, 110, 110)

    dr.text((WIDTH // 2, 80), "請 求 書", font=f_title, fill=black, anchor="mt")
    # 右上：請求番号と発行日
    right = WIDTH - MARGIN
    dr.text((right, 180), f"請求番号：{truth['invoice_number']}", font=f_body, fill=black, anchor="ra")
    dr.text((right, 215), f"発行日：{format_date(truth['issue_date'], truth['date_style'])}", font=f_body, fill=black, anchor="ra")
    # 左：宛先
    dr.text((MARGIN, 270), f"{truth['recipient_name']} 御中", font=f_big, fill=black)
    dr.line((MARGIN, 318, MARGIN + 560, 318), fill=black, width=2)
    dr.text((MARGIN, 345), "下記のとおりご請求申し上げます。", font=f_body, fill=black)
    # 右：発行元
    y = 270
    dr.text((right, y), truth["issuer_name"], font=f_mid, fill=black, anchor="ra")
    dr.text((right, y + 40), truth["issuer_address"], font=f_small, fill=black, anchor="ra")
    dr.text((right, y + 72), f"登録番号：{truth['registration_number']}", font=f_body, fill=black, anchor="ra")
    # ご請求金額
    dr.rectangle((MARGIN, 420, MARGIN + 620, 490), outline=black, width=2)
    dr.text((MARGIN + 20, 455), "ご請求金額（税込）", font=f_mid, fill=black, anchor="lm")
    dr.text((MARGIN + 600, 455), yen(truth["total"]), font=f_big, fill=black, anchor="rm")

    # 明細の表
    cols = [MARGIN, MARGIN + 560, MARGIN + 680, MARGIN + 870, right]  # 品名 | 数量 | 単価 | 金額
    top, row_h = 540, 48
    dr.rectangle((cols[0], top, cols[-1], top + row_h), fill=(235, 235, 235), outline=black, width=1)
    for x0, x1, label in zip(cols[:-1], cols[1:], ["品名", "数量", "単価", "金額"]):
        dr.text(((x0 + x1) // 2, top + row_h // 2), label, font=f_body, fill=black, anchor="mm")
    rows = truth["line_items"]
    n_rows = max(len(rows), 8)
    for r in range(n_rows):
        y0 = top + row_h * (r + 1)
        dr.rectangle((cols[0], y0, cols[-1], y0 + row_h), outline=black, width=1)
        if r < len(rows):
            it = rows[r]
            desc = it["description"] + ("　※" if it["reduced_rate"] else "")
            cy = y0 + row_h // 2
            dr.text((cols[0] + 14, cy), desc, font=f_body, fill=black, anchor="lm")
            dr.text((cols[2] - 14, cy), f"{it['quantity']:,}", font=f_body, fill=black, anchor="rm")
            dr.text((cols[3] - 14, cy), f"{it['unit_price']:,}", font=f_body, fill=black, anchor="rm")
            dr.text((cols[4] - 14, cy), f"{it['amount']:,}", font=f_body, fill=black, anchor="rm")
    for x in cols[1:-1]:
        dr.line((x, top, x, top + row_h * (n_rows + 1)), fill=black, width=1)
    y = top + row_h * (n_rows + 1) + 14
    dr.text((MARGIN, y), "※は軽減税率（8%）対象", font=f_small, fill=black)

    # 税率ごとの合計
    labels = [
        ("10%対象 小計", truth["subtotal_10"]), ("10%対象 消費税", truth["tax_10"]),
        ("8%対象 小計", truth["subtotal_8"]), ("8%対象 消費税", truth["tax_8"]),
        ("合計（税込）", truth["total"]),
    ]
    x0, x1 = MARGIN + 560, right
    y = top + row_h * (n_rows + 1) + 60
    for k, (label, v) in enumerate(labels):
        last = k == len(labels) - 1
        dr.rectangle((x0, y, x1, y + row_h), outline=black, width=2 if last else 1,
                     fill=(245, 245, 245) if last else None)
        dr.text((x0 + 14, y + row_h // 2), label, font=f_body, fill=black, anchor="lm")
        dr.text((x1 - 14, y + row_h // 2), yen(v), font=f_mid if last else f_body, fill=black, anchor="rm")
        y += row_h

    dr.text((MARGIN, y + 40), "お支払期限：発行日の翌月末日", font=f_small, fill=black)
    dr.text((WIDTH // 2, HEIGHT - 60), FOOTER, font=f_small, fill=gray, anchor="mm")
    return img


def degrade(img: Image.Image, rng: random.Random) -> tuple[bytes, dict[str, Any]]:
    """スキャンを模した劣化（CI-IR3）：±2度の回転 → ぼかし（半径0.8）→ 横900px に縮めて元に戻す → JPEG 品質40。

    スキャンや FAX を厳密に再現するものではない。返り値は JPEG のバイト列と、使った角度。
    """
    angle = round(rng.uniform(-2.0, 2.0), 3)
    out = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor="white")
    out = out.filter(ImageFilter.GaussianBlur(radius=0.8))
    w, h = out.size
    small = out.resize((900, round(h * 900 / w)), Image.BILINEAR)
    out = small.resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=40)
    return buf.getvalue(), {"angle_deg": angle, "blur_radius": 0.8, "downscale_width": 900, "jpeg_quality": 40}


def split_halves(img: Image.Image, overlap: int = 100) -> tuple[Image.Image, Image.Image]:
    """上半分と下半分（重なり overlap ピクセル）に分ける（CI-IR4）。"""
    w, h = img.size
    if overlap < 0 or overlap >= h:
        raise ValueError("overlap must be in [0, height)")
    mid = h // 2
    top = img.crop((0, 0, w, mid + overlap // 2 + overlap % 2))
    bottom = img.crop((0, mid - overlap // 2, w, h))
    return top, bottom


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
