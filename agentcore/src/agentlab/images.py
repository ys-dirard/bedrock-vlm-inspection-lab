"""画像の読み込みと縮小。11月の vlmlab.data.load_image と同じ規則（テストで突き合わせる）。

vlmlab.data は pandas などに依存するため、エージェントの ZIP には入れず、同じ処理をここに置く。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

_FORMATS = {"JPEG": "jpeg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}


@dataclass(frozen=True)
class EncodedImage:
    data: bytes
    format: str
    width: int
    height: int


def encode_image(raw: bytes, long_edge: int | None = None) -> EncodedImage:
    """long_edge が None なら原寸のバイト列をそのまま。縮小するときは長辺をそろえ、JPEG（quality=95）で符号化し直す。"""
    with Image.open(io.BytesIO(raw)) as im:
        fmt = _FORMATS.get(im.format or "")
        if fmt is None:
            raise ValueError(f"unsupported image format {im.format}")
        w, h = im.size
        if long_edge is None or max(w, h) <= long_edge:
            return EncodedImage(raw, fmt, w, h)
        scale = long_edge / max(w, h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=95)
    return EncodedImage(buf.getvalue(), "jpeg", size[0], size[1])
