"""不良箇所の位置の評価（12/2 の記事。docs/contracts.md の CI-L1〜L13）。

- 応答の枠（0〜1000 の [x1, y1, x2, y2]）の読み取り
- 0〜1000 の座標 → 画素、2×2 の分割と元の座標への戻し
- マスクの連結成分 → 正解枠、IoU、当たり・中心の当たり、座標のぶれ

画素の座標は連続値で、画素 (i, j) は [i, i+1) × [j, j+1) を占める（CI-L4）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)

# Nova 2 ユーザーガイド（prompting-multimodal.html）の parse_coord_text と同じ正規表現
_GROUP = re.compile(r"\[([^\[\]]*?)\]")
_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")
_NONE = re.compile(r"\bNONE\b")
SCALE = 1000


@dataclass(frozen=True)
class BoxParse:
    boxes: list[Box] | None  # 読めなければ None。枠なしは []
    error: str | None


TRUNCATED = "truncated: stopReason max_tokens"
BOTH = "boxes: both NONE and [..] groups"


def parse_boxes(raw_text: str | None, stop_reason: str | None = None) -> BoxParse:
    """応答から枠を読む（CI-L1、L2、L2b）。例外は投げない。

    stop_reason が max_tokens なら、閉じた [...] があっても解析失敗にする（後ろの枠が失われているため）。
    """
    if stop_reason == "max_tokens":
        return BoxParse(None, TRUNCATED)
    if raw_text is None:
        return BoxParse(None, "empty: no text")
    if not raw_text.strip():
        return BoxParse(None, "empty: blank text")
    groups = _GROUP.findall(raw_text)
    has_none = bool(_NONE.search(raw_text))
    if not groups:
        if has_none:
            return BoxParse([], None)
        return BoxParse(None, "boxes: no [..] group and no NONE")
    if has_none:
        return BoxParse(None, BOTH)
    boxes: list[Box] = []
    for i, g in enumerate(groups):
        parts = [p.strip() for p in g.split(",")]
        if len(parts) != 4 or not all(_NUMBER.match(p) for p in parts):
            return BoxParse(None, f"boxes: group {i} is not 4 numbers: [{g[:40]}]")
        x1, y1, x2, y2 = (float(p) for p in parts)
        if not all(0 <= v <= SCALE for v in (x1, y1, x2, y2)):
            return BoxParse(None, f"boxes: group {i} out of 0..{SCALE}: [{g[:40]}]")
        if not (x1 < x2 and y1 < y2):
            return BoxParse(None, f"boxes: group {i} is not x1<x2 and y1<y2: [{g[:40]}]")
        boxes.append((x1, y1, x2, y2))
    return BoxParse(boxes, None)


def area_fraction(box: Box) -> float:
    """0〜1000 の枠が、その領域（画像全体、またはタイル）に占める面積の割合（CI-L14）。"""
    return (box[2] - box[0]) * (box[3] - box[1]) / (SCALE * SCALE)


def error_kind(error: str | None) -> str | None:
    """解析失敗の理由を、集計で数える種類に分ける。"""
    if error is None:
        return None
    if error == TRUNCATED:
        return "truncated"
    if error == BOTH:
        return "both_none_and_boxes"
    if error.startswith("empty"):
        return "empty"
    return "other"


def norm_to_px(box: Box, region: Box) -> Box:
    """0〜1000 の枠を、region (X0, Y0, X1, Y1) の画素の座標に戻す（CI-L4）。"""
    X0, Y0, X1, Y1 = region
    w, h = X1 - X0, Y1 - Y0
    x1, y1, x2, y2 = box
    return (X0 + x1 / SCALE * w, Y0 + y1 / SCALE * h, X0 + x2 / SCALE * w, Y0 + y2 / SCALE * h)


def tile_regions(width: int, height: int, n: int = 2) -> list[tuple[int, int, int, int]]:
    """n×n の分割の範囲（CI-L5）。行の順（左上、右上、左下、右下）。"""
    if n < 1 or width < n or height < n:
        raise ValueError(f"cannot split {width}x{height} into {n}x{n}")
    xs = [i * width // n for i in range(n)] + [width]
    ys = [j * height // n for j in range(n)] + [height]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(n) for i in range(n)]


def load_mask(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 0


def gt_boxes(mask: np.ndarray, min_box_area: int = 20) -> list[Box]:
    """マスクの 8 近傍の連結成分の外接矩形（CI-L6）。"""
    from scipy import ndimage

    lab, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        if (xs.stop - xs.start) * (ys.stop - ys.start) >= min_box_area:
            out.append((float(xs.start), float(ys.start), float(xs.stop), float(ys.stop)))
    return sorted(out, key=lambda b: (b[1], b[0]))


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def iou(a: Box, b: Box) -> float:
    """CI-L7。"""
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def box_hits_mask(box: Box, mask: np.ndarray) -> bool:
    """枠が覆う画素に異常が1つでもあるか（CI-L8）。"""
    h, w = mask.shape
    x_lo, y_lo = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x_hi, y_hi = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
    if x_lo >= x_hi or y_lo >= y_hi:
        return False
    return bool(mask[y_lo:y_hi, x_lo:x_hi].any())


def center_in_mask(box: Box, mask: np.ndarray) -> bool:
    """枠の中心を含む画素が異常か（CI-L8）。"""
    h, w = mask.shape
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    px = min(max(math.floor(cx), 0), w - 1)
    py = min(max(math.floor(cy), 0), h - 1)
    return bool(mask[py, px])


@dataclass(frozen=True)
class ImageScore:
    hit: bool
    center_hit: bool
    n_pred: int
    n_pred_hit: int
    n_pred_center_hit: int
    gt_best_iou: list[float]  # 正解枠ごとの最良 IoU


def score_anomaly(pred: Sequence[Box], mask: np.ndarray, gts: Sequence[Box]) -> ImageScore:
    """不良画像1枚・1回の採点（CI-L9）。pred は画素の座標。"""
    hits = [box_hits_mask(b, mask) for b in pred]
    centers = [center_in_mask(b, mask) for b in pred]
    best = [max((iou(g, p) for p in pred), default=0.0) for g in gts]
    return ImageScore(any(hits), any(centers), len(pred), sum(hits), sum(centers), best)


def split_gt_count(gts: Sequence[Box], pred_by_tile: Sequence[tuple[int, Box]]) -> int:
    """2つ以上の異なるタイルの予測枠と重なる正解枠の数（CI-L12）。"""
    n = 0
    for g in gts:
        tiles = {t for t, p in pred_by_tile if intersection(g, p) > 0}
        if len(tiles) >= 2:
            n += 1
    return n


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def set_iou(a: Sequence[Box], b: Sequence[Box]) -> float | None:
    """枠の集合どうしの対称な最良一致 IoU（CI-L10）。両方とも空なら None。"""
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    ab = _mean([max(iou(x, y) for y in b) for x in a])
    ba = _mean([max(iou(y, x) for x in a) for y in b])
    return (ab + ba) / 2


@dataclass(frozen=True)
class Jitter:
    pair_iou_mean: float | None  # 有効な組の平均（両方枠なしの組は除く）
    n_pairs: int  # 両方の回が読めた組
    n_pairs_both_empty: int


def jitter(repeats: Sequence[Sequence[Box] | None]) -> Jitter:
    """同じ画像の各回の枠（読めなかった回は None）から、座標のぶれを出す（CI-L10）。"""
    vals: list[float] = []
    n_pairs = n_empty = 0
    for a, b in combinations(repeats, 2):
        if a is None or b is None:
            continue
        n_pairs += 1
        v = set_iou(a, b)
        if v is None:
            n_empty += 1
        else:
            vals.append(v)
    return Jitter(_mean(vals) if vals else None, n_pairs, n_empty)
