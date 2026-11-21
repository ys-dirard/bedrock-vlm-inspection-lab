"""画像を外に出さずに判定する（12/13 の記事）。docs/contracts.md の CI-E1〜E8。

手元の AnomalyDINO のパッチの距離の地図（anomalydino.py の save_maps）から、欠陥の候補の領域を計測し、
区分の言葉に直す。言葉の区切りは、学習用の良品（テストと見本に重ならない）だけで決める。
ここで作った文だけを、テキストのモデル（Nova Micro）に渡す（run.py の text_only）。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anomalydino import load_map, map_filename

GRANULARITIES = ("coarse", "fine")

# 位置（9分割）の呼び方。行（上・中・下）× 列（左・中央・右）
POSITION_NAMES = (("左上", "上", "右上"), ("左", "中央", "右"), ("左下", "下", "右下"))

# 大きさの区分（製品の前景の面積に対する割合）と形の区分。テストを見る前に決めた固定値（CI-E4）
SIZE_BINS = ((0.005, "小さい"), (0.02, "中くらい"), (math.inf, "大きい"))
ELONGATED_ASPECT = 2.0


# ---------------------------------------------------------------- 学習用の良品（CI-E1）

def make_calib_split(knn_train: pd.DataFrame, refs: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """CI-E1：区分を決める学習用の良品。visa_knn_train の train の良品から、見本（全 rank）を除く。"""
    c = knn_train[(knn_train["label"] == "normal") & (knn_train["split"] == "train")]
    c = c[~c["image_id"].isin(set(refs["image_id"]))]
    check_calib(c, refs, test)
    return c.sort_values(["category", "image_id"]).reset_index(drop=True)


def check_calib(calib: pd.DataFrame, refs: pd.DataFrame, test: pd.DataFrame) -> None:
    """CI-E1：学習用の良品がすべて train の良品で、テストとも見本とも重ならないこと。"""
    if calib.empty:
        raise ValueError("calibration set is empty")
    if (calib["label"] != "normal").any() or (calib["split"] != "train").any():
        raise ValueError("calibration images must be train normals")
    for name, other in (("test", test), ("refs", refs)):
        overlap = set(calib["image_id"]) & set(other["image_id"])
        if overlap:
            raise ValueError(f"calibration overlaps with {name}: {sorted(overlap)[:5]}")
    missing = set(test["category"]) - set(calib["category"])
    if missing:
        raise ValueError(f"no calibration images for categories {sorted(missing)}")


# ---------------------------------------------------------------- 計測（CI-E2、E3）

@dataclass(frozen=True)
class Calibration:
    category: str
    n_images: int
    patch_quantile: float
    patch_threshold: float          # 領域とみなすパッチの距離のしきい値
    scores: tuple[float, ...]       # 学習用の良品の画像の異常度（昇順）

    @property
    def score_q90(self) -> float:
        return float(np.quantile(self.scores, 0.90))

    @property
    def score_max(self) -> float:
        return float(self.scores[-1])


def calibrate(category: str, maps: list[tuple[np.ndarray, np.ndarray]], scores: list[float],
              patch_quantile: float) -> Calibration:
    """CI-E2：学習用の良品の、前景のパッチの距離をまとめた分位点を、領域のしきい値にする。"""
    if not maps or len(maps) != len(scores):
        raise ValueError("need one map per calibration score")
    if not 0 < patch_quantile < 1:
        raise ValueError("patch_quantile must be in (0, 1)")
    fg = np.concatenate([d[m] for d, m in maps])
    if fg.size == 0:
        raise ValueError(f"{category}: no foreground patches in calibration maps")
    if not np.isfinite(fg).all() or not np.isfinite(scores).all():
        raise ValueError(f"{category}: non-finite calibration values")
    return Calibration(category, len(maps), patch_quantile, float(np.quantile(fg, patch_quantile)),
                       tuple(sorted(float(s) for s in scores)))


@dataclass(frozen=True)
class Measurement:
    n_foreground: int
    n_regions: int
    total_area_frac: float | None   # 領域の合計 / 前景のパッチ数
    largest_area: int               # 最大の領域のパッチ数（0 なら領域なし）
    largest_area_frac: float | None
    largest_row: int | None         # 9分割の行（0 上、1 中、2 下）
    largest_col: int | None         # 9分割の列（0 左、1 中央、2 右）
    largest_aspect: float | None    # 外接矩形の長辺 / 短辺


def measure(dist: np.ndarray, mask: np.ndarray, threshold: float) -> Measurement:
    """CI-E3：前景で距離 ≥ threshold のパッチを 8 近傍でつなぎ、領域の数・大きさ・位置・形を測る。"""
    import cv2

    dist = np.asarray(dist, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if dist.ndim != 2 or dist.shape != mask.shape:
        raise ValueError(f"map shape {dist.shape} / mask {mask.shape}")
    n_fg = int(mask.sum())
    binary = ((dist >= threshold) & mask).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    n_regions = n - 1
    if n_regions == 0:
        return Measurement(n_fg, 0, 0.0 if n_fg else None, 0, 0.0 if n_fg else None, None, None, None)
    areas = stats[1:, cv2.CC_STAT_AREA]
    # 最大の領域。同じ面積なら、距離の最大値が大きいほう、さらに同じなら番号の小さいほう
    peaks = [float(dist[labels == i].max()) for i in range(1, n)]
    order = sorted(range(n_regions), key=lambda i: (-int(areas[i]), -peaks[i], i))
    k = order[0] + 1
    h, w = dist.shape
    cx, cy = centroids[k]
    bw, bh = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
    return Measurement(
        n_foreground=n_fg,
        n_regions=int(n_regions),
        total_area_frac=float(areas.sum()) / n_fg,
        largest_area=int(areas[k - 1]),
        largest_area_frac=float(areas[k - 1]) / n_fg,
        largest_row=min(int(cy // (h / 3)), 2),
        largest_col=min(int(cx // (w / 3)), 2),
        largest_aspect=max(bw, bh) / min(bw, bh),
    )


# ---------------------------------------------------------------- 言葉に直す（CI-E4）

def score_level3(score: float, cal: Calibration) -> str:
    if score > cal.score_max:
        return "学習用の良品のどれよりも高い"
    if score > cal.score_q90:
        return "学習用の良品の上位10%に入る高さ"
    return "学習用の良品と同じくらい"


def score_rank_text(score: float, cal: Calibration) -> str:
    """学習用の良品のうち、異常度が score 以上だった割合を「上位 x%」で言う（切り上げの整数）。"""
    if score > cal.score_max:
        ratio = score / cal.score_max if cal.score_max > 0 else math.inf
        if ratio >= 1.5:
            how = "大きく上回る"
        elif ratio >= 1.2:
            how = "上回る"
        else:
            how = "わずかに上回る"
        return f"学習用の良品{cal.n_images}枚のどれよりも高い（最も高い良品を{how}）"
    n_ge = sum(1 for s in cal.scores if s >= score)
    pct = math.ceil(100 * n_ge / cal.n_images)
    return f"学習用の良品{cal.n_images}枚の中で上位{pct}%に入る高さ"


def size_text(frac: float) -> str:
    for upper, name in SIZE_BINS:
        if frac < upper:
            return name
    raise AssertionError("unreachable")


def shape_text(m: Measurement) -> str:
    if m.largest_area == 1:
        return "点状"
    return "細長い" if (m.largest_aspect or 1.0) >= ELONGATED_ASPECT else "かたまり状"


def verbalize(score: float, m: Measurement, cal: Calibration, granularity: str) -> str:
    """CI-E4：計測値を区分の言葉に直す。数値（距離・面積・座標）そのものは書かない。"""
    if granularity == "coarse":
        region = "あり" if m.n_regions > 0 else "なし"
        return f"異常度: {score_level3(score, cal)}\n良品と違って見える領域: {region}"
    if granularity == "fine":
        lines = [f"異常度: {score_rank_text(score, cal)}"]
        if m.n_regions == 0:
            lines.append("良品と違って見える領域: なし")
        else:
            pos = POSITION_NAMES[m.largest_row][m.largest_col]
            lines.append(f"良品と違って見える領域: {m.n_regions}か所")
            lines.append(f"最も大きい領域: 位置は製品の画像の{pos}、大きさは{size_text(m.largest_area_frac)}"
                         f"（製品の面積に対して）、形は{shape_text(m)}")
        return "\n".join(lines)
    raise ValueError(f"granularity must be one of {GRANULARITIES}")


# ---------------------------------------------------------------- 表にまとめる（CI-E5）

def load_scores_with_maps(run_dir: str | Path) -> pd.DataFrame:
    """anomalydino の記録（scores.jsonl）と地図の置き場所。1枚1行、地図が無ければ止める。"""
    from .run import load_jsonl

    run_dir = Path(run_dir)
    df = pd.DataFrame(load_jsonl(run_dir / "scores.jsonl"))
    if df.empty:
        raise ValueError(f"no scores in {run_dir}")
    if df["image_id"].duplicated().any():
        raise ValueError(f"duplicated image_id in {run_dir}")
    df["map_path"] = [run_dir / "maps" / map_filename(i) for i in df["image_id"]]
    missing = [str(p) for p in df["map_path"] if not p.exists()]
    if missing:
        raise ValueError(f"missing maps: {missing[:3]} (run anomalydino with save_maps: true)")
    return df


def build_features(test_scores: pd.DataFrame, calib_scores: pd.DataFrame, calib: pd.DataFrame,
                   refs: pd.DataFrame, test: pd.DataFrame, patch_quantile: float
                   ) -> tuple[pd.DataFrame, dict[str, Calibration]]:
    """CI-E5：テストの各画像の計測値と、粗い・細かいの2通りの文。区切りは学習用の良品だけで決める。"""
    check_calib(calib, refs, test)
    if set(calib_scores["image_id"]) != set(calib["image_id"]):
        raise ValueError("calibration scores do not match the calibration split")
    if set(test_scores["image_id"]) != set(test["image_id"]):
        raise ValueError("test scores do not match the test split")
    if (calib_scores["label"] != "normal").any():
        raise ValueError("calibration scores must be normals only")
    cals: dict[str, Calibration] = {}
    for cat, g in calib_scores.groupby("category", sort=True):
        maps = [load_map(p) for p in g["map_path"]]
        cals[cat] = calibrate(cat, maps, g["score"].tolist(), patch_quantile)
    rows = []
    for r in test_scores.sort_values(["category", "label", "image_id"]).itertuples(index=False):
        cal = cals[r.category]
        dist, mask = load_map(r.map_path)
        m = measure(dist, mask, cal.patch_threshold)
        rows.append({"image_id": r.image_id, "category": r.category, "label": r.label, "score": r.score,
                     **asdict(m),
                     "text_coarse": verbalize(r.score, m, cal, "coarse"),
                     "text_fine": verbalize(r.score, m, cal, "fine")})
    return pd.DataFrame(rows), cals


def calibration_table(cals: dict[str, Calibration]) -> pd.DataFrame:
    return pd.DataFrame([{"category": c.category, "n_images": c.n_images, "patch_quantile": c.patch_quantile,
                          "patch_threshold": c.patch_threshold, "score_q90": c.score_q90,
                          "score_max": c.score_max} for c in cals.values()])


def save_calibrations(cals: dict[str, Calibration], path: str | Path) -> None:
    Path(path).write_text(json.dumps({c: asdict(v) for c, v in cals.items()}, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def load_calibrations(path: str | Path) -> dict[str, Calibration]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c: Calibration(**{**v, "scores": tuple(v["scores"])}) for c, v in d.items()}
