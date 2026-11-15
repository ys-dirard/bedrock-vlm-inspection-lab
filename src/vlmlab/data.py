"""VisA の取得・抽出・画像の読み込みと縮小。

出典: Registry of Open Data on AWS（s3://amazon-visual-anomaly/VisA_20220922.tar、us-west-2、CC BY 4.0）
分割: tar に含まれる split_csv/1cls.csv（spot-diff リポジトリと同じもの）
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

BUCKET = "amazon-visual-anomaly"
KEY = "VisA_20220922.tar"
BUCKET_REGION = "us-west-2"
SEED = 42

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
SPLIT_COLUMNS = ["image_id", "category", "label", "split", "path"]


def download(dest_dir: str | Path) -> Path:
    """tar を認証なし（--no-sign-request と同じ）で取得する。既にあれば取得しない。"""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    dest = Path(dest_dir) / KEY
    s3 = boto3.client("s3", region_name=BUCKET_REGION, config=Config(signature_version=UNSIGNED))
    size = s3.head_object(Bucket=BUCKET, Key=KEY)["ContentLength"]
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".tar.part")
    s3.download_file(BUCKET, KEY, str(part))
    part.replace(dest)
    return dest


def extract(tar_path: str | Path, dest_dir: str | Path) -> Path:
    """tar を展開する。展開先に split_csv/1cls.csv があれば展開しない。"""
    dest = Path(dest_dir)
    if (dest / "split_csv" / "1cls.csv").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")
    return dest


def load_split_table(visa_root: str | Path) -> pd.DataFrame:
    """split_csv/1cls.csv を読み、image_id などの列を足す。

    1cls.csv の列: object, split (train/test), label (normal/anomaly), image（VisA の根からの相対パス）, mask
    """
    df = pd.read_csv(Path(visa_root) / "split_csv" / "1cls.csv")
    required = {"object", "split", "label", "image"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"1cls.csv lacks columns {sorted(missing)}; got {list(df.columns)}")
    out = pd.DataFrame({
        "category": df["object"],
        "label": df["label"],
        "split": df["split"],
        "path": df["image"],
    })
    # 良品と不良品でファイル名が重なるので、品目・正解・ファイル名で ID を作る
    out.insert(0, "image_id", out["category"] + "/" + out["label"] + "/" + out["path"].map(lambda p: Path(p).stem))
    if out["image_id"].duplicated().any():
        raise ValueError("image_id is not unique")
    return out


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"need {n} rows but only {len(df)} available")
    return df.sample(n, random_state=SEED)


def make_test_split(table: pd.DataFrame, categories=("pcb1", "capsules"), n_per_label: int = 50) -> pd.DataFrame:
    """基本のテスト：各品目の test から良品・不良品を n 枚ずつ（random_state=42）。"""
    rows = []
    for cat in categories:
        test = table[(table.category == cat) & (table.split == "test")]
        for label in ("normal", "anomaly"):
            rows.append(_sample(test[test.label == label], n_per_label))
    return pd.concat(rows, ignore_index=True)[SPLIT_COLUMNS]


def make_refs(table: pd.DataFrame, n_per_category: int = 3, categories=CATEGORIES) -> pd.DataFrame:
    """良品の見本：各品目の train の normal から n 枚。test とは重ならない（split が違う）。"""
    rows = []
    for cat in categories:
        train_ok = table[(table.category == cat) & (table.split == "train") & (table.label == "normal")]
        rows.append(_sample(train_ok, n_per_category))
    out = pd.concat(rows, ignore_index=True)[SPLIT_COLUMNS]
    out.insert(1, "ref_rank", out.groupby("category").cumcount() + 1)  # n_refs=1 なら rank 1 だけ使う
    return out


def make_wide_split(table: pd.DataFrame, n_per_label: int = 5, categories=CATEGORIES) -> pd.DataFrame:
    """品目の広がり：12品目 × test の良品・不良品 n 枚ずつ（random_state=42）。"""
    return make_test_split(table, categories=categories, n_per_label=n_per_label)


def make_knn_train(table: pd.DataFrame, exclude_ids: set[str], n_normal: int = 100, n_anomaly: int = 20,
                   categories=("pcb1", "capsules")) -> pd.DataFrame:
    """埋め込み＋k近傍法（条件 E）の学習用：各品目の良品 n_normal 枚と不良品 n_anomaly 枚（random_state=42）。

    VisA の 1cls.csv では train に不良品が無い（不良品は test だけ）。そこで、良品は train から、
    不良品は test のうち exclude_ids（評価に使う画像）に入っていないものから選ぶ。
    """
    rows = []
    for cat in categories:
        t = table[table.category == cat]
        normal = t[(t.split == "train") & (t.label == "normal") & ~t.image_id.isin(exclude_ids)]
        anomaly = t[(t.split == "test") & (t.label == "anomaly") & ~t.image_id.isin(exclude_ids)]
        rows.append(_sample(normal, n_normal))
        rows.append(_sample(anomaly, n_anomaly))
    out = pd.concat(rows, ignore_index=True)[SPLIT_COLUMNS]
    if set(out.image_id) & exclude_ids:
        raise ValueError("knn train overlaps with excluded images")
    return out


def write_splits(visa_root: str | Path, out_dir: str | Path) -> dict[str, Path]:
    table = load_split_table(visa_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = make_test_split(table)
    refs = make_refs(table)
    wide = make_wide_split(table)
    if set(refs.image_id) & (set(test.image_id) | set(wide.image_id)):
        raise ValueError("refs overlap with test images")
    knn = make_knn_train(table, set(test.image_id) | set(wide.image_id))
    paths = {
        "visa_test_200": out / "visa_test_200.csv",
        "visa_refs": out / "visa_refs.csv",
        "visa_wide_120": out / "visa_wide_120.csv",
        "visa_knn_train": out / "visa_knn_train.csv",
    }
    test.to_csv(paths["visa_test_200"], index=False, lineterminator="\n")
    refs.to_csv(paths["visa_refs"], index=False, lineterminator="\n")
    wide.to_csv(paths["visa_wide_120"], index=False, lineterminator="\n")
    knn.to_csv(paths["visa_knn_train"], index=False, lineterminator="\n")
    return paths


@dataclass(frozen=True)
class EncodedImage:
    data: bytes
    format: str  # Converse の image.format（jpeg / png）
    width: int
    height: int


_FORMATS = {"JPEG": "jpeg", "PNG": "png"}


def load_image(path: str | Path, long_edge: int | None = None) -> EncodedImage:
    """画像を読む。long_edge が None なら原寸のバイト列をそのまま返す。

    縮小するときは、長辺を long_edge にそろえ（拡大はしない）、JPEG（quality=95）で符号化し直す。
    """
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        fmt = _FORMATS.get(im.format or "")
        if fmt is None:
            raise ValueError(f"unsupported image format {im.format} for {path}")
        w, h = im.size
        if long_edge is None or max(w, h) <= long_edge:
            return EncodedImage(raw, fmt, w, h)
        scale = long_edge / max(w, h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=95)
    return EncodedImage(buf.getvalue(), "jpeg", size[0], size[1])


# ---- 12/5 の記事：画像の分割と欠陥の大きさ ----

def grid_boxes(width: int, height: int, n: int = 2) -> list[tuple[int, int, int, int]]:
    """画像を n×n に、重なりなく分ける切り出し枠 (left, top, right, bottom) を返す（CI-T1〜T3）。

    順は行優先（左上 → 右上 → 左下 → 右下）。幅・高さが n で割り切れないときの端数は、
    右の列と下の行に入れる（左・上の列と行は width // n、height // n）。
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if width < n or height < n:
        raise ValueError(f"image {width}x{height} is too small to split into {n}x{n}")
    cw, ch = width // n, height // n
    xs = [i * cw for i in range(n)] + [width]
    ys = [j * ch for j in range(n)] + [height]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(n) for i in range(n)]


def load_tiles(path: str | Path, n: int = 2, long_edge: int | None = None) -> list[EncodedImage]:
    """画像を原寸のまま n×n に分け、各タイルを JPEG（quality=95）で符号化して返す（CI-T4）。

    long_edge を指定したときは、切り出したあとで各タイルの長辺をそろえて縮める（拡大はしない）。
    """
    with Image.open(Path(path)) as im:
        if (im.format or "") not in _FORMATS:
            raise ValueError(f"unsupported image format {im.format} for {path}")
        rgb = im.convert("RGB")
    out = []
    for box in grid_boxes(rgb.width, rgb.height, n):
        tile = rgb.crop(box)
        if long_edge is not None and max(tile.size) > long_edge:
            scale = long_edge / max(tile.size)
            tile = tile.resize((max(1, round(tile.width * scale)), max(1, round(tile.height * scale))),
                               Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        tile.save(buf, format="JPEG", quality=95)
        out.append(EncodedImage(buf.getvalue(), "jpeg", tile.width, tile.height))
    return out


SIZE_BINS = ("small", "medium", "large")


def mask_area(mask_path: str | Path) -> tuple[int, int, int, int]:
    """正解マスクの欠陥の画素数を数える。0 より大きい画素を欠陥とする（CI-S1）。

    返り値: (欠陥の画素数, マスクの幅, マスクの高さ, 画素数の合計)
    """
    import numpy as np

    with Image.open(Path(mask_path)) as m:
        a = np.asarray(m.convert("L"))
    h, w = a.shape
    return int((a > 0).sum()), w, h, w * h


def assign_size_bins(df: pd.DataFrame, labels: tuple[str, ...] = SIZE_BINS) -> pd.Series:
    """defect_area_ratio の小さい順に並べ（同じ値は image_id の順）、件数で等分した区分を返す（CI-S2）。

    i 番目（0 始まり）の区分は floor(i × k / N)。N が k で割り切れないときは、小さい側の区分が多くなる
    （100枚・3区分なら 34・33・33）。
    """
    if df.empty:
        raise ValueError("no rows to bin")
    if df["defect_area_ratio"].isna().any():
        raise ValueError("defect_area_ratio has missing values")
    k = len(labels)
    order = df.sort_values(["defect_area_ratio", "image_id"], kind="stable").index
    rank = pd.Series(range(len(df)), index=order)
    return (rank * k // len(df)).map(lambda i: labels[i]).reindex(df.index)


def make_defect_size_table(visa_root: str | Path, split: pd.DataFrame) -> pd.DataFrame:
    """split の不良品について、正解マスクの欠陥の面積の割合と、小・中・大の区分を作る（CI-S1〜S4）。

    マスクのパスは split_csv/1cls.csv の mask 列から、画像のパス（image 列）で引く。
    """
    root = Path(visa_root)
    cls = pd.read_csv(root / "split_csv" / "1cls.csv")
    mask_of = dict(zip(cls["image"], cls["mask"]))
    anomalies = split[split.label == "anomaly"]
    rows = []
    for r in anomalies.itertuples(index=False):
        mask_rel = mask_of.get(r.path)
        if not isinstance(mask_rel, str) or not mask_rel:
            raise ValueError(f"no mask for anomaly image {r.image_id}")
        defect_px, mw, mh, total = mask_area(root / mask_rel)
        if defect_px == 0:
            raise ValueError(f"mask of {r.image_id} has no defect pixels")
        with Image.open(root / r.path) as im:
            if im.size != (mw, mh):
                raise ValueError(f"mask size {(mw, mh)} differs from image size {im.size} for {r.image_id}")
        rows.append({"image_id": r.image_id, "category": r.category, "mask_path": mask_rel,
                     "defect_px": defect_px, "image_px": total,
                     "defect_area_ratio": round(defect_px / total, 9)})
    out = pd.DataFrame(rows)
    out["size_bin"] = assign_size_bins(out)
    return out
