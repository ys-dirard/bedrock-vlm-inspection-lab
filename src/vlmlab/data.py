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
        "mask": df["mask"] if "mask" in df.columns else None,  # 不良品のマスク（良品は空）
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
    loc = make_loc_split(table, test)
    paths = {
        "visa_test_200": out / "visa_test_200.csv",
        "visa_refs": out / "visa_refs.csv",
        "visa_wide_120": out / "visa_wide_120.csv",
        "visa_knn_train": out / "visa_knn_train.csv",
        "visa_loc_140": out / "visa_loc_140.csv",
    }
    test.to_csv(paths["visa_test_200"], index=False, lineterminator="\n")
    refs.to_csv(paths["visa_refs"], index=False, lineterminator="\n")
    wide.to_csv(paths["visa_wide_120"], index=False, lineterminator="\n")
    knn.to_csv(paths["visa_knn_train"], index=False, lineterminator="\n")
    loc.to_csv(paths["visa_loc_140"], index=False, lineterminator="\n")
    return paths


def make_loc_split(table: pd.DataFrame, test: pd.DataFrame, n_normal_per_category: int = 20) -> pd.DataFrame:
    """位置の評価（12/2 の記事）：test の不良品すべてと、test の良品から品目ごとに n 枚（random_state=42）。

    マスクのパスの列 `mask` を付ける（L-D1）。不良品にマスクが無ければ止める。
    """
    rows = []
    for cat in test.category.unique():
        t = test[test.category == cat]
        rows.append(t[t.label == "anomaly"])
        rows.append(_sample(t[t.label == "normal"], n_normal_per_category))
    out = pd.concat(rows, ignore_index=True)[SPLIT_COLUMNS]
    masks = table.set_index("image_id")["mask"]
    out["mask"] = [masks[i] if lab == "anomaly" else None for i, lab in zip(out.image_id, out.label)]
    lacking = out[(out.label == "anomaly") & out["mask"].isna()]
    if len(lacking):
        raise ValueError(f"anomaly images without mask: {list(lacking.image_id)[:5]}")
    return out


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


def load_tile(path: str | Path, region: tuple[int, int, int, int]) -> EncodedImage:
    """画像の一部（region = (x0, y0, x1, y1)、画素）を切り出し、JPEG（quality=95）で符号化する（CI-L5）。"""
    with Image.open(path) as im:
        w, h = im.size
        x0, y0, x1, y1 = region
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"tile {region} is outside the image {w}x{h}")
        tile = im.convert("RGB").crop(region)
    buf = io.BytesIO()
    tile.save(buf, format="JPEG", quality=95)
    return EncodedImage(buf.getvalue(), "jpeg", tile.width, tile.height)


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size
