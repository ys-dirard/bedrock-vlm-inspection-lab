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


# VisA の image_anno.csv の欠陥の種類（英語）→ 見本の説明に使う日本語（12/3 の記事。訳は筆者）
DEFECT_TYPE_JA = {
    "melt": "溶け", "missing": "部品の欠落", "bent": "曲がり", "scratch": "傷",
    "bubble": "気泡", "discolor": "変色", "leak": "漏れ", "misshape": "変形",
}


def load_defect_types(visa_root: str | Path, categories=("pcb1", "capsules")) -> pd.DataFrame:
    """各品目の image_anno.csv を読み、不良品の画像のパスと欠陥の種類（label 列。複数は「,」区切り）を返す。"""
    rows = []
    for cat in categories:
        anno = pd.read_csv(Path(visa_root) / cat / "image_anno.csv")
        anno = anno[anno["label"] != "normal"]
        rows.append(pd.DataFrame({"path": anno["image"], "defect_type": anno["label"]}))
    return pd.concat(rows, ignore_index=True)


def defect_note(defect_type: str) -> str:
    """欠陥の種類（"bubble,discolor" など）を日本語の説明（"気泡・変色"）にする。知らない種類なら止める。"""
    parts = [t.strip() for t in defect_type.split(",")]
    unknown = [t for t in parts if t not in DEFECT_TYPE_JA]
    if unknown:
        raise ValueError(f"unknown defect types {unknown}; add them to DEFECT_TYPE_JA")
    return "・".join(DEFECT_TYPE_JA[t] for t in parts)


def make_defect_refs(table: pd.DataFrame, defect_types: pd.DataFrame, exclude_ids: set[str], n_per_category: int = 3,
                     categories=("pcb1", "capsules")) -> pd.DataFrame:
    """不良品の見本（CI-D2）：各品目で、評価に使わない不良品から、欠陥の種類が違う n 枚。

    - 候補は test の anomaly のうち exclude_ids（評価に使う画像）に入っていないもの（VisA の train に不良品は無い）
    - 欠陥の種類（image_anno.csv の label の文字列そのもの）ごとに候補を数え、多い順（同数なら名前の順）に n 種類
    - 種類ごとに1枚を random_state=42 で選ぶ。ref_rank は種類の順（見本1枚の条件では rank 1 を使う）
    """
    with_type = table.merge(defect_types, on="path", how="left")
    rows = []
    for cat in categories:
        pool = with_type[(with_type.category == cat) & (with_type.split == "test") & (with_type.label == "anomaly")
                         & ~with_type.image_id.isin(exclude_ids)]
        if pool["defect_type"].isna().any():
            raise ValueError(f"defect type missing for some {cat} anomalies")
        pool = pool.sort_values("image_id")  # 入力の並びによらず同じ結果にする
        counts = pool.groupby("defect_type").size().reset_index(name="n")
        counts = counts.sort_values(["n", "defect_type"], ascending=[False, True])
        if len(counts) < n_per_category:
            raise ValueError(f"{cat}: only {len(counts)} defect types available, need {n_per_category}")
        for rank, dtype in enumerate(counts["defect_type"].head(n_per_category), start=1):
            rows.append(_sample(pool[pool.defect_type == dtype], 1).assign(ref_rank=rank))
    out = pd.concat(rows, ignore_index=True)
    out["note"] = out["defect_type"].map(defect_note)
    out = out[["image_id", "ref_rank", "category", "label", "split", "path", "defect_type", "note"]]
    if set(out.image_id) & exclude_ids:
        raise ValueError("defect refs overlap with excluded images")
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
    defect_refs = make_defect_refs(table, load_defect_types(visa_root), set(test.image_id) | set(wide.image_id))
    paths = {
        "visa_test_200": out / "visa_test_200.csv",
        "visa_refs": out / "visa_refs.csv",
        "visa_wide_120": out / "visa_wide_120.csv",
        "visa_knn_train": out / "visa_knn_train.csv",
        "visa_defect_refs": out / "visa_defect_refs.csv",
    }
    test.to_csv(paths["visa_test_200"], index=False, lineterminator="\n")
    refs.to_csv(paths["visa_refs"], index=False, lineterminator="\n")
    wide.to_csv(paths["visa_wide_120"], index=False, lineterminator="\n")
    knn.to_csv(paths["visa_knn_train"], index=False, lineterminator="\n")
    defect_refs.to_csv(paths["visa_defect_refs"], index=False, lineterminator="\n")
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
