"""MMAD の問題の表、層別の抜き出し、画像の読み込みと縮小（CI-MD1〜MD5）。

出典
- 問題: GitHub jam-cc/MMAD の dataset/MMAD/mmad.json（commit を MMAD_JSON_COMMIT に固定）
- 画像: Hugging Face jiang-cc/MMAD の ZIP（revision を HF_REVISION に固定）。元は MVTec AD、MVTec LOCO、
  VisA、GoodsAD、DS-MVTec。研究目的のみで、元のデータセットのライセンスに従う
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from ..data import EncodedImage

MMAD_JSON_COMMIT = "49db95b39b49acd344c7b915d0b422b09d303102"
MMAD_JSON_URL = f"https://raw.githubusercontent.com/jam-cc/MMAD/{MMAD_JSON_COMMIT}/dataset/MMAD/mmad.json"
MMAD_JSON_SHA256 = "c219c6b0c8794c017519d398374b72522fe18649d0fbac1bc0c563501de08143"
HF_REVISION = "13e5c0e403fb9c9a09babfab20027bc03736fb97"
HF_ZIP_URL = "https://huggingface.co/datasets/jiang-cc/MMAD/resolve/{rev}/{source}.zip"
SEED = 42

# summary.py の normalize_dataset_name と同じ
DATASET_MERGE = {"DS-MVTec": "MVTec-AD"}
DATASETS = ("GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA")  # 名前の順（抜き出しの割り振りに使う）

# summary.py と同じく、物体の構造・詳細は「物体の分析」にまとめる
TASK_MERGE = {"Object Structure": "Object Analysis", "Object Details": "Object Analysis"}
TASKS = (
    "Anomaly Detection", "Defect Classification", "Defect Localization", "Defect Description",
    "Defect Analysis", "Object Classification", "Object Analysis",
)

SPLIT_COLUMNS = [
    "record_id", "qid", "image_key", "dataset", "task", "raw_type", "q_index",
    "is_normal", "answer", "n_options", "template",
]


def fetch_mmad_json(dest: str | Path) -> Path:
    """mmad.json を固定した commit から取り、SHA-256 を確かめる（CI-MD1）。既にあれば確かめるだけ。"""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MMAD_JSON_URL) as r:
            data = r.read()
        part = dest.with_suffix(".part")
        part.write_bytes(data)
        part.replace(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != MMAD_JSON_SHA256:
        raise ValueError(f"mmad.json sha256 mismatch: {digest} (expected {MMAD_JSON_SHA256})")
    return dest


def question_table(mmad: dict[str, Any]) -> pd.DataFrame:
    """mmad.json を、1行1問の表にする（CI-MD2）。

    - qid = "<画像のキー>#<画像の中での問題の番号（0始まり）>"
    - dataset は DS-MVTec を MVTec-AD にまとめた名前、task は7課題にまとめた名前
    - is_normal は summary.py と同じく、画像のキーに "good" が含まれるか（異常の判別だけで使う）
    - template は random_templates[0]（論文の既定の 1-shot）
    """
    rows = []
    for key, item in mmad.items():
        source = key.split("/")[0]
        templates = item.get("random_templates") or []
        for i, q in enumerate(item.get("conversation", [])):
            raw_type = q["type"]
            rows.append({
                "qid": f"{key}#{i}",
                "image_key": key,
                "source": source,
                "dataset": DATASET_MERGE.get(source, source),
                "task": TASK_MERGE.get(raw_type, raw_type),
                "raw_type": raw_type,
                "q_index": i,
                "is_normal": "good" in key,
                "answer": q["Answer"],
                "n_options": len(q["Options"]),
                "template": templates[0] if templates else None,
            })
    df = pd.DataFrame(rows)
    unknown = set(df["task"]) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown question types: {sorted(unknown)}")
    unknown_ds = set(df["dataset"]) - set(DATASETS)
    if unknown_ds:
        raise ValueError(f"unknown datasets: {sorted(unknown_ds)}")
    return df


def quotas(per_task: int) -> dict[str, int]:
    """1課題の問題数を、データセットの名前の順に、できるだけ均等に割り振る（CI-MD3）。"""
    base, extra = divmod(per_task, len(DATASETS))
    return {ds: base + (1 if i < extra else 0) for i, ds in enumerate(DATASETS)}


def sample_questions(table: pd.DataFrame, per_task: int = 150, seed: int = SEED) -> pd.DataFrame:
    """課題 × データセット（異常の判別はさらに良品・不良品）で層を分け、無作為に抜き出す（CI-MD3、MD4）。

    - 各課題 per_task 問を quotas() で割り振る
    - 異常の判別は、各データセットの中で良品 q//2、不良品 q - q//2
    - 層の中は qid の順に並べてから pandas の sample(random_state=seed) で選ぶ
    - 層の問題数が足りなければ ValueError（足りない分を他の層で埋めない）
    """
    if table["template"].isna().any():
        raise ValueError("some questions have no random_templates")
    out = _stratified(table, per_task, seed)
    out.insert(0, "record_id", [f"m{i + 1:04d}" for i in range(len(out))])
    return out[SPLIT_COLUMNS]


def subsample_split(split: pd.DataFrame, per_task: int, seed: int = SEED) -> pd.DataFrame:
    """抜き出し済みの一覧から、同じ層の分け方でさらに少なく抜き出す（CI-MD6）。record_id は元のまま残す。

    予算の都合で一部のモデルだけ問題を減らすときに使う（11/19 の Sonnet 4.6）。結果は元の一覧の部分集合。
    """
    out = _stratified(split, per_task, seed)
    return out.sort_values("record_id").reset_index(drop=True)[SPLIT_COLUMNS]


def _stratified(table: pd.DataFrame, per_task: int, seed: int) -> pd.DataFrame:
    picks = []
    for task in TASKS:
        for ds, q in quotas(per_task).items():
            cell = table[(table["task"] == task) & (table["dataset"] == ds)]
            if task == "Anomaly Detection":
                strata = [(cell[cell["is_normal"]], q // 2), (cell[~cell["is_normal"]], q - q // 2)]
            else:
                strata = [(cell, q)]
            for stratum, n in strata:
                if len(stratum) < n:
                    raise ValueError(f"stratum {task}/{ds} has {len(stratum)} questions, need {n}")
                picks.append(stratum.sort_values("qid").sample(n=n, random_state=seed).sort_values("qid"))
    return pd.concat(picks, ignore_index=True)


def load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"record_id": str, "answer": str})
    missing = set(SPLIT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"split lacks columns {sorted(missing)}")
    if df["record_id"].duplicated().any() or df["qid"].duplicated().any():
        raise ValueError("record_id / qid must be unique")
    return df


def needed_images(split: pd.DataFrame, n_shots: int = 1) -> list[str]:
    keys = set(split["image_key"])
    if n_shots > 0:
        keys |= set(split["template"])
    return sorted(keys)


def encode_like_gpt4o(raw: bytes, max_size: int = 512) -> EncodedImage:
    """gpt4o.py の encode_image_to_base64 と同じ大きさにして JPEG にする（CI-MD5）。

    scale = min(max_size / 幅, max_size / 高さ)、新しい大きさ = (int(幅×scale), int(高さ×scale))。
    gpt4o.py と同じく、小さい画像は拡大する。縮小は面積平均（cv2.INTER_AREA に近い BOX）、拡大は BICUBIC。
    JPEG の画質は 95（cv2.imencode の既定）。
    """
    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
        scale = min(max_size / w, max_size / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resample = Image.Resampling.BOX if scale < 1 else Image.Resampling.BICUBIC
        small = im.convert("RGB").resize(size, resample)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=95)
    return EncodedImage(buf.getvalue(), "jpeg", size[0], size[1])


def load_mmad_image(images_root: str | Path, key: str, max_size: int = 512) -> EncodedImage:
    return encode_like_gpt4o((Path(images_root) / key).read_bytes(), max_size)


def load_mmad(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
