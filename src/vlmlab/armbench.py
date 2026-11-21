"""ARMBench の欠陥検知（画像）を、外観検査と同じ方法で判定する（12/19 の記事）。docs/contracts.md の CI-AB1〜AB7。

出典: ARMBench（Amazon、CC BY 4.0、http://armbench.com 、arXiv:2303.16382）。
配布: Hugging Face `amazon/armbench-defect-image`（ゲートなし）。本体の armbench-defects-image-0.1.tar.gz は 65.7 GiB で、
gzip なので一部だけは取り出せない。試運転は同じ所にある見本（armbench-defects-tiny.tar.gz、約66 MiB、52枚）で行う。

画像の置き方は `<root>/imagenet/<train|test>/<nominal|multi_pick|package_defect>/<pick_id>_<view>.jpg`
（見本の tar で確かめた形。本体も同じかは未確認）。ラベルは画像（視点）ごとで、同じピックでも視点によって
nominal と multi_pick に分かれることがある（見本の 009022001652099684534）。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import metrics as M

SEED = 42
CATEGORY = "armbench"
DEFECT_TYPES = ("multi_pick", "package_defect")
NOMINAL = "nominal"
TYPES = (NOMINAL, *DEFECT_TYPES)
COLUMNS = ["image_id", "category", "label", "split", "path", "defect_type", "pick_id", "view"]
_NAME = re.compile(r"^(?P<pick>\d{21})_(?P<view>\d+)\.(?:jpg|png)$", re.IGNORECASE)  # ミラーには PNG が混ざる（CI-AB7）

# モデルが返す欠陥の種類（プロンプトで列挙したもの）→ ARMBench の欠陥の種類
PRED_TYPE = {"multi_pick": "multi_pick", "package_open": "package_defect", "deconstruction": "package_defect"}


# ---------------------------------------------------------------- 一覧

def read_pick_ids(path: str | Path) -> set[str]:
    """公式の train.csv / test.csv（見出しなし・1列・CRLF・0埋めの21桁）を文字列のまま読む（CI-AB1）。"""
    with Path(path).open(encoding="utf-8", newline="") as f:
        ids = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
    bad = [i for i in ids if not re.fullmatch(r"\d{21}", i)]
    if bad:
        raise ValueError(f"unexpected pick ids in {path}: {bad[:3]}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicated pick ids in {path}")
    return set(ids)


def scan_imagenet(root: str | Path, train_ids: set[str] | None = None, test_ids: set[str] | None = None) -> pd.DataFrame:
    """imagenet の形の置き場所から一覧を作る（CI-AB1）。split は公式の一覧で決め、どちらにも無ければ "none"。

    path は root からの相対パス（`vlmlab run --visa-root <root>` で読む）。
    """
    root = Path(root)
    train_ids, test_ids = train_ids or set(), test_ids or set()
    if train_ids & test_ids:
        raise ValueError("train and test pick ids overlap")
    rows = []
    for p in sorted(root.glob("imagenet/*/*/*")):
        if not p.is_file():
            continue
        dtype = p.parent.name
        if dtype not in TYPES:
            raise ValueError(f"unknown defect type folder: {p.parent}")
        m = _NAME.match(p.name)
        if not m:
            raise ValueError(f"unexpected file name: {p}")
        pick, view = m["pick"], int(m["view"])
        split = "train" if pick in train_ids else "test" if pick in test_ids else "none"
        rows.append({
            "image_id": f"{dtype}/{pick}_{view}",
            "category": CATEGORY,
            "label": "normal" if dtype == NOMINAL else "anomaly",
            "split": split,
            "path": p.relative_to(root).as_posix(),
            "defect_type": dtype,
            "pick_id": pick,
            "view": view,
        })
    if not rows:
        raise ValueError(f"no images under {root}/imagenet")
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df["image_id"].duplicated().any():
        raise ValueError("image_id is not unique")
    return df


def _quotas(n: int, available: dict[str, int]) -> dict[str, int]:
    """n 枚を種類ごとにできるだけ均等に割る。足りない種類の残りはほかの種類へ回す（CI-AB2）。"""
    if sum(available.values()) < n:
        raise ValueError(f"need {n} defect images but only {sum(available.values())} available: {available}")
    q = {t: 0 for t in available}
    left = n
    while left > 0:
        open_types = [t for t in sorted(available) if q[t] < available[t]]
        share, extra = divmod(left, len(open_types))
        for i, t in enumerate(open_types):
            add = min(share + (1 if i < extra else 0), available[t] - q[t])
            q[t] += add
            left -= add
    return q


def sample(table: pd.DataFrame, n_normal: int, n_defect: int, splits: tuple[str, ...] = ("test",),
           seed: int = SEED) -> pd.DataFrame:
    """評価に使う画像を選ぶ（CI-AB2）。

    - 1つのピックから1枚だけ（同じ物の別の視点が並ぶと、件数のわりに独立な例が少なくなるため）
    - 欠陥は種類ごとにできるだけ均等。正常は、欠陥の画像が1枚も無いピックからだけ選ぶ
    - 並べ替えは random_state=seed の1回だけで、同じ入力なら同じ結果
    """
    t = table[table["split"].isin(splits)]
    if t.empty:
        raise ValueError(f"no images in splits {splits}")
    shuffled = t.sample(frac=1.0, random_state=seed)
    one_per_pick = lambda d: d.drop_duplicates("pick_id", keep="first")  # noqa: E731

    defect_picks = set(t.loc[t["defect_type"] != NOMINAL, "pick_id"])
    used: set[str] = set()
    parts = []
    # 種類ごとの候補（ピックで重複を除く）。ほかの種類に先に取られたピックは数に入れない
    avail = {d: len(one_per_pick(shuffled[shuffled["defect_type"] == d])) for d in DEFECT_TYPES}
    quotas = _quotas(n_defect, avail)
    for d in DEFECT_TYPES:
        cand = one_per_pick(shuffled[(shuffled["defect_type"] == d) & ~shuffled["pick_id"].isin(used)])
        if len(cand) < quotas[d]:
            raise ValueError(f"{d}: need {quotas[d]} picks but only {len(cand)} left")
        chosen = cand.head(quotas[d])
        used |= set(chosen["pick_id"])
        parts.append(chosen)
    normal = one_per_pick(shuffled[(shuffled["defect_type"] == NOMINAL) & ~shuffled["pick_id"].isin(defect_picks)])
    if len(normal) < n_normal:
        raise ValueError(f"need {n_normal} nominal picks but only {len(normal)} available")
    parts.append(normal.head(n_normal))
    out = pd.concat(parts, ignore_index=True)[COLUMNS]
    assert not out["pick_id"].duplicated().any()
    return out


# ---------------------------------------------------------------- 集計

def pred_type(defect: Any) -> str | None:
    """モデルの defect をARMBench の種類へ。列挙に無い値は "other"、空・欠けは None（CI-AB5）。"""
    if defect is None or (isinstance(defect, float) and pd.isna(defect)):
        return None
    s = str(defect).strip()
    if s == "":
        return None
    return PRED_TYPE.get(s, "other")


def by_defect_type(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """欠陥の種類ごとの集計（CI-AB3〜AB5）。df は report.load_run の結果（1実験・1回）。

    - 記録の画像の集合が一覧と一致しなければ止める
    - ng_rate：その種類の有効な応答のうち NG の割合（欠陥なら再現率、nominal なら誤検知率）
    - type_match_rate：NG と判定した欠陥のうち、モデルの defect の種類が正解と同じ割合
    """
    if df["repeat"].nunique() != 1:
        raise ValueError("by_defect_type expects a single repeat")
    if set(df["image_id"]) != set(split["image_id"]):
        missing = set(split["image_id"]) - set(df["image_id"])
        extra = set(df["image_id"]) - set(split["image_id"])
        raise ValueError(f"records do not match the split: missing {len(missing)}, extra {len(extra)}")
    d = df.merge(split[["image_id", "defect_type"]], on="image_id", how="left", validate="one_to_one")
    rows = []
    for dtype in TYPES:
        g = d[d["defect_type"] == dtype]
        valid = g[~g["pred"].map(M._is_missing)]
        ng = valid[valid["pred"] == "NG"]
        row: dict[str, Any] = {
            "defect_type": dtype,
            "n": len(g),
            "n_valid": len(valid),
            "n_parse_fail": len(g) - len(valid),
            "n_ng": len(ng),
            "ng_rate": None if len(valid) == 0 else len(ng) / len(valid),
            "metric": "false_positive_rate" if dtype == NOMINAL else "recall",
        }
        if dtype != NOMINAL:
            types = ng["defect"].map(pred_type)
            row["n_type_match"] = int((types == dtype).sum())
            row["type_match_rate"] = None if len(ng) == 0 else row["n_type_match"] / len(ng)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_row(df: pd.DataFrame) -> dict[str, Any]:
    """1実験の全体の指標と費用（CI-AB6）。"""
    m = M.binary_metrics(df)
    return {**m, **M.cost_stats(df), **M.token_stats(df), **M.latency_stats(df)}


def build_report(run_dirs: list[str | Path], split_csv: str | Path, out_dir: str | Path,
                 tag: str = "armbench_summary") -> dict[str, Path]:
    from .report import load_run

    split = pd.read_csv(split_csv, dtype={"pick_id": str})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_type, overall = [], []
    for rd in run_dirs:
        df = load_run(rd)
        name, model = df["run_name"].iloc[0], df["model_id"].iloc[0]
        t = by_defect_type(df, split)
        t.insert(0, "model_id", model)
        t.insert(0, "run_name", name)
        by_type.append(t)
        overall.append({"run_name": name, "model_id": model, "n_api_error": df.attrs["n_api_error"],
                        "cost_all_calls_usd": df.attrs["cost_all_calls_usd"], **summary_row(df)})
    paths = {"by_type": out / f"{tag}.csv", "overall": out / f"{tag}_overall.csv"}
    pd.concat(by_type, ignore_index=True).to_csv(paths["by_type"], index=False, lineterminator="\n")
    pd.DataFrame(overall).to_csv(paths["overall"], index=False, lineterminator="\n")
    return paths


# ---------------------------------------------------------------- 第三者のミラー（parquet）から画像を書き出す（CI-AB7）

MIRROR_REPO = "correll/armbench_defect_dataset"  # CC BY 4.0。公式の配布ではない（公式のカードは split を保たないと注意している）
_MAGIC = {b"\xff\xd8\xff": ".jpg", b"\x89PNG": ".png"}
_MIRROR_ID = re.compile(r"^(?P<pick>\d{21})_(?P<view>\d+)(?:\.jpg)?$")  # ミラーの id には .jpg の無いものもある


def export_mirror(parquet_paths: list[str | Path], out_root: str | Path, batch_size: int = 32) -> pd.DataFrame:
    """ミラーの parquet（列 rgb={bytes, path}、meta={id, label, sublabel, polygon}）から、
    `<out_root>/imagenet/mirror/<label>/<id の拡張子を中身に合わせた名前>` に画像のバイト列をそのまま書き出し、
    1枚1行の表（id・label・sublabel・shard・format・bytes）を返す。

    - label が nominal・multi_pick・package_defect 以外、id が「21桁_視点」（.jpg は付いても付かなくてもよい）でない、
      JPEG でも PNG でもない、id が重なるなら止める
    - ミラーの id は .jpg 付きと無しが混ざり、中身が PNG のものがある（parquet 0〜3番で 423枚、うち package_defect 316枚）。
      バイト列は変えず、拡張子だけ中身に合わせる。判定の前に縮めて JPEG で符号化し直すので、送る画像の形式はそろう
    - pyarrow が要る（`uv run --with pyarrow ...`。既定の依存には入れていない）
    """
    import pyarrow.parquet as pq

    out_root = Path(out_root)
    seen: dict[str, str] = {}
    rows = []
    for p in map(Path, parquet_paths):
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=["rgb", "meta"], batch_size=batch_size):
            for rgb, meta in zip(batch.column("rgb").to_pylist(), batch.column("meta").to_pylist()):
                iid, label = meta["id"], meta["label"]
                if label not in TYPES:
                    raise ValueError(f"{p.name}: unknown label {label!r} for {iid}")
                m = _MIRROR_ID.match(iid)
                if not m:
                    raise ValueError(f"{p.name}: unexpected id {iid!r}")
                data = rgb["bytes"] or b""
                ext = next((e for m, e in _MAGIC.items() if data.startswith(m)), None)
                if ext is None:
                    raise ValueError(f"{p.name}: {iid} is neither JPEG nor PNG")
                if iid in seen:
                    raise ValueError(f"duplicated id {iid} in {seen[iid]} and {p.name}")
                seen[iid] = p.name
                dest = out_root / "imagenet" / "mirror" / label / f"{m['pick']}_{m['view']}{ext}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                rows.append({"id": iid, "label": label, "sublabel": meta.get("sublabel"), "shard": p.name,
                             "format": ext.lstrip("."), "bytes": len(data)})
    if not rows:
        raise ValueError("no rows in parquet files")
    return pd.DataFrame(rows)
