"""MVTec AD・MVTec AD 2・MVTec LOCO AD の画像の一覧と、テスト・見本の抜き出し（12/7・12/15 の記事）。docs/contracts.md の CI-V1〜V6。

データは公式ページのフォーム（氏名・メール・非商用の確認）から取得する。ここでは取得しない。
ライセンスはどれも CC BY-NC-SA 4.0（非商用）。データ本体はコミットしない。

置き場所（例）: data/mvtec/ の下に、展開したデータセットを1つずつ置く。
  data/mvtec/mvtec_ad/<品目>/{train/good, test/<good|欠陥の種類>, ground_truth}
  data/mvtec/mvtec_ad_2/<品目>/{train/good, validation/good, test_public/{good,bad,ground_truth}, test_private, test_private_mixed}
  data/mvtec/mvtec_loco_ad/<品目>/{train/good, validation/good, test/{good,logical_anomalies,structural_anomalies}, ground_truth}
一覧の path は、画像の根（--image-root。例 data/mvtec）からの相対パス。`vlmlab run --image-root data/mvtec` で読む。

ディレクトリの形は、論文の分割の名前と広く使われている形からの推定（実物で未確認）。
推定が外れても気づけるよう、品目ごとの枚数を論文の表と突き合わせ、合わなければ止める（CI-V2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SEED = 42
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SPLIT_COLUMNS = ["image_id", "category", "label", "split", "path", "dataset", "subtype"]

DATASETS = ("mvtec_ad", "mvtec_ad_2", "mvtec_loco_ad")


def norm_name(name: str) -> str:
    """品目名の照合用。大文字・小文字、_、-、空白の違いを無視する（wall_plugs と wallplugs など）。"""
    return "".join(ch for ch in name.lower() if ch.isalnum())


# ---------------------------------------------------------------- 論文の枚数（CI-V2）
# MVTec AD: Bergmann et al., IJCV 129 (2021), Table 1（https://link.springer.com/article/10.1007/s11263-020-01400-4/tables/1、2026-11-21 確認）
#   (train の良品, test の良品, test の不良品)
EXPECTED_AD = {
    "carpet": (280, 28, 89), "grid": (264, 21, 57), "leather": (245, 32, 92), "tile": (230, 33, 84),
    "wood": (247, 19, 60), "bottle": (209, 20, 63), "cable": (224, 58, 92), "capsule": (219, 23, 109),
    "hazelnut": (391, 40, 70), "metalnut": (220, 22, 93), "pill": (267, 26, 141), "screw": (320, 41, 119),
    "toothbrush": (60, 12, 30), "transistor": (213, 60, 40), "zipper": (240, 32, 119),
}
# MVTec AD 2: Heckler-Kram et al., arXiv:2503.21622v1, Table IV（https://arxiv.org/html/2503.21622v1、2026-11-21 確認）
#   (train, validation, test_public の良品, test_public の不良品)。正解が公開されているのは test_public だけ
EXPECTED_AD2 = {
    "can": (412, 46, 72, 90), "fabric": (387, 43, 66, 90), "fruitjelly": (263, 37, 20, 60),
    "rice": (313, 35, 42, 90), "sheetmetal": (137, 19, 24, 90), "vial": (291, 41, 35, 105),
    "wallplugs": (293, 33, 60, 90), "walnuts": (432, 48, 60, 90),
}
# MVTec LOCO AD: Bergmann et al., IJCV 130 (2022), Table 1（https://link.springer.com/article/10.1007/s11263-022-01578-9/tables/1、2026-11-21 確認）
#   (train, validation, test の良品, test の構造的な異常, test の論理的な異常)
EXPECTED_LOCO = {
    "breakfastbox": (351, 62, 102, 90, 83), "screwbag": (360, 60, 122, 82, 137),
    "pushpins": (372, 69, 138, 81, 91), "splicingconnectors": (354, 59, 119, 85, 108),
    "juicebottle": (335, 54, 94, 94, 142),
}


class LayoutError(ValueError):
    """ディレクトリの形か枚数が想定と違う。"""


def _images(d: Path) -> list[Path]:
    if not d.is_dir():
        raise LayoutError(f"directory not found: {d}")
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise LayoutError(f"no images in {d}")
    return files


def _rows(files: list[Path], image_root: Path, dataset: str, category: str, split: str,
          label: str, subtype: str) -> list[dict]:
    out = []
    for f in files:
        try:
            rel = f.relative_to(image_root).as_posix()
        except ValueError as e:
            raise LayoutError(f"{f} is not under image_root {image_root}") from e
        out.append({
            # 品目・分割・下位ディレクトリ・ファイル名で ID を作る（000.png が下位ディレクトリごとに重なるため）
            "image_id": f"{dataset}/{category}/{split}/{f.parent.name}/{f.stem}",
            "category": category, "label": label, "split": split, "path": rel,
            "dataset": dataset, "subtype": subtype,
        })
    return out


def _check(name: str, table: dict, got: tuple, what: str, allow_mismatch: bool, warnings: list[str]) -> None:
    exp = table.get(norm_name(name))
    if exp is None:
        msg = f"{name}: not in the paper's table ({what}); cannot check counts"
    elif tuple(exp) != tuple(got):
        msg = f"{name}: counts {got} differ from the paper's table {exp} ({what})"
    else:
        return
    if not allow_mismatch:
        raise LayoutError(msg + "; pass allow_count_mismatch=True to continue")
    warnings.append(msg)


@dataclass
class Listing:
    table: pd.DataFrame
    warnings: list[str]


def list_ad(root: Path, image_root: Path, categories: list[str], allow_count_mismatch: bool = False) -> Listing:
    """MVTec AD：train/good（見本の候補）と test（good は normal、それ以外の下位ディレクトリは anomaly）。"""
    rows, warns = [], []
    for cat in categories:
        base = root / cat
        train = _rows(_images(base / "train" / "good"), image_root, "mvtec_ad", cat, "train", "normal", "good")
        test_dir = base / "test"
        if not test_dir.is_dir():
            raise LayoutError(f"directory not found: {test_dir}")
        test = []
        for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            label = "normal" if sub.name == "good" else "anomaly"
            test += _rows(_images(sub), image_root, "mvtec_ad", cat, "test", label, sub.name)
        n_good = sum(r["label"] == "normal" for r in test)
        _check(cat, EXPECTED_AD, (len(train), n_good, len(test) - n_good), "train/test good/defective",
               allow_count_mismatch, warns)
        rows += train + test
    return Listing(_finish(rows), warns)


def list_ad2(root: Path, image_root: Path, categories: list[str], allow_count_mismatch: bool = False) -> Listing:
    """MVTec AD 2：train/good（見本の候補）と test_public の good・bad だけ。

    test_private・test_private_mixed は正解が評価サーバーにしかないので使わない（一覧に入れない）。
    validation は枚数の確かめだけに使う。
    """
    rows, warns = [], []
    for cat in categories:
        base = root / cat
        train = _rows(_images(base / "train" / "good"), image_root, "mvtec_ad_2", cat, "train", "normal", "good")
        n_val = len(_images(base / "validation" / "good"))
        good = _rows(_images(base / "test_public" / "good"), image_root, "mvtec_ad_2", cat, "test_public", "normal", "good")
        bad = _rows(_images(base / "test_public" / "bad"), image_root, "mvtec_ad_2", cat, "test_public", "anomaly", "bad")
        _check(cat, EXPECTED_AD2, (len(train), n_val, len(good), len(bad)), "train/validation/test_public good/bad",
               allow_count_mismatch, warns)
        rows += train + good + bad
    return Listing(_finish(rows), warns)


LOCO_SUBTYPES = {"good": "normal", "logical_anomalies": "anomaly", "structural_anomalies": "anomaly"}


def list_loco(root: Path, image_root: Path, categories: list[str], allow_count_mismatch: bool = False) -> Listing:
    """MVTec LOCO AD：train/good（見本の候補）と test の good・logical_anomalies・structural_anomalies。"""
    rows, warns = [], []
    for cat in categories:
        base = root / cat
        train = _rows(_images(base / "train" / "good"), image_root, "mvtec_loco_ad", cat, "train", "normal", "good")
        n_val = len(_images(base / "validation" / "good"))
        test_dir = base / "test"
        found = sorted(p.name for p in test_dir.iterdir() if p.is_dir()) if test_dir.is_dir() else []
        if set(found) != set(LOCO_SUBTYPES):
            raise LayoutError(f"{test_dir}: expected subdirectories {sorted(LOCO_SUBTYPES)}, found {found}")
        test = {sub: _rows(_images(test_dir / sub), image_root, "mvtec_loco_ad", cat, "test", label, sub)
                for sub, label in LOCO_SUBTYPES.items()}
        got = (len(train), n_val, len(test["good"]), len(test["structural_anomalies"]), len(test["logical_anomalies"]))
        _check(cat, EXPECTED_LOCO, got, "train/validation/test good/structural/logical", allow_count_mismatch, warns)
        rows += train + test["good"] + test["structural_anomalies"] + test["logical_anomalies"]
    return Listing(_finish(rows), warns)


def _finish(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SPLIT_COLUMNS)
    if df["image_id"].duplicated().any():
        raise LayoutError("image_id is not unique")
    return df


# ---------------------------------------------------------------- 抜き出し

def _sample(df: pd.DataFrame, n: int, what: str) -> pd.DataFrame:
    """image_id の順に並べてから random_state=42 で n 枚（入力の並びによらず同じ結果。CI-V4）。"""
    if len(df) < n:
        raise ValueError(f"{what}: need {n} images but only {len(df)} available")  # CI-V3：他から埋めない
    return df.sort_values("image_id", kind="stable").sample(n, random_state=SEED)


def sample_test(table: pd.DataFrame, n_per_subtype: dict[str, int]) -> pd.DataFrame:
    """品目ごとに、下位の種類ごとの枚数で抜き出す（CI-V3・V4）。

    n_per_subtype のキーは subtype（"good"、"bad"、"logical_anomalies" など）か、"anomaly"（不良品の種類をまとめて）。
    MVTec AD は {"good": 25, "anomaly": 25}、AD 2 は {"good": 25, "bad": 25}、
    LOCO は {"good": 50, "logical_anomalies": 25, "structural_anomalies": 25}。
    """
    test = table[table["split"] != "train"]
    out = []
    for cat in sorted(test["category"].unique()):
        t = test[test["category"] == cat]
        for key, n in n_per_subtype.items():
            pool = t[t["label"] == "anomaly"] if key == "anomaly" else t[t["subtype"] == key]
            out.append(_sample(pool, n, f"{cat}/{key}"))
    return pd.concat(out, ignore_index=True)[SPLIT_COLUMNS]


def sample_refs(table: pd.DataFrame, n_per_category: int = 3) -> pd.DataFrame:
    """良品の見本：各品目の train の良品から n 枚。ref_rank 1..n（n_refs=1 なら 1 だけ使う）。test とは分割が違うので重ならない（CI-V5）。"""
    out = []
    for cat in sorted(table["category"].unique()):
        pool = table[(table["category"] == cat) & (table["split"] == "train") & (table["label"] == "normal")]
        out.append(_sample(pool, n_per_category, f"{cat}/train/good"))
    refs = pd.concat(out, ignore_index=True)[SPLIT_COLUMNS]
    refs.insert(1, "ref_rank", refs.groupby("category").cumcount() + 1)
    return refs


def write_split(test: pd.DataFrame, refs: pd.DataFrame, test_path: Path, refs_path: Path) -> None:
    if set(test["image_id"]) & set(refs["image_id"]):
        raise ValueError("refs overlap with test images")  # CI-V5
    for p in (test_path, refs_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(test_path, index=False, lineterminator="\n")
    refs.to_csv(refs_path, index=False, lineterminator="\n")


LISTERS = {"mvtec_ad": list_ad, "mvtec_ad_2": list_ad2, "mvtec_loco_ad": list_loco}
DEFAULT_COUNTS = {
    "mvtec_ad": {"good": 25, "anomaly": 25},
    "mvtec_ad_2": {"good": 25, "bad": 25},
    "mvtec_loco_ad": {"good": 50, "logical_anomalies": 25, "structural_anomalies": 25},
}
