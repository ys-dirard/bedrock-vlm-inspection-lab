"""12/17：記憶に入れる事例の選び方と、記録の文（docs/contracts.md の CI-AC-M1〜M5）。

2026-11-21 に書き手が決めたこと（記事の results/memory_cases_options.md の (Recommended) の3つ）:
- 不良品は VisA の 1cls の test の不良品のうち、visa_test_200.csv に入っていないもの
- 100件版は不良品50・良品50（品目ごとに 25・25）。20件版は100件版から品目×正解ごとに5件ずつ
- 特徴の文は 12/13 のコード（AnomalyDINO の地図 → 領域の計測 → 文）で作る
"""

from __future__ import annotations

import pandas as pd

SEED = 42
CATEGORIES = ("capsules", "pcb1")
PER_GROUP_100 = 25
PER_GROUP_20 = 5
JUDGEMENT = {"anomaly": "NG", "normal": "OK"}


def image_id_of(row: pd.Series) -> str:
    """1cls.csv の行から、lab の split と同じ形の image_id（<品目>/<正解>/<番号>）を作る。"""
    return f"{row['object']}/{row['label']}/{row['image'].split('/')[-1].split('.')[0]}"


def from_1cls(one_cls: pd.DataFrame) -> pd.DataFrame:
    df = one_cls[one_cls["object"].isin(CATEGORIES)].copy()
    df["image_id"] = df.apply(image_id_of, axis=1)
    return df.rename(columns={"object": "category", "image": "path"})[
        ["image_id", "category", "label", "split", "path"]]


def select_memory_cases(one_cls: pd.DataFrame, test200: pd.DataFrame, refs: pd.DataFrame,
                        calib: pd.DataFrame, eval_split: pd.DataFrame) -> pd.DataFrame:
    """CI-AC-M1〜M3。記憶の100件（in_v20 の列で20件版の印）。"""
    pool = from_1cls(one_cls)
    exclude = set(test200["image_id"]) | set(refs["image_id"]) | set(calib["image_id"]) | set(eval_split["image_id"])
    anomalies = pool[(pool["split"] == "test") & (pool["label"] == "anomaly") & ~pool["image_id"].isin(exclude)]
    normals = pool[(pool["split"] == "train") & (pool["label"] == "normal") & ~pool["image_id"].isin(exclude)]
    parts = []
    for cat in CATEGORIES:
        for label, src in (("anomaly", anomalies), ("normal", normals)):
            g = src[src["category"] == cat].sort_values("image_id")
            if len(g) < PER_GROUP_100:
                raise ValueError(f"not enough {cat}/{label}: {len(g)} < {PER_GROUP_100}")
            parts.append(g.sample(PER_GROUP_100, random_state=SEED))
    cases = pd.concat(parts).reset_index(drop=True)
    v20 = set()
    for _, g in cases.groupby(["category", "label"], sort=True):
        v20 |= set(g.sort_values("image_id").sample(PER_GROUP_20, random_state=SEED)["image_id"])
    cases["in_v20"] = cases["image_id"].isin(v20)
    check_memory_cases(cases, exclude)
    return cases.sort_values(["category", "label", "image_id"]).reset_index(drop=True)


def check_memory_cases(cases: pd.DataFrame, exclude: set[str]) -> None:
    """CI-AC-M1〜M3 の事後条件。"""
    if cases["image_id"].duplicated().any():
        raise ValueError("duplicated image_id in memory cases")
    overlap = set(cases["image_id"]) & exclude
    if overlap:
        raise ValueError(f"memory cases overlap with test/refs/calib/eval: {sorted(overlap)[:5]}")
    counts = cases.groupby(["category", "label"]).size()
    if len(counts) != 4 or (counts != PER_GROUP_100).any():
        raise ValueError(f"each category x label must have {PER_GROUP_100}: {counts.to_dict()}")
    c20 = cases[cases["in_v20"]].groupby(["category", "label"]).size()
    if len(c20) != 4 or (c20 != PER_GROUP_20).any():
        raise ValueError(f"v20 must have {PER_GROUP_20} per category x label: {c20.to_dict()}")
    bad = cases[(cases["label"] == "anomaly") & (cases["split"] != "test")]
    if not bad.empty or ((cases["label"] == "normal") & (cases["split"] != "train")).any():
        raise ValueError("anomalies must come from test, normals from train")


def record_text(category: str, feature_text: str, label: str) -> str:
    """CI-AC-M4。記憶の1件の本文：品目・12/13 の細かい文・確定した判定。"""
    if label not in JUDGEMENT:
        raise ValueError(f"label must be normal/anomaly: {label!r}")
    if not feature_text.strip():
        raise ValueError("feature text is empty")
    return f"品目: {category}\n{feature_text.strip()}\n確定した判定: {JUDGEMENT[label]}"


def query_text(category: str, feature_text: str) -> str:
    """CI-AC-M5。検索の文：品目と 12/13 の細かい文だけ（正解・判定は入れない）。"""
    if not feature_text.strip():
        raise ValueError("feature text is empty")
    return f"品目: {category}\n{feature_text.strip()}"
