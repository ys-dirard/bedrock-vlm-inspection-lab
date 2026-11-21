"""12/20 の記事：似た不良画像を探す（Nova Multimodal Embeddings ＋ 手元の検索）。

  python -m vlmlab.search.simsearch split --visa-root DIR [--out data/splits]
      VisA の欠陥の種類（各品目の image_anno.csv の label）から、索引とクエリの一覧を作る
  python -m vlmlab.search.simsearch evaluate --index RUN_DIR --query RUN_DIR --split CSV --out DIR [--k 5]
      上位 k 件に同じ種類の不良が入るかを数える（API は呼ばない）

埋め込みは `python -m vlmlab.search embed CONFIG` で作る（索引は GENERIC_INDEX、クエリは IMAGE_RETRIEVAL）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import SEED, load_split_table
from .index import chance_hit_at_k, chance_precision, hit_at_k, precision_at_k, reciprocal_rank, top_k
from .store import load_vectors

# capsules は1種類だけの不良（bubble）が単独で付いた画像しかなく、種類を見分ける検索にならないので除く。
# pcb4 は単独のラベルの不良が 44 枚で、3種類が 2 枚ずつしかないので除く（2026-11-21 に数えた）
CATEGORIES_20 = ["candle", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
                 "pcb1", "pcb2", "pcb3", "pipe_fryum"]
DETAIL = "STANDARD_IMAGE"


def load_defect_types(visa_root: Path, categories: list[str]) -> pd.DataFrame:
    """各品目の image_anno.csv から、不良品の欠陥の種類を読む（列: path, defect_type, mask_path, n_labels）。"""
    rows = []
    for cat in categories:
        a = pd.read_csv(visa_root / cat / "image_anno.csv", keep_default_na=False)
        if not {"image", "label", "mask"} <= set(a.columns):
            raise ValueError(f"{cat}/image_anno.csv lacks image/label/mask")
        a = a[a.label != "normal"]
        for r in a.itertuples(index=False):
            labels = [x.strip() for x in str(r.label).split(",") if x.strip()]
            rows.append({"path": r.image, "defect_type": ",".join(labels), "n_labels": len(labels),
                         "mask_path": r.mask})
    return pd.DataFrame(rows)


def make_split(visa_root: Path, categories: list[str] = CATEGORIES_20, n_per_cat: int = 36, n_query: int = 6,
               n_normal: int = 30, seed: int = SEED) -> pd.DataFrame:
    """索引（index）・クエリ（query）・良品の索引（normal_index）の一覧（CI-D20）。

    1. 品目ごとに、欠陥のラベルがちょうど1つで「other」でない不良品から n_per_cat 枚（random_state=seed）
    2. その中で3枚以上ある種類だけをクエリの候補にし、種類の多い順（同数なら名前順）に1枚ずつ巡回して
       n_query 枚をクエリにする。どのクエリにも、索引に同じ種類が2枚以上残るようにする
    3. 残りを索引にする。良品は test の normal から品目ごとに n_normal 枚（random_state=seed）
    """
    table = load_split_table(visa_root)
    types = load_defect_types(visa_root, categories)
    anom = table[(table.label == "anomaly") & table.category.isin(categories)].merge(types, on="path", how="left")
    if anom.defect_type.isna().any():
        raise ValueError("some anomalies have no row in image_anno.csv")
    rng_state = seed
    out = []
    for cat in categories:
        pool = anom[(anom.category == cat) & (anom.n_labels == 1) & (anom.defect_type != "other")]
        pool = pool.sort_values("image_id")
        if len(pool) < n_per_cat:
            raise ValueError(f"{cat}: only {len(pool)} single-label anomalies (< {n_per_cat})")
        sample = pool.sample(n=n_per_cat, random_state=rng_state).reset_index(drop=True)
        counts = sample.defect_type.value_counts()
        eligible = sorted([t for t, c in counts.items() if c >= 3], key=lambda t: (-counts[t], t))
        picked: dict[str, list[int]] = {t: [] for t in eligible}
        queries: list[int] = []
        while len(queries) < n_query:
            progressed = False
            for t in eligible:
                if len(queries) == n_query:
                    break
                if counts[t] - len(picked[t]) - 1 < 2:
                    continue
                cand = [i for i in sample.index[sample.defect_type == t] if i not in picked[t]]
                picked[t].append(cand[0])
                queries.append(cand[0])
                progressed = True
            if not progressed:
                raise ValueError(f"{cat}: cannot pick {n_query} queries with >= 2 same-type images left")
        sample["role"] = ["query" if i in set(queries) else "index" for i in sample.index]
        out.append(sample)
        normals = table[(table.category == cat) & (table.label == "normal") & (table.split == "test")]
        normals = normals.sort_values("image_id").sample(n=n_normal, random_state=rng_state)
        out.append(normals.assign(defect_type="", n_labels=0, mask_path="", role="normal_index"))
    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={"image_id": "item_id"})
    df.insert(1, "modality", "image")
    df.insert(2, "detail", DETAIL)
    cols = ["item_id", "modality", "detail", "path", "category", "label", "defect_type", "mask_path", "role"]
    return df[cols].sort_values(["role", "category", "item_id"], kind="stable").reset_index(drop=True)


def write_split(visa_root: Path, out_dir: Path) -> dict[str, Path]:
    df = make_split(visa_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"split": out_dir / "ac20_simsearch.csv", "index": out_dir / "ac20_index.csv",
             "query": out_dir / "ac20_query.csv"}
    df.to_csv(paths["split"], index=False, lineterminator="\n")
    df[df.role.isin(["index", "normal_index"])].to_csv(paths["index"], index=False, lineterminator="\n")
    df[df.role == "query"].to_csv(paths["query"], index=False, lineterminator="\n")
    return paths


def evaluate(split: pd.DataFrame, index_emb: Path, query_emb: Path, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2つの索引（不良品だけ／良品も混ぜる）で、クエリごとの上位 k 件を数える（CI-S8）。

    正解 = 同じ品目で、同じ欠陥の種類の不良品。クエリ自身は索引に入っていない（make_split で分けている）。
    """
    q = split[split.role == "query"]
    variants = {"anomaly_only": split[split.role == "index"],
                "with_normals": split[split.role.isin(["index", "normal_index"])]}
    q_ids, q_vec = load_vectors(query_emb, list(q.item_id))
    meta = split.set_index("item_id")
    results, per_query = [], []
    for variant, idx in variants.items():
        ids, vec = load_vectors(index_emb, list(idx.item_id))
        t0 = time.perf_counter()
        all_hits = top_k(q_vec, vec, ids, k)
        search_ms = (time.perf_counter() - t0) * 1000 / len(q_ids)
        for qid, hits in zip(q_ids, all_hits):
            cat, typ = meta.at[qid, "category"], meta.at[qid, "defect_type"]
            relevant = set(idx[(idx.category == cat) & (idx.defect_type == typ) & (idx.role == "index")].item_id)
            for h in hits:
                results.append({"variant": variant, "query_id": qid, "query_category": cat, "query_type": typ,
                                "rank": h.rank, "item_id": h.item_id, "score": round(h.score, 6),
                                "hit_category": meta.at[h.item_id, "category"],
                                "hit_label": meta.at[h.item_id, "label"],
                                "hit_type": meta.at[h.item_id, "defect_type"],
                                "relevant": h.item_id in relevant})
            per_query.append({
                "variant": variant, "query_id": qid, "category": cat, "defect_type": typ,
                "hit_at_1": hit_at_k(hits, relevant, 1), f"hit_at_{k}": hit_at_k(hits, relevant, k),
                f"precision_at_{k}": precision_at_k(hits, relevant, k), "rr": reciprocal_rank(hits, relevant),
                f"same_category_at_{k}": sum(meta.at[h.item_id, "category"] == cat for h in hits[:k]) / k,
                f"normals_in_top{k}": sum(meta.at[h.item_id, "label"] == "normal" for h in hits[:k]),
                "n_relevant": len(relevant), "n_candidates": len(ids),
                f"chance_precision_at_{k}": chance_precision(len(relevant), len(ids)),
                f"chance_hit_at_{k}": chance_hit_at_k(len(relevant), len(ids), k),
                # 同じ品目の中からでたらめに選んだときの期待値（検索がほぼ同じ品目を返すので、こちらとも比べる）
                f"chance_precision_at_{k}_in_category": chance_precision(len(relevant), int((idx.category == cat).sum())),
                "search_ms_per_query": search_ms,
            })
    pq = pd.DataFrame(per_query)
    metrics = ["hit_at_1", f"hit_at_{k}", f"precision_at_{k}", "rr", f"same_category_at_{k}",
               f"normals_in_top{k}", f"chance_precision_at_{k}", f"chance_hit_at_{k}",
               f"chance_precision_at_{k}_in_category"]
    overall = pq.groupby("variant", sort=False)[metrics].mean().reset_index().assign(category="(all)")
    by_cat = pq.groupby(["variant", "category"], sort=False)[metrics].mean().reset_index()
    n = pq.groupby(["variant"], sort=False).size().rename("n_queries").reset_index()
    summary = pd.concat([overall.merge(n, on="variant"), by_cat], ignore_index=True)
    summary["search_ms_per_query"] = summary.variant.map(pq.groupby("variant").search_ms_per_query.first())
    return pd.DataFrame(results), summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m vlmlab.search.simsearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--visa-root", default="data/visa")
    s.add_argument("--out", default="data/splits")
    e = sub.add_parser("evaluate")
    e.add_argument("--index", required=True)
    e.add_argument("--query", required=True)
    e.add_argument("--split", default="data/splits/ac20_simsearch.csv")
    e.add_argument("--out", required=True)
    e.add_argument("--k", type=int, default=5)
    a = p.parse_args(argv)
    if a.cmd == "split":
        print(json.dumps({k: str(v) for k, v in write_split(Path(a.visa_root), Path(a.out)).items()}))
    else:
        split = pd.read_csv(a.split, keep_default_na=False)
        res, summ = evaluate(split, Path(a.index) / "embeddings.jsonl", Path(a.query) / "embeddings.jsonl", a.k)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "search_results.csv", index=False, lineterminator="\n")
        summ.to_csv(out / "search_summary.csv", index=False, lineterminator="\n", float_format="%.4f")
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
