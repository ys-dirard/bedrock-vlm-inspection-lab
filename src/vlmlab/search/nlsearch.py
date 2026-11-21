"""12/21 の記事：言葉で不良画像を探す（説明文を付けて探す A と、画像のまま埋め込んで探す B）。

  python -m vlmlab.search.nlsearch positions --visa-root DIR
      12/20 の索引の不良品 300 枚に、正解マスクから位置（3×3 の区画）を付け、説明文づくりの一覧も作る
  python -m vlmlab.search.nlsearch query-template [--per-category 3]
      クエリの雛形（品目×種類×位置と正解の枚数）を作る。クエリの文は書き手が書く
  python -m vlmlab.search.nlsearch captions RUN_DIR [--out data/splits/ac21_captions.csv]
      説明文の記録（vlmlab run の calls.jsonl）から、テキストの埋め込みの一覧を作る
  python -m vlmlab.search.nlsearch queries [--template data/splits/ac21_queries.csv]
      書き手が書いたクエリの文から、テキストの埋め込みの一覧を作る
  python -m vlmlab.search.nlsearch evaluate --captions RUN --images RUN --query-a RUN --query-b RUN --out DIR

正解（CI-S10）：クエリと同じ品目・同じ欠陥の種類・同じ区画（厳しい）。参考に、位置を問わない（ゆるい）正解も数える。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..run import load_jsonl
from .index import chance_hit_at_k, chance_precision, hit_at_k, precision_at_k, reciprocal_rank, top_k
from .store import load_vectors

CELLS = ["左上", "上", "右上", "左", "中央", "右", "左下", "下", "右下"]  # 行ごと（上の行から）


def cell_counts(mask: np.ndarray) -> np.ndarray:
    """マスク（0 以外が欠陥）を縦横3等分し、区画ごとの欠陥の画素数を返す（長さ9、CELLS の順）。

    画素 (y, x) の区画は 行 = floor(3y / H)、列 = floor(3x / W)。
    """
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    rows, cols = (3 * ys) // h, (3 * xs) // w
    return np.bincount(rows * 3 + cols, minlength=9)


def cell_of(mask: np.ndarray) -> tuple[str, float, int]:
    """欠陥の画素が一番多い区画（CI-S9）。同数なら CELLS の順で先。(区画, その区画の割合, 欠陥がある区画の数)。"""
    c = cell_counts(mask)
    total = int(c.sum())
    if total == 0:
        raise ValueError("mask has no defect pixels")
    i = int(np.argmax(c))  # 最初の最大
    return CELLS[i], float(c[i] / total), int((c > 0).sum())


def make_positions(split: pd.DataFrame, visa_root: Path) -> pd.DataFrame:
    idx = split[split.role == "index"].copy()
    out = []
    for r in idx.itertuples(index=False):
        if not r.mask_path:
            raise ValueError(f"{r.item_id} has no mask")
        cell, share, n_cells = cell_of(np.array(Image.open(visa_root / r.mask_path)))
        out.append({"item_id": r.item_id, "category": r.category, "defect_type": r.defect_type,
                    "cell": cell, "cell_share": round(share, 4), "n_cells": n_cells, "path": r.path})
    return pd.DataFrame(out)


def make_query_template(pos: pd.DataFrame, per_category: int = 3, n_total: int = 30,
                        min_relevant: int = 2) -> pd.DataFrame:
    """クエリの雛形（CI-S11）。品目×種類×区画の組で、正解（厳しい）が min_relevant 枚以上のものから、
    品目ごとに正解の多い順（同数なら種類・区画の名前順）に per_category 個。n_total に足りなければ、
    残りの組から正解の多い順（同数なら品目・種類・区画の順）に足す。"""
    g = pos.groupby(["category", "defect_type", "cell"]).size().rename("n_relevant_strict").reset_index()
    loose = pos.groupby(["category", "defect_type"]).size().rename("n_relevant_loose").reset_index()
    g = g[g.n_relevant_strict >= min_relevant].merge(loose, on=["category", "defect_type"])
    g["_cell_order"] = g.cell.map(CELLS.index)
    g = g.sort_values(["category", "n_relevant_strict", "defect_type", "_cell_order"],
                      ascending=[True, False, True, True], kind="stable")
    first = g.groupby("category", sort=True).head(per_category)
    rest = g.drop(index=first.index).sort_values(
        ["n_relevant_strict", "category", "defect_type", "_cell_order"], ascending=[False, True, True, True],
        kind="stable").head(max(0, n_total - len(first)))
    t = pd.concat([first, rest]).sort_values(["category", "n_relevant_strict", "defect_type", "_cell_order"],
                                             ascending=[True, False, True, True], kind="stable")
    t = t.drop(columns="_cell_order").reset_index(drop=True)
    t.insert(0, "query_id", [f"q{i + 1:02d}" for i in range(len(t))])
    t["query_text"] = ""  # 書き手が書く
    return t


def captions_manifest(calls_jsonl: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """説明文の記録から、テキストの埋め込みの一覧（item_id, modality, text）を作る（CI-S12）。

    使わない行：API のエラー、本文が空、stopReason が end_turn でない（途中で切れた）。
    同じ画像の記録が複数あれば、最後の有効な行を使う（再開で呼び直したもの）。
    """
    ok, bad = {}, []
    for r in load_jsonl(calls_jsonl):
        text = (r.get("raw_text") or "").strip()
        if r.get("api_error") or not text or r.get("stop_reason") != "end_turn":
            bad.append({"item_id": r["image_id"], "reason": r.get("api_error") or r.get("stop_reason") or "empty"})
            continue
        ok[r["image_id"]] = text
    m = pd.DataFrame({"item_id": list(ok), "modality": "text", "text": list(ok.values())})
    bad_df = pd.DataFrame(bad, columns=["item_id", "reason"])
    return m, bad_df[~bad_df.item_id.isin(ok)]


def queries_manifest(template: pd.DataFrame) -> pd.DataFrame:
    empty = template[template.query_text.fillna("").str.strip() == ""]
    if len(empty):
        raise ValueError(f"{len(empty)} queries have no text yet (書き手が query_text を書く): {list(empty.query_id)[:5]}")
    return pd.DataFrame({"item_id": template.query_id, "modality": "text", "text": template.query_text.str.strip()})


def evaluate(pos: pd.DataFrame, template: pd.DataFrame, emb: dict[str, Path], k: int = 5
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """方法 A（クエリ TEXT_RETRIEVAL → 説明文）と B（クエリ IMAGE_RETRIEVAL → 画像）を数える（CI-S10）。

    索引は両方とも同じ 300 枚（説明文が作れなかった画像は、A では候補に無いまま数え、件数を表に書く）。
    """
    meta = pos.set_index("item_id")
    rows, per_query = [], []
    for method, (q_key, i_key) in {"A_caption": ("query_a", "captions"), "B_image": ("query_b", "images")}.items():
        q_ids, q_vec = load_vectors(emb[q_key], list(template.query_id))
        recs = {r["item_id"] for r in load_jsonl(emb[i_key]) if r.get("vector") is not None}
        ids = [i for i in pos.item_id if i in recs]
        _, vec = load_vectors(emb[i_key], ids)
        all_hits = top_k(q_vec, vec, ids, k)
        for t, hits in zip(template.itertuples(index=False), all_hits):
            strict = set(pos[(pos.category == t.category) & (pos.defect_type == t.defect_type)
                             & (pos.cell == t.cell)].item_id)
            loose = set(pos[(pos.category == t.category) & (pos.defect_type == t.defect_type)].item_id)
            for h in hits:
                rows.append({"method": method, "query_id": t.query_id, "query_text": t.query_text, "rank": h.rank,
                             "item_id": h.item_id, "score": round(h.score, 6),
                             "hit_category": meta.at[h.item_id, "category"],
                             "hit_type": meta.at[h.item_id, "defect_type"], "hit_cell": meta.at[h.item_id, "cell"],
                             "strict": h.item_id in strict, "loose": h.item_id in loose})
            per_query.append({
                "method": method, "query_id": t.query_id, "category": t.category,
                "n_index": len(ids), "n_relevant_strict": len(strict),
                "hit_at_1_strict": hit_at_k(hits, strict, 1), f"hit_at_{k}_strict": hit_at_k(hits, strict, k),
                f"precision_at_{k}_strict": precision_at_k(hits, strict, k), "rr_strict": reciprocal_rank(hits, strict),
                f"hit_at_{k}_loose": hit_at_k(hits, loose, k), f"precision_at_{k}_loose": precision_at_k(hits, loose, k),
                f"same_category_at_{k}": sum(meta.at[h.item_id, "category"] == t.category for h in hits[:k]) / k,
                f"chance_hit_at_{k}_strict": chance_hit_at_k(len(strict), len(pos), k),
                f"chance_precision_at_{k}_strict": chance_precision(len(strict), len(pos)),
            })
    pq = pd.DataFrame(per_query)
    metrics = [c for c in pq.columns if c not in ("method", "query_id", "category", "n_index", "n_relevant_strict")]
    summ = pq.groupby("method", sort=False)[metrics].mean().reset_index()
    summ.insert(1, "n_queries", pq.groupby("method", sort=False).size().values)
    summ.insert(2, "n_index", pq.groupby("method", sort=False).n_index.first().values)
    return pd.DataFrame(rows), pd.concat([summ, pq], ignore_index=True, sort=False)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m vlmlab.search.nlsearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("positions")
    s.add_argument("--visa-root", required=True)
    s.add_argument("--split", default="data/splits/ac20_simsearch.csv")
    s.add_argument("--out", default="data/splits")
    q = sub.add_parser("query-template")
    q.add_argument("--positions", default="data/splits/ac21_positions.csv")
    q.add_argument("--per-category", type=int, default=3)
    q.add_argument("--n-total", type=int, default=30)
    q.add_argument("--out", default="data/splits/ac21_queries.csv")
    c = sub.add_parser("captions")
    c.add_argument("run_dir")
    c.add_argument("--out", default="data/splits/ac21_captions.csv")
    m = sub.add_parser("queries")
    m.add_argument("--template", default="data/splits/ac21_queries.csv")
    m.add_argument("--out", default="data/splits/ac21_query_texts.csv")
    e = sub.add_parser("evaluate")
    for k_ in ("captions", "images", "query-a", "query-b"):
        e.add_argument(f"--{k_}", required=True, help="runs/<name>（embeddings.jsonl のあるところ）")
    e.add_argument("--positions", default="data/splits/ac21_positions.csv")
    e.add_argument("--template", default="data/splits/ac21_queries.csv")
    e.add_argument("--out", required=True)
    e.add_argument("--k", type=int, default=5)
    a = p.parse_args(argv)
    if a.cmd == "positions":
        split = pd.read_csv(a.split, keep_default_na=False)
        pos = make_positions(split, Path(a.visa_root))
        out = Path(a.out)
        pos.drop(columns="path").to_csv(out / "ac21_positions.csv", index=False, lineterminator="\n")
        # 説明文づくり（vlmlab run）の一覧：image_id, category, label, path
        pos.rename(columns={"item_id": "image_id"}).assign(label="anomaly")[
            ["image_id", "category", "label", "path"]].to_csv(out / "ac21_caption_300.csv", index=False,
                                                              lineterminator="\n")
        print(pos.cell.value_counts().to_string())
    elif a.cmd == "query-template":
        t = make_query_template(pd.read_csv(a.positions, keep_default_na=False), a.per_category, a.n_total)
        t.to_csv(a.out, index=False, lineterminator="\n")
        print(f"{len(t)} queries -> {a.out}")
    elif a.cmd == "captions":
        m_, bad = captions_manifest(Path(a.run_dir) / "calls.jsonl")
        m_.to_csv(a.out, index=False, lineterminator="\n")
        print(json.dumps({"captions": len(m_), "unusable": bad.to_dict("records")}, ensure_ascii=False))
    elif a.cmd == "queries":
        queries_manifest(pd.read_csv(a.template, keep_default_na=False)).to_csv(a.out, index=False,
                                                                               lineterminator="\n")
    else:
        emb = {"captions": a.captions, "images": a.images, "query_a": a.query_a, "query_b": a.query_b}
        res, summ = evaluate(pd.read_csv(a.positions, keep_default_na=False),
                             pd.read_csv(a.template, keep_default_na=False),
                             {k_: Path(v) / "embeddings.jsonl" for k_, v in emb.items()}, a.k)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "nl_search_results.csv", index=False, lineterminator="\n")
        summ.to_csv(out / "nl_search_summary.csv", index=False, lineterminator="\n", float_format="%.4f")
        print(summ[summ.query_id.isna()].to_string(index=False))


if __name__ == "__main__":
    main()
