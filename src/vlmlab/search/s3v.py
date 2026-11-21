"""手元の埋め込み（embeddings.jsonl）を Amazon S3 Vectors に入れ、検索して手元（numpy）の結果と比べる。

  python -m vlmlab.search.s3v create-bucket --bucket NAME
  python -m vlmlab.search.s3v load --bucket NAME --index NAME --emb RUN_DIR --meta CSV
        [--where COL=V1,V2] [--meta-cols A,B] [--text-col COL]
      埋め込みを 500 件ずつ PutVectors で入れる（同じ key は上書きなので、流し直してよい）
  python -m vlmlab.search.s3v compare --bucket NAME --index NAME --emb RUN_DIR --query RUN_DIR --meta CSV
        [--where COL=V1,V2] [--k 5] --out CSV
      クエリごとに QueryVectors（topK k）と手元の top_k を比べる

S3 Vectors の上限（2026-11-21 確認、https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html）：
次元 1〜4,096、PutVectors は1回 500 件・20 MiB まで、QueryVectors の topK は 10,000 まで。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .index import top_k
from .store import load_vectors

REGION = "ap-northeast-1"
MAX_BATCH = 500
TEXT_KEY = "text"  # 絞り込みに使わないメタデータ（本文）


def select_rows(meta: pd.DataFrame, where: dict[str, list[str]] | None) -> pd.DataFrame:
    if not where:
        return meta
    mask = pd.Series(True, index=meta.index)
    for col, vals in where.items():
        if col not in meta.columns:
            raise ValueError(f"meta has no column {col!r}")
        mask &= meta[col].astype(str).isin(vals)
    return meta[mask]


def build_records(ids: list[str], vectors: np.ndarray, meta: pd.DataFrame, meta_cols: list[str],
                  text_col: str | None = None) -> list[dict[str, Any]]:
    """PutVectors の vectors（CI-S17）。key = item_id、data は float32、メタデータは指定した列だけ（文字列）。"""
    if len(ids) != len(vectors) or len(set(ids)) != len(ids):
        raise ValueError("ids must be unique and match the vectors")
    if vectors.ndim != 2 or not 1 <= vectors.shape[1] <= 4096:
        raise ValueError("vectors must be 2-D with dimension 1..4096")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("vectors must be finite")
    m = meta.set_index("item_id")
    missing = [i for i in ids if i not in m.index]
    if missing:
        raise ValueError(f"{len(missing)} ids are not in meta: {missing[:5]}")
    out = []
    for i, v in zip(ids, vectors):
        md = {c: str(m.at[i, c]) for c in meta_cols}
        if text_col:
            md[TEXT_KEY] = str(m.at[i, text_col])
        out.append({"key": i, "data": {"float32": [float(x) for x in np.asarray(v, dtype=np.float32)]},
                    "metadata": md})
    return out


def batches(items: list[Any], size: int = MAX_BATCH) -> list[list[Any]]:
    if not 1 <= size <= MAX_BATCH:
        raise ValueError(f"batch size must be 1..{MAX_BATCH}")
    return [items[i:i + size] for i in range(0, len(items), size)]


def put_bytes(records: list[dict[str, Any]]) -> int:
    """書き込みのデータ量の目安（費用の見積り用）：ベクトル 4 バイト × 次元 ＋ key とメタデータの UTF-8 の長さ。"""
    n = 0
    for r in records:
        n += 4 * len(r["data"]["float32"]) + len(r["key"].encode("utf-8"))
        n += sum(len(k.encode("utf-8")) + len(v.encode("utf-8")) for k, v in r["metadata"].items())
    return n


def compare_hits(local: list[tuple[str, float]], remote: list[tuple[str, float]]) -> dict[str, Any]:
    """手元と S3 Vectors の上位 k 件を比べる（CI-S18）。remote の値はコサイン距離なので、類似度 = 1 − 距離。

    same_order：key の並びが完全に同じ。same_set：集合が同じ。overlap：共通の件数。
    max_abs_score_diff：共通の key での類似度の差の最大（float32 で保存されるので 1e-6 程度の差はありうる）。
    """
    lk, rk = [k for k, _ in local], [k for k, _ in remote]
    rs = {k: 1.0 - d for k, d in remote}
    common = [k for k, _ in local if k in rs]
    diff = max((abs(s - rs[k]) for k, s in local if k in rs), default=None)
    return {"same_order": lk == rk, "same_set": set(lk) == set(rk), "overlap": len(common),
            "max_abs_score_diff": diff}


class S3Vectors:
    def __init__(self, bucket: str, region: str = REGION, client: Any = None):
        if client is None:
            import boto3
            client = boto3.client("s3vectors", region_name=region)
        self.c, self.bucket = client, bucket

    def create_bucket(self) -> None:
        self.c.create_vector_bucket(vectorBucketName=self.bucket)

    def ensure_index(self, index: str, dimension: int, non_filterable: list[str]) -> bool:
        """無ければ作る（cosine、float32）。作ったら True。"""
        names = {i["indexName"] for i in self.c.list_indexes(vectorBucketName=self.bucket).get("indexes", [])}
        if index in names:
            got = self.c.get_index(vectorBucketName=self.bucket, indexName=index)["index"]
            if got["dimension"] != dimension or got["distanceMetric"] != "cosine":
                raise RuntimeError(f"index {index} exists with dimension {got['dimension']} / {got['distanceMetric']}")
            return False
        kw: dict[str, Any] = {"vectorBucketName": self.bucket, "indexName": index, "dataType": "float32",
                              "dimension": dimension, "distanceMetric": "cosine"}
        if non_filterable:
            kw["metadataConfiguration"] = {"nonFilterableMetadataKeys": non_filterable}
        self.c.create_index(**kw)
        return True

    def put(self, index: str, records: list[dict[str, Any]], log: Callable[[str], None] = print) -> int:
        n = 0
        for b in batches(records):
            self.c.put_vectors(vectorBucketName=self.bucket, indexName=index, vectors=b)
            n += 1
            log(f"put {len(b)} vectors into {index}")
        return n

    def query(self, index: str, vector: np.ndarray, k: int) -> tuple[list[tuple[str, float]], float]:
        t0 = time.perf_counter()
        r = self.c.query_vectors(vectorBucketName=self.bucket, indexName=index, topK=k,
                                 queryVector={"float32": [float(x) for x in np.asarray(vector, dtype=np.float32)]},
                                 returnDistance=True, returnMetadata=False)
        return [(v["key"], float(v["distance"])) for v in r["vectors"]], time.perf_counter() - t0


def _where(args: list[str] | None) -> dict[str, list[str]]:
    out = {}
    for a in args or []:
        col, vals = a.split("=", 1)
        out[col] = vals.split(",")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m vlmlab.search.s3v")
    sub = p.add_subparsers(dest="cmd", required=True)
    cb = sub.add_parser("create-bucket")
    cb.add_argument("--bucket", required=True)
    for name in ("load", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--bucket", required=True)
        s.add_argument("--index", required=True)
        s.add_argument("--emb", required=True, help="索引の runs/<name>")
        s.add_argument("--meta", required=True, help="item_id の列がある CSV")
        s.add_argument("--where", action="append", help="COL=V1,V2（その値の行だけ）")
        if name == "load":
            s.add_argument("--meta-cols", default="")
            s.add_argument("--text-col")
            s.add_argument("--out", help="入れた件数・データ量の記録（JSON）")
        else:
            s.add_argument("--query", required=True, help="クエリの runs/<name>")
            s.add_argument("--k", type=int, default=5)
            s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    store = S3Vectors(a.bucket)
    if a.cmd == "create-bucket":
        store.create_bucket()
        print(f"created vector bucket {a.bucket} in {REGION}")
        return
    meta = select_rows(pd.read_csv(a.meta, keep_default_na=False, dtype=str), _where(a.where))
    ids, vec = load_vectors(Path(a.emb) / "embeddings.jsonl", list(meta.item_id))
    if a.cmd == "load":
        cols = [c for c in a.meta_cols.split(",") if c]
        recs = build_records(ids, vec, meta, cols, a.text_col)
        created = store.ensure_index(a.index, vec.shape[1], [TEXT_KEY] if a.text_col else [])
        n_req = store.put(a.index, recs)
        info = {"bucket": a.bucket, "index": a.index, "created": created, "vectors": len(recs),
                "put_requests": n_req, "put_bytes": put_bytes(recs), "dimension": int(vec.shape[1])}
        print(json.dumps(info, ensure_ascii=False))
        if a.out:
            Path(a.out).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    q_ids, q_vec = load_vectors(Path(a.query) / "embeddings.jsonl")
    local = top_k(q_vec, vec, ids, a.k)
    rows = []
    for qid, qv, lh in zip(q_ids, q_vec, local):
        remote, lat = store.query(a.index, qv, a.k)
        c = compare_hits([(h.item_id, h.score) for h in lh], remote)
        rows.append({"index": a.index, "query_id": qid, "local": ";".join(h.item_id for h in lh),
                     "s3vectors": ";".join(k for k, _ in remote), **c, "latency_s": round(lat, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, lineterminator="\n")
    print(json.dumps({"index": a.index, "queries": len(df), "same_order": int(df.same_order.sum()),
                      "same_set": int(df.same_set.sum()), "overlap_mean": float(df.overlap.mean()),
                      "max_abs_score_diff": float(df.max_abs_score_diff.max()),
                      "latency_p50_s": float(df.latency_s.median()),
                      "latency_p95_s": float(df.latency_s.quantile(0.95))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
