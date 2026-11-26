"""手元（numpy）のコサイン類似度での上位 k 件の検索と、検索の指標（12/20・12/21・12/24 の記事）。

S3 Vectors などのベクトルストアは使わない。数百〜数千件なら全件との類似度を計算しても一瞬で済む。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("expected a 2-D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.all(np.isfinite(x)) or np.any(norms == 0):
        raise ValueError("vectors must be finite and non-zero")
    return x / norms


@dataclass(frozen=True)
class Hit:
    item_id: str
    rank: int  # 1 から
    score: float  # コサイン類似度


def top_k(queries: np.ndarray, index: np.ndarray, index_ids: list[str], k: int,
          exclude: list[set[str]] | None = None) -> list[list[Hit]]:
    """クエリごとに、索引の上位 k 件を返す（CI-S6）。

    - 類似度はコサイン類似度（両方を L2 正規化した内積）。大きい順
    - 同じ類似度なら、索引の並びで先のものを上にする（安定ソート）
    - exclude[i] に入っている item_id は、クエリ i の候補から除く（クエリ自身が索引にあるときなど）
    - 候補が k 件に満たなければ、あるだけ返す
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(index_ids) != len(index) or len(set(index_ids)) != len(index_ids):
        raise ValueError("index_ids must be unique and match the index rows")
    q, x = _normalize(queries), _normalize(index)
    if q.shape[1] != x.shape[1]:
        raise ValueError(f"dimension mismatch: {q.shape[1]} vs {x.shape[1]}")
    if exclude is not None and len(exclude) != len(q):
        raise ValueError("exclude must have one set per query")
    sims = q @ x.T
    out = []
    for i, row in enumerate(sims):
        order = np.argsort(-row, kind="stable")
        hits = []
        for j in order:
            if exclude is not None and index_ids[j] in exclude[i]:
                continue
            hits.append(Hit(index_ids[j], len(hits) + 1, float(row[j])))
            if len(hits) == k:
                break
        out.append(hits)
    return out


def hit_at_k(hits: list[Hit], relevant: set[str], k: int) -> int:
    """上位 k 件に正解が1件でもあれば 1、なければ 0（CI-S7）。"""
    return int(any(h.item_id in relevant for h in hits[:k]))


def precision_at_k(hits: list[Hit], relevant: set[str], k: int) -> float:
    """上位 k 件のうち正解の割合。分母は k（候補が k 件に満たなくても k）（CI-S7）。"""
    return sum(h.item_id in relevant for h in hits[:k]) / k


def reciprocal_rank(hits: list[Hit], relevant: set[str]) -> float:
    """最初の正解の順位の逆数。返した候補に正解が無ければ 0（CI-S7）。"""
    for h in hits:
        if h.item_id in relevant:
            return 1.0 / h.rank
    return 0.0


def chance_precision(n_relevant: int, n_candidates: int) -> float:
    """でたらめに k 件選んだときの precision@k の期待値 = 正解の数 / 候補の数（k によらない）（CI-S7）。"""
    if n_candidates <= 0 or not 0 <= n_relevant <= n_candidates:
        raise ValueError("need 0 <= n_relevant <= n_candidates and n_candidates > 0")
    return n_relevant / n_candidates


def chance_hit_at_k(n_relevant: int, n_candidates: int, k: int) -> float:
    """でたらめに k 件選んだとき、正解が1件以上入る確率 = 1 − C(N−R, k) / C(N, k)（CI-S7）。"""
    from math import comb

    if n_candidates <= 0 or not 0 <= n_relevant <= n_candidates or k < 1:
        raise ValueError("invalid arguments")
    k = min(k, n_candidates)
    return 1 - comb(n_candidates - n_relevant, k) / comb(n_candidates, k)
