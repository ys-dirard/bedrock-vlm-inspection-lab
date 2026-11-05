"""条件 E：学習用とテスト用の画像を埋め込み、品目ごとに k近傍法で判定する。

- 埋め込みは runs/<name>/embeddings.jsonl に1枚1行で残し、再実行では取得済みの画像を飛ばす（k を変えても呼び直さない）
- 判定は calls.jsonl（run.py と同じ形、repeat=1）に書くので、`vlmlab report` でそのまま集計できる
  - calls.jsonl の費用・トークン数・レイテンシは、テスト画像1枚の埋め込みの値（判定1件あたりの費用）
  - 学習用画像の埋め込み（索引づくり）の費用は summary.json の index_cost_usd に分けて書く
- 試運転（--limit N）はテスト画像 N 枚を埋め込むだけで、判定はしない
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .cost import BudgetExceeded, BudgetGuard
from .data import EncodedImage, load_image
from .embed import ImageEmbedder, knn_predict
from .run import JST, load_jsonl, select_limit


@dataclass
class EmbedConfig:
    name: str
    train_split: str
    test_split: str
    model_id: str
    region: str
    budget_usd: float
    embedding_purpose: str = "GENERIC_INDEX"
    embedding_dimension: int = 1024
    k: int = 3
    image_long_edge: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "EmbedConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def identity(self) -> dict[str, Any]:
        """埋め込みの中身を決める設定。再開のとき、前の記録と違えば止める（CI-E4）。k は含めない。"""
        return {"model_id": self.model_id, "embedding_purpose": self.embedding_purpose,
                "embedding_dimension": self.embedding_dimension, "image_long_edge": self.image_long_edge}


def load_embedding_price(pricing_path: str | Path, model_id: str) -> float:
    """pricing.yaml の embeddings から、標準の画像1枚あたりの単価（USD）。null なら実行しない（CI-C5 と同じ）。"""
    doc = yaml.safe_load(Path(pricing_path).read_text(encoding="utf-8")) or {}
    for e in doc.get("embeddings", []):
        if e.get("model_id") == model_id:
            if e.get("per_image_usd") is None:
                raise RuntimeError(f"per_image_usd for {model_id} is null in pricing.yaml; confirm it first")
            v = float(e["per_image_usd"])
            if v < 0:
                raise ValueError("per_image_usd must be >= 0")
            return v
    raise KeyError(f"pricing.yaml has no embeddings entry for {model_id!r}")


def embed_cost(responded: bool, per_image_usd: float) -> float:
    """費用 = 応答が返った呼び出し1回につき画像1枚分の単価。応答が無い（API のエラー）なら 0（CI-E2）。

    応答が返れば、埋め込みの形が不正（BadEmbedding）でも課金されたものとして数える。
    """
    return per_image_usd if responded else 0.0


def plan_images(cfg: EmbedConfig, base_dir: Path, limit: int | None) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(base_dir / cfg.test_split)
    train = pd.read_csv(base_dir / cfg.train_split)
    overlap = set(train.image_id) & set(test.image_id)
    if overlap:  # CI-E5
        raise ValueError(f"train and test overlap: {sorted(overlap)[:5]}")
    if limit is not None:
        return f"{cfg.name}__trial{limit}", select_limit(test, limit).assign(role="test"), test
    images = pd.concat([train.assign(role="train"), test.assign(role="test")], ignore_index=True)
    return cfg.name, images, test


def execute_embed(cfg: EmbedConfig, embedder: ImageEmbedder, per_image_usd: float, visa_root: Path,
                  base_dir: Path, runs_dir: Path, limit: int | None = None,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    run_name, images, test = plan_images(cfg, base_dir, limit)
    out = runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / "embeddings.jsonl"
    previous = load_jsonl(emb_path)
    ident = cfg.identity()
    for r in previous:
        diff = {k: (r.get(k), v) for k, v in ident.items() if r.get(k) != v}
        if diff:
            raise RuntimeError(f"config differs from existing embeddings in {out}: {diff}")
    have = {r["image_id"]: r for r in previous if r.get("vector") is not None}
    guard = BudgetGuard.resume(cfg.budget_usd, [r.get("cost_usd") for r in previous])
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg.__dict__, "parse": {}}, allow_unicode=True,
                                                    sort_keys=False), encoding="utf-8")
    run_id = datetime.now(JST).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    stopped, n_called = None, 0
    with emb_path.open("a", encoding="utf-8") as f:
        for row in images.itertuples(index=False):
            if row.image_id in have:
                continue
            try:
                guard.check_before_call()
            except BudgetExceeded as e:
                stopped = str(e)
                break
            img: EncodedImage = load_image(visa_root / row.path, cfg.image_long_edge)
            called_at = datetime.now(JST).isoformat(timespec="seconds")
            res = embedder.embed(img)
            # 画像は1枚あたりの課金。ヘッダーのトークン数（画像では 0）は記録だけして、費用には使わない（CI-E2）
            cost = embed_cost(res.latency_s is not None, per_image_usd)
            rec = {"run_name": run_name, "run_id": run_id, "image_id": row.image_id, "role": row.role,
                   "category": row.category, "label": row.label, "path": row.path, **ident,
                   "vector": res.vector, "input_tokens": res.input_tokens, "latency_s": res.latency_s,
                   "attempts": res.attempts, "api_error": res.api_error, "cost_usd": cost,
                   "header_names": res.header_names, "image_w": img.width, "image_h": img.height,
                   "timestamp": called_at}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_called += 1
            if res.vector is not None:
                have[row.image_id] = rec
            if res.api_error:
                log(f"[api_error] {row.image_id}: {res.api_error}")
            guard.add(cost)
    all_recs = load_jsonl(emb_path)
    summary: dict[str, Any] = {
        "run_name": run_name, "run_id": run_id, "called": n_called, "stopped": stopped,
        "embedded": len(have), "planned": len(images),
        "cost_known": all(r.get("cost_usd") is not None for r in all_recs),
        "spent_usd": sum(r["cost_usd"] for r in all_recs if r.get("cost_usd") is not None),
    }
    if limit is None and stopped is None and len(have) == len(images):
        summary.update(_classify(cfg, out, run_id, run_name, images, have))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))
    return summary


def _classify(cfg: EmbedConfig, out: Path, run_id: str, run_name: str, images: pd.DataFrame,
              have: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """品目ごとに k近傍法で判定し、calls.jsonl を作り直す（埋め込みから決まる派生データ）。"""
    records = []
    for cat, g in images.groupby("category", sort=True):
        tr, te = g[g.role == "train"], g[g.role == "test"]
        if len(te) == 0:
            continue
        x_tr = np.array([have[i]["vector"] for i in tr.image_id])
        x_te = np.array([have[i]["vector"] for i in te.image_id])
        preds = knn_predict(x_tr, list(tr.label), x_te, cfg.k)
        for row, (pred, p_anom) in zip(te.itertuples(index=False), preds):
            e = have[row.image_id]
            conf = p_anom if pred == "NG" else 1 - p_anom  # 業務指標のスコア（D1）が p_anom になる
            raw = json.dumps({"judgement": pred, "confidence": round(conf, 9), "p_anomaly": p_anom},
                             ensure_ascii=False)
            records.append({
                "run_name": run_name, "run_id": run_id, "repeat": 1, "image_id": row.image_id,
                "category": cat, "label": row.label, "path": row.path, "model_id": cfg.model_id,
                "service_tier_requested": "standard", "service_tier_resolved": None,
                "prompt_version": f"knn_k{cfg.k}_{cfg.embedding_purpose}_{cfg.embedding_dimension}",
                "raw_text": raw, "parsed": json.loads(raw), "parse_error": None,
                "input_tokens": e["input_tokens"], "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "latency_s": e["latency_s"], "stop_reason": None, "cost_usd": e["cost_usd"],
                "attempts": e["attempts"], "api_error": None, "server_latency_ms": None,
                "timestamp": e["timestamp"],
            })
    with (out / "calls.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    train_ids = set(images[images.role == "train"].image_id)
    test_ids = set(images[images.role == "test"].image_id)
    return {
        "classified": len(records),
        "index_cost_usd": sum(have[i]["cost_usd"] for i in train_ids),
        "test_cost_usd": sum(have[i]["cost_usd"] for i in test_ids),
    }
