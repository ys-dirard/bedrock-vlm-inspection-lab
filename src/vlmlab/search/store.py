"""一覧（manifest の CSV）の各行を埋め込み、runs/<name>/embeddings.jsonl に1件1行で残す。

- 一覧の列: item_id（一意）、modality（image / text）、image なら detail（STANDARD_IMAGE / DOCUMENT_IMAGE）と
  path（--data-root からの相対パス）、text なら text。ほかの列は記録しない（集計で一覧と突き合わせる）
- 再実行では、埋め込みが取れた item_id を飛ばす。設定（モデル・purpose・次元・縮小）や中身（SHA-256）が
  前の記録と違えば、呼ぶ前に止める（CI-S5）
- 費用は単価表（pricing.yaml の embeddings）から出す（CI-S3）。累計が budget_usd を超えそうなら呼ばずに止める
- 試運転（--limit N）は一覧の先頭 N 行だけを、別の名前（<name>__trialN）で回す
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from ..cost import BudgetExceeded, BudgetGuard
from ..data import load_image
from ..run import JST, load_jsonl
from .nova import DETAILS, DIMENSIONS, PURPOSES, EmbedInput, Embedder

PER = 1_000_000


@dataclass
class EmbedRunConfig:
    name: str
    manifest: str
    model_id: str
    region: str
    purpose: str
    dimension: int
    budget_usd: float
    image_long_edge: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "EmbedRunConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {cfg.purpose!r}")
        if cfg.dimension not in DIMENSIONS:
            raise ValueError(f"dimension must be one of {DIMENSIONS}")
        return cfg

    def identity(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "purpose": self.purpose, "dimension": self.dimension,
                "image_long_edge": self.image_long_edge}


@dataclass(frozen=True)
class EmbedPrices:
    per_image_usd: float | None
    per_document_image_usd: float | None
    per_mtok_text_usd: float | None

    @classmethod
    def load(cls, pricing_path: str | Path, model_id: str) -> "EmbedPrices":
        doc = yaml.safe_load(Path(pricing_path).read_text(encoding="utf-8")) or {}
        for e in doc.get("embeddings", []):
            if e.get("model_id") == model_id:
                vals = [e.get("per_image_usd"), e.get("per_document_image_usd"), e.get("per_mtok_text_usd")]
                if any(v is not None and float(v) < 0 for v in vals):
                    raise ValueError("prices must be >= 0")
                return cls(*[None if v is None else float(v) for v in vals])
        raise KeyError(f"pricing.yaml has no embeddings entry for {model_id!r}")

    def require(self, modalities: set[tuple[str, str | None]]) -> None:
        """使う種類の単価が null なら、呼ぶ前に止める（CI-C5 と同じ考え方）。"""
        for modality, detail in modalities:
            if modality == "text" and self.per_mtok_text_usd is None:
                raise RuntimeError("per_mtok_text_usd is null in pricing.yaml; confirm it first")
            if (modality, detail) == ("image", "STANDARD_IMAGE") and self.per_image_usd is None:
                raise RuntimeError("per_image_usd is null in pricing.yaml; confirm it first")
            if (modality, detail) == ("image", "DOCUMENT_IMAGE") and self.per_document_image_usd is None:
                raise RuntimeError("per_document_image_usd is null in pricing.yaml; confirm it first")


def embed_cost(prices: EmbedPrices, modality: str, detail: str | None, responded: bool,
               input_tokens: int | None) -> float | None:
    """1件の費用（CI-S3）。

    - 応答が返らなかった（API のエラー）なら 0
    - 画像は1枚あたりの単価（STANDARD_IMAGE と DOCUMENT_IMAGE で別）。埋め込みの形が不正でも、応答が返れば1枚分
    - テキストは HTTP ヘッダーの入力トークン数 × 100万トークンあたりの単価。トークン数が無ければ None（費用不明）
    """
    if not responded:
        return 0.0
    if modality == "image":
        p = prices.per_image_usd if detail == "STANDARD_IMAGE" else prices.per_document_image_usd
        if detail not in DETAILS or p is None:
            raise ValueError(f"no price for image detail {detail!r}")
        return p
    if modality == "text":
        if input_tokens is None or prices.per_mtok_text_usd is None:
            return None
        return input_tokens * prices.per_mtok_text_usd / PER
    raise ValueError(f"unknown modality {modality!r}")


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"item_id": str}, keep_default_na=False)
    for col in ("item_id", "modality"):
        if col not in df.columns:
            raise ValueError(f"manifest lacks column {col!r}")
    if df["item_id"].duplicated().any():
        raise ValueError(f"item_id is not unique in {path}")
    bad = set(df["modality"]) - {"image", "text"}
    if bad:
        raise ValueError(f"unknown modality {sorted(bad)}")
    img = df[df.modality == "image"]
    if len(img) and ("detail" not in df.columns or "path" not in df.columns):
        raise ValueError("image rows need detail and path columns")
    if len(img) and not set(img["detail"]) <= set(DETAILS):
        raise ValueError(f"detail must be one of {DETAILS}")
    if (df.modality == "text").any() and "text" not in df.columns:
        raise ValueError("text rows need a text column")
    return df


def load_input(row: Any, data_root: Path, long_edge: int | None) -> EmbedInput:
    if row.modality == "text":
        return EmbedInput("text", text=row.text)
    img = load_image(data_root / row.path, long_edge)
    return EmbedInput("image", data=img.data, format=img.format, detail=row.detail)


def content_sha(inp: EmbedInput) -> str:
    raw = inp.text.encode("utf-8") if inp.modality == "text" else (inp.data or b"")
    return hashlib.sha256(raw).hexdigest()


def execute_embed(cfg: EmbedRunConfig, embedder: Embedder, prices: EmbedPrices, data_root: Path,
                  base_dir: Path, runs_dir: Path, limit: int | None = None,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    manifest = read_manifest(base_dir / cfg.manifest)
    run_name = cfg.name
    if limit is not None:
        manifest, run_name = manifest.head(limit), f"{cfg.name}__trial{limit}"
    detail_col = manifest["detail"] if "detail" in manifest.columns else pd.Series([None] * len(manifest))
    prices.require({(m, d if m == "image" else None) for m, d in zip(manifest.modality, detail_col)})
    out = runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / "embeddings.jsonl"
    previous = load_jsonl(emb_path)
    ident = cfg.identity()
    for r in previous:  # CI-S5
        diff = {k: (r.get(k), v) for k, v in ident.items() if r.get(k) != v}
        if diff:
            raise RuntimeError(f"config differs from existing embeddings in {out}: {diff}")
    have = {r["item_id"]: r for r in previous if r.get("vector") is not None}
    guard = BudgetGuard.resume(cfg.budget_usd, [r.get("cost_usd") for r in previous])
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.__dict__, allow_unicode=True, sort_keys=False),
                                     encoding="utf-8")
    run_id = datetime.now(JST).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    stopped, n_called = None, 0
    with emb_path.open("a", encoding="utf-8") as f:
        for row in manifest.itertuples(index=False):
            inp = load_input(row, data_root, cfg.image_long_edge)
            sha = content_sha(inp)
            if row.item_id in have:
                if have[row.item_id].get("content_sha256") != sha:  # CI-S5
                    raise RuntimeError(f"content of {row.item_id} changed since it was embedded")
                continue
            try:
                guard.check_before_call()
            except BudgetExceeded as e:
                stopped = str(e)
                break
            called_at = datetime.now(JST).isoformat(timespec="seconds")
            res = embedder.embed(inp)
            detail = inp.detail if inp.modality == "image" else None
            cost = embed_cost(prices, inp.modality, detail, res.latency_s is not None, res.input_tokens)
            rec = {"run_name": run_name, "run_id": run_id, "item_id": row.item_id, "modality": inp.modality,
                   "detail": detail, **ident, "content_sha256": sha, "vector": res.vector,
                   "input_tokens": res.input_tokens, "latency_s": res.latency_s, "attempts": res.attempts,
                   "api_error": res.api_error, "cost_usd": cost, "header_names": res.header_names,
                   "timestamp": called_at}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_called += 1
            if res.vector is not None:
                have[row.item_id] = rec
            if res.api_error:
                log(f"[api_error] {row.item_id}: {res.api_error}")
            if cost is None:  # 費用が分からないまま続けない
                stopped = f"cost unknown for {row.item_id} (no token count header)"
                break
            guard.add(cost)
    all_recs = load_jsonl(emb_path)
    lat = [r["latency_s"] for r in all_recs if r.get("latency_s") is not None]
    summary = {
        "run_name": run_name, "run_id": run_id, "called": n_called, "stopped": stopped,
        "embedded": len(have), "planned": len(manifest),
        "api_errors": sum(1 for r in all_recs if r.get("api_error")),
        "cost_known": all(r.get("cost_usd") is not None for r in all_recs),
        "spent_usd": sum(r["cost_usd"] for r in all_recs if r.get("cost_usd") is not None),
        "input_tokens_sum": sum(r["input_tokens"] or 0 for r in all_recs),
        "latency_p50_s": float(np.percentile(lat, 50)) if lat else None,
        "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))
    return summary


def load_vectors(emb_path: Path, item_ids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """embeddings.jsonl から (item_id の並び, 行列) を返す。item_ids を渡せばその順で、欠けていれば止める。"""
    recs = {r["item_id"]: r for r in load_jsonl(emb_path) if r.get("vector") is not None}
    ids = list(recs) if item_ids is None else list(item_ids)
    missing = [i for i in ids if i not in recs]
    if missing:
        raise ValueError(f"{len(missing)} items have no embedding in {emb_path}: {missing[:5]}")
    return ids, np.array([recs[i]["vector"] for i in ids], dtype=float)
