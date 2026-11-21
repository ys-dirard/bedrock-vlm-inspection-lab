"""請求書の抽出を回す（CI-IR5〜IR8）。記録は1行1呼び出しの JSONL（runs/invoice/<name>/calls.jsonl）。

採点は記録の raw_text から report.py で行う（呼び直さずにやり直せる）。
"""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image

from ..bedrock import ModelRequest, ModelResponse
from ..cost import BudgetGuard, PricingTable, call_cost
from ..data import EncodedImage
from .generate import SPLIT_PATH
from .render import png_bytes, split_halves
from .schema import PROMPT_VERSION, build_text, output_config, text_sha256

JST = timezone(timedelta(hours=9))
CONDITIONS = ("H-SO", "H-P", "N", "N-T", "Q")


@dataclass
class InvoiceConfig:
    name: str
    condition: str
    model_id: str
    region: str
    budget_usd: float
    structured: bool = False
    split_halves: bool = False
    variants: list[str] = field(default_factory=lambda: ["clean", "degraded"])
    repeat: int = 1
    split: str = SPLIT_PATH
    inference: dict[str, Any] = field(default_factory=lambda: {"temperature": 0, "maxTokens": 2000})

    @classmethod
    def load(cls, path: str | Path) -> "InvoiceConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if unknown := set(d) - set(cls.__dataclass_fields__):
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if not set(self.variants) <= {"clean", "degraded"} or not self.variants:
            raise ValueError("variants must be a non-empty subset of [clean, degraded]")
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")
        if set(self.inference) - {"temperature", "maxTokens", "topP"}:
            raise ValueError("inference accepts temperature, maxTokens, topP only")


def load_split(path: Path, variants: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["variant"] in variants]


def select_trial(rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    """試運転：一覧の順に等間隔で limit 件（CI-IR5）。一覧はきれい60 → 劣化60 の順なので、2件なら同じ請求書のきれい・劣化になる。"""
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("limit must be > 0")
    idx = sorted({int(i * len(rows) / limit) for i in range(limit)})
    return [rows[i] for i in idx]


def encode_images(path: Path, halves: bool) -> list[EncodedImage]:
    """1件ぶんの画像。halves なら上下2枚の PNG（劣化の JPEG もいったん復号して PNG にする。CI-IR6）。"""
    if not halves:
        fmt = "png" if path.suffix.lower() == ".png" else "jpeg"
        with Image.open(path) as im:
            w, h = im.size
        return [EncodedImage(data=path.read_bytes(), format=fmt, width=w, height=h)]
    with Image.open(path) as im:
        im = im.convert("RGB")
        top, bottom = split_halves(im, overlap=100)
    return [EncodedImage(data=png_bytes(p), format="png", width=p.size[0], height=p.size[1]) for p in (top, bottom)]


def identity(cfg: InvoiceConfig, text: str, split_sha: str) -> dict[str, Any]:
    """呼び出しの中身を決める設定。再開のとき前の記録と違えば止める（CI-IR7）。"""
    return {
        "condition": cfg.condition, "model_id": cfg.model_id, "structured": cfg.structured,
        "split_halves": cfg.split_halves, "repeat": cfg.repeat, "prompt_version": PROMPT_VERSION,
        "text_sha256": text_sha256(text), "split_sha256": split_sha,
        "temperature": float(cfg.inference.get("temperature", 0)),
        "max_tokens": int(cfg.inference.get("maxTokens", 2000)),
        "top_p": cfg.inference.get("topP"),
    }


def build_request(cfg: InvoiceConfig, images: list[EncodedImage], text: str) -> ModelRequest:
    return ModelRequest(
        model_id=cfg.model_id, images=images, text=text,
        temperature=float(cfg.inference.get("temperature", 0)),
        max_tokens=int(cfg.inference.get("maxTokens", 2000)),
        top_p=cfg.inference.get("topP"), layout="refs_first",
        output_config=output_config() if cfg.structured else None,
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def execute(cfg: InvoiceConfig, base_dir: Path, runs_dir: Path, pricing: PricingTable, model: Any,
            limit: int | None = None,
            image_encoder: Callable[[Path, bool], list[EncodedImage]] = encode_images) -> Path:
    """1件ずつ Converse で呼ぶ（CI-IR8）。成功した doc_id は飛ばして再開する。予算を超える前に止める（CI-C4）。"""
    price = pricing.get(cfg.model_id, "standard")
    if not price.runnable:
        raise ValueError(f"price for {cfg.model_id} is unknown; refuse to run")
    split_path = base_dir / cfg.split
    rows = select_trial(load_split(split_path, cfg.variants), limit)
    text = build_text(cfg.structured, cfg.split_halves)
    ident = identity(cfg, text, hashlib.sha256(split_path.read_bytes()).hexdigest())

    name = cfg.name if limit is None else f"{cfg.name}__trial{limit}"
    out_dir = runs_dir / "invoice" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = out_dir / "calls.jsonl"
    previous = load_records(calls)
    for rec in previous:
        if diff := {k: (rec.get(k), v) for k, v in ident.items() if rec.get(k) != v}:
            raise ValueError(f"settings differ from previous records in this run directory: {diff}")
    guard = BudgetGuard.resume(cfg.budget_usd, [r["cost_usd"] for r in previous])
    done = {r["doc_id"] for r in previous if not r.get("api_error")}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")

    run_id = uuid.uuid4().hex[:12]
    seq = len(previous)
    with calls.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            if row["doc_id"] in done:
                continue
            guard.check_before_call()
            images = image_encoder(base_dir / row["path"], cfg.split_halves)
            resp: ModelResponse = model.invoke(build_request(cfg, images, text))
            cost = call_cost(price, resp.input_tokens, resp.output_tokens, resp.cache_read_tokens, resp.cache_write_tokens)
            rec = {
                "run_name": name, "run_id": run_id, "seq": seq, **ident,
                "doc_id": row["doc_id"], "invoice_id": row["invoice_id"], "variant": row["variant"],
                "n_images": len(images), "image_sizes": [[i.width, i.height] for i in images],
                "raw_text": resp.raw_text, "api_error": resp.api_error,
                "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "cache_read_tokens": resp.cache_read_tokens, "cache_write_tokens": resp.cache_write_tokens,
                "latency_s": resp.latency_s, "stop_reason": resp.stop_reason, "attempts": resp.attempts,
                "extra": resp.extra, "cost_usd": cost,
                "timestamp": datetime.now(JST).isoformat(timespec="seconds"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            seq += 1
            guard.add(cost)
    return out_dir
