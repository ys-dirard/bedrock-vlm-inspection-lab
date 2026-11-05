"""MMAD の実験を回す（オンデマンド）、バッチ推論の入力を作る、バッチの出力を記録にする（CI-MR1〜MR5、CI-MB1〜MB3）。

記録は1行1問の JSONL（runs/<name>/calls.jsonl）。オンデマンドとバッチで同じ形にし、集計（report.py）は区別しない。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from ..bedrock import ModelRequest, ModelResponse
from ..cost import BudgetGuard, PricingTable, call_cost
from ..data import EncodedImage
from .data import load_mmad_image, load_split
from .invoke import response_from_anthropic
from .prompt import anthropic_body, build_text, load_instruction, text_sha256
from .score import extract_answer

JST = timezone(timedelta(hours=9))
APIS = ("converse", "anthropic")
MODES = ("ondemand", "batch")
BATCH_MIN_RECORDS = 100  # Service Quotas「Minimum number of records per batch inference job」（2026-11-05、3モデルとも100）


@dataclass
class MmadConfig:
    name: str
    split: str
    model_id: str
    api: str  # converse（Nova）/ anthropic（Claude の Messages API）
    region: str
    budget_usd: float
    mode: str = "ondemand"
    n_shots: int = 1  # 1 なら random_templates[0] を見本にする。0 なら見本なし
    image_max: int = 512
    inference: dict[str, Any] = field(default_factory=lambda: {"temperature": 0, "maxTokens": 200})
    mmad_json: str = "data/mmad/mmad.json"
    images_root: str = "data/mmad/images"

    @classmethod
    def load(cls, path: str | Path) -> "MmadConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if unknown := set(d) - set(cls.__dataclass_fields__):
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.api not in APIS:
            raise ValueError(f"api must be one of {APIS}")
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if cfg.n_shots not in (0, 1):
            raise ValueError("n_shots must be 0 or 1 (MMAD has one random template per image in this setup)")
        if cfg.mode == "batch" and cfg.api != "anthropic":
            raise ValueError("batch mode is implemented only for api=anthropic")
        return cfg

    @property
    def pricing_tier(self) -> str:
        return "batch" if self.mode == "batch" else "standard"

    @property
    def temperature(self) -> float:
        return float(self.inference.get("temperature", 0))

    @property
    def max_tokens(self) -> int:
        return int(self.inference.get("maxTokens", 200))


def identity(cfg: MmadConfig, instruction: str, split_sha: str) -> dict[str, Any]:
    """呼び出しの中身を決める設定。再開のとき、前の記録と違えば止める（CI-MR2）。"""
    return {
        "model_id": cfg.model_id, "api": cfg.api, "mode": cfg.mode, "n_shots": cfg.n_shots,
        "image_max": cfg.image_max, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
        "instruction_sha256": text_sha256(instruction), "split_sha256": split_sha,
    }


def select_trial(split: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """試運転の問題：record_id の順に等間隔で limit 問（CI-MR1）。課題の順に並んだ一覧なので、課題が散らばる。"""
    if limit is None or limit >= len(split):
        return split
    if limit <= 0:
        raise ValueError("limit must be > 0")
    idx = sorted({int(i * len(split) / limit) for i in range(limit)})
    return split.iloc[idx]


class RequestBuilder:
    """1問ぶんの [画像…] と文を作る。画像は読み込んで縮小したものを覚えておく（見本は使い回されるため）。"""

    def __init__(self, cfg: MmadConfig, mmad: dict[str, Any], instruction: str, base_dir: Path,
                 image_loader: Callable[[str], EncodedImage] | None = None):
        self.cfg = cfg
        self.mmad = mmad
        self.instruction = instruction
        root = base_dir / cfg.images_root
        self._load = image_loader or (lambda key: load_mmad_image(root, key, cfg.image_max))
        self._cache: dict[str, EncodedImage] = {}

    def image(self, key: str) -> EncodedImage:
        if key not in self._cache:
            self._cache[key] = self._load(key)
        return self._cache[key]

    def parts(self, row: Any) -> tuple[list[EncodedImage], str]:
        conversation = self.mmad[row.image_key]["conversation"]
        text = build_text(self.instruction, conversation, int(row.q_index), self.cfg.n_shots)
        images = ([self.image(row.template)] if self.cfg.n_shots else []) + [self.image(row.image_key)]
        return images, text

    def anthropic(self, row: Any) -> dict[str, Any]:
        images, text = self.parts(row)
        return anthropic_body(images, text, self.cfg.max_tokens, self.cfg.temperature)

    def converse(self, row: Any) -> ModelRequest:
        images, text = self.parts(row)
        return ModelRequest(
            model_id=self.cfg.model_id, images=images, text=text,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens, layout="refs_first",
        )


def make_record(cfg: MmadConfig, ident: dict[str, Any], run_id: str, run_name: str, row: Any,
                resp: ModelResponse, cost: float | None, text: str | None) -> dict[str, Any]:
    pred = None if resp.api_error else extract_answer(resp.raw_text)
    return {
        "run_name": run_name, "run_id": run_id,
        "record_id": row.record_id, "qid": row.qid, "dataset": row.dataset, "task": row.task,
        "raw_type": row.raw_type, "q_index": int(row.q_index), "is_normal": bool(row.is_normal),
        "answer": row.answer, **ident,
        "text_sha256": text_sha256(text) if text is not None else None,
        "raw_text": resp.raw_text, "pred": pred, "api_error": resp.api_error,
        "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
        "cache_read_tokens": resp.cache_read_tokens, "cache_write_tokens": resp.cache_write_tokens,
        "latency_s": resp.latency_s, "stop_reason": resp.stop_reason, "attempts": resp.attempts,
        "extra": resp.extra, "cost_usd": cost, "timestamp": datetime.now(JST).isoformat(timespec="seconds"),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_identity(previous: list[dict[str, Any]], ident: dict[str, Any]) -> None:
    for rec in previous:
        diff = {k: (rec.get(k), v) for k, v in ident.items() if rec.get(k) != v}
        if diff:
            raise ValueError(f"settings differ from previous records in this run directory: {diff}")


def _split_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(cfg: MmadConfig, runs_dir: Path, limit: int | None) -> tuple[str, Path]:
    name = cfg.name if limit is None else f"{cfg.name}__trial{limit}"
    return name, runs_dir / name


def execute_ondemand(cfg: MmadConfig, base_dir: Path, runs_dir: Path, pricing: PricingTable,
                     model: Any, limit: int | None = None, mmad: dict[str, Any] | None = None,
                     image_loader: Callable[[str], EncodedImage] | None = None) -> Path:
    """オンデマンドで1問ずつ呼ぶ。成功した record_id は飛ばして再開する（CI-MR3）。予算を超える前に止める（CI-C4）。

    model は api=converse なら invoke(ModelRequest)、api=anthropic なら invoke(model_id, body) を持つもの。
    """
    if cfg.mode != "ondemand":
        raise ValueError("execute_ondemand requires mode=ondemand")
    price = pricing.get(cfg.model_id, cfg.pricing_tier)
    if not price.runnable:
        raise ValueError(f"price for {cfg.model_id}/{cfg.pricing_tier} is unknown; refuse to run")
    split_path = base_dir / cfg.split
    split = select_trial(load_split(split_path), limit)
    instruction = load_instruction(base_dir)
    ident = identity(cfg, instruction, _split_sha(split_path))
    if mmad is None:
        mmad = json.loads((base_dir / cfg.mmad_json).read_text(encoding="utf-8"))
    builder = RequestBuilder(cfg, mmad, instruction, base_dir, image_loader)

    name, out_dir = run_dir(cfg, runs_dir, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = out_dir / "calls.jsonl"
    previous = load_records(calls)
    check_identity(previous, ident)
    guard = BudgetGuard.resume(cfg.budget_usd, [r["cost_usd"] for r in previous])
    done = {r["record_id"] for r in previous if not r.get("api_error")}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")

    run_id = uuid.uuid4().hex[:12]
    with calls.open("a", encoding="utf-8") as f:
        for row in split.itertuples(index=False):
            if row.record_id in done:
                continue
            guard.check_before_call()
            if cfg.api == "converse":
                req = builder.converse(row)
                text = req.text
                resp = model.invoke(req)
            else:
                body = builder.anthropic(row)
                text = body["messages"][0]["content"][-1]["text"]
                resp = model.invoke(cfg.model_id, body)
            cost = call_cost(price, resp.input_tokens, resp.output_tokens, resp.cache_read_tokens, resp.cache_write_tokens)
            f.write(json.dumps(make_record(cfg, ident, run_id, name, row, resp, cost, text), ensure_ascii=False) + "\n")
            f.flush()
            guard.add(cost)
    return out_dir


def prepare_batch(cfg: MmadConfig, base_dir: Path, runs_dir: Path, mmad: dict[str, Any] | None = None,
                  image_loader: Callable[[str], EncodedImage] | None = None) -> dict[str, Any]:
    """バッチ推論の入力（1行1問、recordId = record_id、modelInput = Messages API の本文）を作る（CI-MB2）。

    件数が最小件数（100）未満なら作らない。manifest.json に設定と件数・ファイルの大きさを残す。
    """
    if cfg.mode != "batch":
        raise ValueError("prepare_batch requires mode=batch")
    split_path = base_dir / cfg.split
    split = load_split(split_path)
    if len(split) < BATCH_MIN_RECORDS:
        raise ValueError(f"batch needs at least {BATCH_MIN_RECORDS} records, got {len(split)}")
    instruction = load_instruction(base_dir)
    ident = identity(cfg, instruction, _split_sha(split_path))
    if mmad is None:
        mmad = json.loads((base_dir / cfg.mmad_json).read_text(encoding="utf-8"))
    builder = RequestBuilder(cfg, mmad, instruction, base_dir, image_loader)

    _, out_dir = run_dir(cfg, runs_dir, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "batch_input.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in split.itertuples(index=False):
            f.write(json.dumps({"recordId": row.record_id, "modelInput": builder.anthropic(row)}) + "\n")
    manifest = {**ident, "name": cfg.name, "n_records": len(split), "input_file": path.name,
                "input_bytes": path.stat().st_size, "created": datetime.now(JST).isoformat(timespec="seconds")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return manifest


def collect_batch(cfg: MmadConfig, base_dir: Path, runs_dir: Path, output_files: list[Path],
                  pricing: PricingTable, job_arn: str | None = None) -> dict[str, int]:
    """バッチ推論の出力（S3 から落とした *.jsonl.out）を、オンデマンドと同じ形の記録にする（CI-MB3）。

    - 出力に無い問題は api_error="missing in batch output" の記録にする（数から落とさない）
    - 一覧に無い recordId や、同じ recordId が2回出てきたら ValueError
    - 費用はバッチの単価で計算する
    """
    split_path = base_dir / cfg.split
    split = load_split(split_path)
    instruction = load_instruction(base_dir)
    ident = identity(cfg, instruction, _split_sha(split_path))
    _, out_dir = run_dir(cfg, runs_dir, None)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if diff := {k: (manifest.get(k), v) for k, v in ident.items() if manifest.get(k) != v}:
        raise ValueError(f"settings differ from the prepared batch input: {diff}")
    price = pricing.get(cfg.model_id, cfg.pricing_tier)
    if not price.runnable:
        raise ValueError(f"price for {cfg.model_id}/{cfg.pricing_tier} is unknown")

    outputs: dict[str, ModelResponse] = {}
    for p in output_files:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("recordId")
            if rid in outputs:
                raise ValueError(f"duplicate recordId in batch output: {rid}")
            if "modelOutput" in rec and rec["modelOutput"] is not None:
                outputs[rid] = response_from_anthropic(rec["modelOutput"])
            else:
                err = rec.get("error") or {}
                outputs[rid] = ModelResponse(raw_text=None, api_error=f"{err.get('errorCode')}: {err.get('errorMessage')}")
    if unknown := set(outputs) - set(split["record_id"]):
        raise ValueError(f"unknown recordIds in batch output: {sorted(unknown)[:5]}")

    run_id = (job_arn or "batch").split("/")[-1]
    counts = {"ok": 0, "api_error": 0, "missing": 0}
    calls = out_dir / "calls.jsonl"
    with calls.open("w", encoding="utf-8", newline="\n") as f:
        for row in split.itertuples(index=False):
            resp = outputs.get(row.record_id)
            if resp is None:
                resp = ModelResponse(raw_text=None, api_error="missing in batch output")
                counts["missing"] += 1
            elif resp.api_error:
                counts["api_error"] += 1
            else:
                counts["ok"] += 1
            cost = call_cost(price, resp.input_tokens, resp.output_tokens, resp.cache_read_tokens, resp.cache_write_tokens)
            f.write(json.dumps(make_record(cfg, ident, run_id, cfg.name, row, resp, cost, None), ensure_ascii=False) + "\n")
    return counts
