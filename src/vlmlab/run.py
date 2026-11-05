"""設定ファイルを読んで実験を回し、1呼び出し1行の JSONL に追記する。

- 途中で止まっても、同じ設定でもう一度実行すれば続きから再開する（成功した (image_id, repeat) は飛ばす）
- 回（repeat）を外側に回すので、予算で止まっても完了した回は丸ごと使える
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from .bedrock import ModelRequest, ModelResponse, VisionModel
from .cost import BudgetExceeded, BudgetGuard, PricingTable, call_cost
from .data import EncodedImage, load_image
from .parse import ParseSpec, parse_response

JST = timezone(timedelta(hours=9))


@dataclass
class RunConfig:
    name: str
    split: str
    model_id: str
    region: str
    prompt: str
    budget_usd: float
    refs: str | None = None
    n_refs: int = 0
    system_prompt: str | None = None
    prompt_vars: dict[str, dict[str, str]] = field(default_factory=dict)  # 品目ごとの差し替え {{key}}
    service_tier: str = "standard"
    image_long_edge: int | None = None
    inference: dict[str, Any] = field(default_factory=lambda: {"temperature": 0, "maxTokens": 300})
    additional_fields: dict[str, Any] | None = None
    repeats: int = 1
    parse: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.n_refs > 0 and not cfg.refs:
            raise ValueError("n_refs > 0 requires refs")
        if cfg.repeats < 1:
            raise ValueError("repeats must be >= 1")
        return cfg

    @property
    def prompt_version(self) -> str:
        return Path(self.prompt).stem


def fill_template(text: str, values: dict[str, str]) -> str:
    """{{key}} を差し替える。JSON の波かっこと混ざらないよう、二重の波かっこだけを使う。"""
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    if "{{" in text:
        raise ValueError(f"unfilled placeholder in prompt: ...{text[text.index('{{'):][:40]}")
    return text


def select_limit(split: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """試運転用に、品目と正解の組から順に1枚ずつ取って limit 枚にする。"""
    if limit is None:
        return split
    order = split.groupby(["category", "label"], sort=False).cumcount()
    return split.assign(_o=order).sort_values("_o", kind="stable").head(limit).drop(columns="_o")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class RunPlan:
    cfg: RunConfig
    run_name: str
    out_dir: Path
    items: pd.DataFrame
    refs_by_cat: dict[str, list[str]]


def plan_run(cfg: RunConfig, base_dir: Path, runs_dir: Path, limit: int | None = None) -> RunPlan:
    split = pd.read_csv(base_dir / cfg.split)
    items = select_limit(split, limit)
    refs_by_cat: dict[str, list[str]] = {}
    if cfg.n_refs > 0:
        refs = pd.read_csv(base_dir / cfg.refs).sort_values(["category", "ref_rank"])
        refs = refs[refs.ref_rank <= cfg.n_refs]
        refs_by_cat = refs.groupby("category")["path"].apply(list).to_dict()
        lacking = [c for c in items.category.unique() if len(refs_by_cat.get(c, [])) < cfg.n_refs]
        if lacking:
            raise ValueError(f"not enough refs for categories {lacking}")
    # 品目ごとの差し替えが足りないことに、途中で（課金のあとで）気づかないよう先に確かめる
    prompt_text, system_text = read_prompts(cfg, base_dir)
    for cat in items.category.unique():
        for text in filter(None, [prompt_text, system_text]):
            fill_template(text, cfg.prompt_vars.get(cat, {}))
    run_name = cfg.name if limit is None else f"{cfg.name}__trial{limit}"
    return RunPlan(cfg, run_name, runs_dir / run_name, items, refs_by_cat)


def execute(
    plan: RunPlan,
    model: VisionModel,
    pricing: PricingTable,
    visa_root: Path,
    base_dir: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    cfg = plan.cfg
    price = pricing.get(cfg.model_id, cfg.service_tier)
    if not price.runnable:  # CI-C5
        raise RuntimeError(f"price for {cfg.model_id} ({cfg.service_tier}) is null in pricing.yaml; confirm it first")

    plan.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = plan.out_dir / "calls.jsonl"
    previous = load_jsonl(jsonl)
    prompt_text, system_text = read_prompts(cfg, base_dir)
    identity = call_identity(cfg, prompt_text, system_text)
    check_resume_identity(previous, identity)
    done = {(r["image_id"], r["repeat"]) for r in previous if r.get("api_error") is None}
    guard = BudgetGuard.resume(cfg.budget_usd, [r.get("cost_usd") for r in previous])  # CI-C6

    # 使った設定とプロンプトを結果と一緒に残す
    _snapshot(cfg, plan.out_dir, base_dir)
    spec = ParseSpec.from_dict(cfg.parse)
    run_id = datetime.now(JST).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]

    image_cache: dict[str, EncodedImage] = {}

    def img(rel: str) -> EncodedImage:
        if rel not in image_cache:
            image_cache[rel] = load_image(visa_root / rel, cfg.image_long_edge)
        return image_cache[rel]

    n_called = n_skipped = 0
    stopped = None
    with jsonl.open("a", encoding="utf-8") as f:
        for repeat in range(1, cfg.repeats + 1):
            for row in plan.items.itertuples(index=False):
                if (row.image_id, repeat) in done:
                    n_skipped += 1
                    continue
                try:
                    guard.check_before_call()  # CI-C4
                except BudgetExceeded as e:
                    stopped = str(e)
                    break
                vars_ = cfg.prompt_vars.get(row.category, {})
                req = ModelRequest(
                    model_id=cfg.model_id,
                    images=[img(p) for p in plan.refs_by_cat.get(row.category, [])] + [img(row.path)],
                    text=fill_template(prompt_text, vars_),
                    system=fill_template(system_text, vars_) if system_text else None,
                    temperature=cfg.inference.get("temperature", 0),
                    max_tokens=cfg.inference.get("maxTokens", 300),
                    service_tier=cfg.service_tier,
                    additional_fields=cfg.additional_fields,
                )
                called_at = datetime.now(JST).isoformat(timespec="seconds")
                resp = model.invoke(req)
                rec = make_record(cfg, run_id, plan.run_name, repeat, row, resp, spec, price, identity, called_at)
                # 先に記録を書く（費用が不明で止まるときも、払った呼び出しの記録は残す）
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                guard.add(rec["cost_usd"])
                n_called += 1
                if resp.api_error:
                    log(f"[api_error] {row.image_id} r{repeat}: {resp.api_error}")
            if stopped:
                break
    summary = {"run_name": plan.run_name, "run_id": run_id, "called": n_called, "skipped": n_skipped,
               "spent_usd": guard.spent_usd, "stopped": stopped}
    log(json.dumps(summary, ensure_ascii=False))
    return summary


def read_prompts(cfg: RunConfig, base_dir: Path) -> tuple[str, str | None]:
    prompt_text = (base_dir / cfg.prompt).read_text(encoding="utf-8")
    system_text = (base_dir / cfg.system_prompt).read_text(encoding="utf-8") if cfg.system_prompt else None
    return prompt_text, system_text


def call_identity(cfg: RunConfig, prompt_text: str, system_text: str | None) -> dict[str, Any]:
    """呼び出しの中身を決める設定。再開のとき、これが前の記録と同じでなければ止める。"""
    prompt_sha = hashlib.sha256(
        json.dumps([prompt_text, system_text, cfg.prompt_vars], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "model_id": cfg.model_id,
        "service_tier_requested": cfg.service_tier,
        "prompt_version": cfg.prompt_version,
        "prompt_sha": prompt_sha,
        "n_refs": cfg.n_refs,
        "image_long_edge": cfg.image_long_edge,
        "temperature": cfg.inference.get("temperature", 0),
        "max_tokens": cfg.inference.get("maxTokens", 300),
        "additional_fields": cfg.additional_fields,
    }


def check_resume_identity(previous: list[dict[str, Any]], identity: dict[str, Any]) -> None:
    for r in previous:
        diff = {k: (r.get(k), v) for k, v in identity.items() if r.get(k) != v}
        if diff:
            raise RuntimeError(
                "config differs from existing records in this run directory; "
                f"use a new name or remove the directory. differences (old, new): {diff}"
            )


def make_record(cfg: RunConfig, run_id: str, run_name: str, repeat: int, row: Any,
                resp: ModelResponse, spec: ParseSpec, price, identity: dict[str, Any],
                called_at: str) -> dict[str, Any]:
    pr = parse_response(resp.raw_text, spec) if resp.api_error is None else None
    return {
        "run_name": run_name,
        "run_id": run_id,
        "repeat": repeat,
        "image_id": row.image_id,
        "category": row.category,
        "label": row.label,
        "path": row.path,
        **identity,
        "service_tier_resolved": resp.service_tier_resolved,
        "raw_text": resp.raw_text,
        "parsed": pr.parsed if pr else None,
        "parse_error": pr.parse_error if pr else None,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cache_read_tokens": resp.cache_read_tokens,
        "cache_write_tokens": resp.cache_write_tokens,
        "latency_s": resp.latency_s,
        "stop_reason": resp.stop_reason,
        "cost_usd": call_cost(price, resp.input_tokens, resp.output_tokens,
                              resp.cache_read_tokens, resp.cache_write_tokens),
        "attempts": resp.attempts,
        "api_error": resp.api_error,
        "server_latency_ms": resp.extra.get("server_latency_ms"),
        "timestamp": called_at,  # 呼び出した時刻（JST）
    }


def _snapshot(cfg: RunConfig, out_dir: Path, base_dir: Path) -> None:
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.__dict__, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for rel in filter(None, [cfg.prompt, cfg.system_prompt]):
        src = base_dir / rel
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        shutil.copyfile(src, out_dir / f"{Path(rel).stem}.{digest}{Path(rel).suffix}")
