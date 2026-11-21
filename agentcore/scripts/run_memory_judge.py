"""12/17：記憶なし・20件・100件の条件で、評価の50枚を判定する（ツールなし、手元から実行。Runtime は使わない）。

- 判定の本体は 12/16 と同じ judge.run_judge（ツールなし、v1_judge、見本1枚）。12/16 のツールなしと同じ設定
- 記憶ありの条件は、判定ごとに RetrieveMemoryRecords（その版の名前空間、品目で絞る、上位3件）で事例を取り出し、
  検査対象の画像と指示の間に文で足す（CI-AC-M8）。検索の文は eval_queries.csv の query_text（正解は入れない。CI-AC-M5）
- 記録は runs/agentcore/<run_name>.jsonl（11月と同じキー ＋ condition・memory_*）。repeat ごとに1行。途中から再開できる
- 呼ぶ前に「累計 + これまでで最も高かった1回（最初は --first-guard）」が --budget を超えるなら止める。
  費用はモデル（cost_usd）＋ 取り出し 0.0005 USD／回（memory_cost_usd）

  uv run python scripts/run_memory_judge.py --run-name mem-v100 --condition v100 --repeats 2 \
      --visa-root C:/.../data/visa --budget 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "agents" / "inspector")]

from agentlab.memcases import NAMESPACES, TOP_K, memory_block  # noqa: E402
from agentlab.memstore import retrieve_request  # noqa: E402
from agentlab.record import ImageMeta, build_record, to_line  # noqa: E402
from agentlab.runcfg import new_run_config  # noqa: E402
from judge import run_judge  # noqa: E402
from vlmlab.cost import PricingTable  # noqa: E402

JST = timezone(timedelta(hours=9))
REGION = "ap-northeast-1"
MEMORY_NAME = "vlmlabCaseMemory"
RETRIEVAL_USD = 0.0005  # 長期記憶の取り出し 1回（pricing_used.md、2026-11-21）。件数で数える場合は ×件数（sources.md の U3）


def no_sandbox():
    raise RuntimeError("tools are disabled in 12/17")


def memory_id() -> str:
    c = boto3.client("bedrock-agentcore-control", region_name=REGION)
    for page in c.get_paginator("list_memories").paginate():
        for m in page.get("memories", []):
            if m["id"].startswith(f"{MEMORY_NAME}-") and m["status"] == "ACTIVE":
                return m["id"]
    sys.exit(f"{MEMORY_NAME} が ACTIVE でない")


def pick(split: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if not limit:
        return split
    groups = [g.reset_index(drop=True) for _, g in sorted(split.groupby(["category", "label"]))]
    return pd.DataFrame([groups[i % len(groups)].iloc[i // len(groups)] for i in range(limit)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--condition", required=True, choices=["none", *NAMESPACES])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="jp.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--long-edge", type=int, default=768)
    ap.add_argument("--split", default=str(LAB / "data" / "splits" / "visa_platform_claude50.csv"))
    ap.add_argument("--queries", default=str(ROOT / "runs" / "agentcore" / "memory_features" / "eval_queries.csv"))
    ap.add_argument("--visa-root", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--first-guard", type=float, default=0.005)
    a = ap.parse_args()

    price = PricingTable.load(LAB / "pricing.yaml").get(a.model, "standard")
    if not price.runnable:
        sys.exit(f"price is null for {a.model} (CI-C5)")
    cfg = new_run_config(
        run_name=a.run_name, model_id=a.model, prompt_version="v1_judge",
        prompt=(LAB / "prompts" / "v1_judge.txt").read_text(encoding="utf-8"),
        input_per_mtok=price.input_per_mtok, output_per_mtok=price.output_per_mtok,
        cache_read_per_mtok=price.cache_read_per_mtok, cache_write_per_mtok=price.cache_write_per_mtok,
        price_source=price.source_url, price_checked_on=price.checked_on, tools_enabled=False,
        platform_condition=f"memory-{a.condition}", image_long_edge=a.long_edge,
        parse={"allow_code_fence": True, "allow_trailing_code_fence": True})

    split = pick(pd.read_csv(a.split), a.limit)
    queries = pd.read_csv(a.queries).set_index("image_id")
    missing = set(split["image_id"]) - set(queries.index)
    if missing:
        sys.exit(f"no query text for {sorted(missing)[:3]}")
    refs = pd.read_csv(LAB / "data" / "splits" / "visa_refs.csv")
    visa = Path(a.visa_root)
    dp = boto3.client("bedrock-agentcore", region_name=REGION) if a.condition != "none" else None
    mid = memory_id() if dp else None
    ns = NAMESPACES.get(a.condition)

    out = ROOT / "runs" / "agentcore" / f"{a.run_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["image_id"], r["repeat"]))
    total, worst, n_new = 0.0, 0.0, 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for repeat in range(1, a.repeats + 1):
            for _, row in split.iterrows():
                if (row.image_id, repeat) in done:
                    continue
                guard = worst if worst > 0 else a.first_guard
                if total + guard > a.budget:
                    print(f"STOP before {row.image_id} r{repeat}: total {total:.4f} + guard {guard:.4f} > {a.budget}")
                    print(f"total_cost_usd={total:.5f} new={n_new} out={out}")
                    return
                hits, block = [], None
                if dp:
                    resp = dp.retrieve_memory_records(**retrieve_request(
                        mid, ns, queries.loc[row.image_id, "query_text"], TOP_K, row.category))
                    hits = resp.get("memoryRecordSummaries", [])
                    block = memory_block(hits)
                ref = refs[(refs.category == row.category) & (refs.ref_rank == 1)].iloc[0]
                o = run_judge(cfg, (visa / row.path).read_bytes(), (visa / ref.path).read_bytes(), no_sandbox,
                              extra_text=block)
                rec = build_record(cfg, ImageMeta(cfg.run_name, row.image_id, row.category, row.label, str(row.path)),
                                   o, None, datetime.now(JST).isoformat(timespec="seconds"))
                md = [h.get("metadata", {}) for h in hits]
                rec.update({
                    "repeat": repeat, "condition": a.condition, "memory_namespace": ns,
                    "memory_hit_ids": [m.get("image_id", {}).get("stringValue") for m in md],
                    "memory_hit_judgements": [m.get("judgement", {}).get("stringValue") for m in md],
                    "memory_hit_scores": [h.get("score") for h in hits],
                    "memory_text": block, "n_retrievals": 1 if dp else 0,
                    "memory_cost_usd": RETRIEVAL_USD if dp else 0.0})
                f.write(to_line(rec) + "\n")
                f.flush()
                n_new += 1
                cost = (rec["cost_usd"] or 0.0) + rec["memory_cost_usd"]
                total += cost
                worst = max(worst, cost)
                print(json.dumps({k: rec[k] for k in ("image_id", "repeat", "label", "pred", "parse_error", "api_error",
                                                      "input_tokens", "output_tokens", "cost_usd",
                                                      "memory_hit_judgements")}, ensure_ascii=False))
    print(f"total_cost_usd={total:.5f} new={n_new} out={out}")


if __name__ == "__main__":
    main()
