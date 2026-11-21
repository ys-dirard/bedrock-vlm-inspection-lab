"""検査エージェントの Runtime（vlmlab_ci_inspector）に、画像を1枚ずつ渡して判定させる（12/16〜）。

- 1枚ごとに新しい Runtime のセッションで呼び、判定のあとに StopRuntimeSession で止める（11/30 の P-idle の結果）
- 記録は runs/agentcore/<run_name>.jsonl に1行ずつ追記する（11月と同じキー。Runtime は何も保存しない）
- 呼ぶ前に「この run の累計 + これまでで最も高かった1枚（最初は --first-guard）」が --budget を超えるなら止める
- 単価は lab の pricing.yaml（null なら止める＝CI-C5）

  uv run python scripts/invoke_runtime.py --run-name ci-haiku-tools-trial --split ../data/splits/visa_platform_claude50.csv \
      --limit 5 --model jp.anthropic.claude-haiku-4-5-20251001-v1:0 --long-edge 768 --prompt-cache \
      --visa-root C:/.../data/visa --budget 0.12
  （ツールなしは --no-tools --prompt v1_judge）
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.payload import build_payload  # noqa: E402
from agentlab.runcfg import new_run_config  # noqa: E402
from vlmlab.cost import PricingTable  # noqa: E402

REGION = "ap-northeast-1"
RUNTIME = "vlmlab_ci_inspector"


def runtime_arn(ctl) -> str:
    for page in ctl.get_paginator("list_agent_runtimes").paginate():
        for r in page.get("agentRuntimes", []):
            if r["agentRuntimeName"] == RUNTIME and r["status"] == "READY":
                return r["agentRuntimeArn"]
    sys.exit(f"{RUNTIME} is not READY")


def read_body(resp) -> dict:
    body = resp["response"]
    raw = body.read() if hasattr(body, "read") else b"".join(body)
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--visa-root", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--first-guard", type=float, default=0.03, help="最初の1枚の見込み（USD）")
    ap.add_argument("--long-edge", type=int)
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--prompt-cache", action="store_true")
    ap.add_argument("--prompt", default=None, help="既定はツールありなら v1_agent、なしなら v1_judge")
    ap.add_argument("--trailing-fence", action="store_true", default=True)
    a = ap.parse_args()

    price = PricingTable.load(LAB / "pricing.yaml").get(a.model, "standard")
    if not price.runnable:
        sys.exit(f"price is null for {a.model} (CI-C5)")
    version = a.prompt or ("v1_judge" if a.no_tools else "v1_agent")
    cfg = new_run_config(
        run_name=a.run_name, model_id=a.model, prompt_version=version,
        prompt=(LAB / "prompts" / f"{version}.txt").read_text(encoding="utf-8"),
        input_per_mtok=price.input_per_mtok, output_per_mtok=price.output_per_mtok,
        cache_read_per_mtok=price.cache_read_per_mtok, cache_write_per_mtok=price.cache_write_per_mtok,
        price_source=price.source_url, price_checked_on=price.checked_on, tools_enabled=not a.no_tools,
        platform_condition="runtime-direct", image_long_edge=a.long_edge, prompt_cache=a.prompt_cache,
        parse={"allow_code_fence": True, "allow_trailing_code_fence": a.trailing_fence})

    split = pd.read_csv(a.split)
    if a.limit:  # 品目×正解の4群から順に1枚ずつ（11/30 の試運転と同じ選び方）
        groups = [g.reset_index(drop=True) for _, g in sorted(split.groupby(["category", "label"]))]
        rows = [groups[i % len(groups)].iloc[i // len(groups)] for i in range(a.limit)]
        split = pd.DataFrame(rows)
    refs = pd.read_csv(LAB / "data" / "splits" / "visa_refs.csv")
    visa = Path(a.visa_root)

    ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    dp = boto3.client("bedrock-agentcore", region_name=REGION)
    arn = runtime_arn(ctl)
    out = ROOT / "runs" / "agentcore" / f"{a.run_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    total, worst, errors = 0.0, 0.0, 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for _, row in split.iterrows():
            guard = worst if worst > 0 else a.first_guard
            if total + guard > a.budget:
                print(f"STOP before {row.image_id}: total {total:.4f} + guard {guard:.4f} > budget {a.budget}")
                break
            ref = None
            if cfg.n_refs > 0:
                r = refs[(refs.category == row.category) & (refs.ref_rank == 1)].iloc[0]
                ref = (visa / r.path).read_bytes()
            payload = build_payload(cfg, row.image_id, row.category, row.label, (visa / row.path).read_bytes(), ref)
            sid = f"{a.run_name}-{uuid.uuid4().hex}"  # 33文字以上
            try:
                body = read_body(dp.invoke_agent_runtime(agentRuntimeArn=arn, runtimeSessionId=sid,
                                                         payload=json.dumps(payload).encode("utf-8")))
            finally:
                try:
                    dp.stop_runtime_session(agentRuntimeArn=arn, runtimeSessionId=sid)
                except Exception as e:  # noqa: BLE001  止められなかったことは表示して続ける（idle 60秒で止まる）
                    print(f"stop_runtime_session failed: {type(e).__name__}: {e}")
            if "error" in body:
                errors += 1
                print("ERROR", row.image_id, body["error"])
                continue
            f.write(body["record_line"] + "\n")
            f.flush()
            s = body["summary"]
            cost = s.get("cost_usd") or 0.0
            total += cost
            worst = max(worst, cost)
            print(json.dumps(s, ensure_ascii=False))
    print(f"total_cost_usd={total:.5f} errors={errors} out={out}")


if __name__ == "__main__":
    main()
