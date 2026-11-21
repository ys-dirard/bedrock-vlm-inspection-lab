"""12/18：12/16 の Runtime のトレースを、画像を除いてから on-demand で評価する。

1枚（= Runtime の1セッション）ごとに:
1. CloudWatch（aws/spans と Runtime のロググループ）から、そのセッションのスパンを集める（SDK の CloudWatchAgentSpanCollector）
2. 画像の Base64 を除き（agentlab.spans.strip_images）、**画像が残っていないことを確かめる**（assert_text_only。
   残っていれば評価を呼ばずに止める）。除いたスパンは runs/agentcore/eval/<run>/ に保存する（あとで見直せるように）
3. Evaluate API に、除いたスパンと TRACE の ID を渡す。結果の tokenUsage から費用を計算する（agentlab.evalcost）
4. 呼ぶ前に「累計 + 1回の上限の見込み（最初は --first-guard、既定 0.48 USD = 入力 20万トークンの組み込みの評価）」が
   --budget を超えるなら止める

  uv run python scripts/evaluate_traces.py --records runs/agentcore/ci-haiku-tools.jsonl --limit 1 --budget 0.5
  （--evaluator は既定で vlmlab_reason_consistency。組み込みなら --evaluator Builtin.Correctness --kind builtin）
出力: runs/agentcore/eval/<records の名前>__<評価器>.jsonl（1行1結果）
区切って流し直せる（CI-AC-S5・S6）：--run-id で記録を絞り、この評価器でエラーなく評価できたセッションは飛ばす。
--budget はこれまでの結果の費用を含めた上限。--max-new で1回に評価する件数を区切る（--limit は対象の先頭 N 件）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentlab.evalcost import eval_cost  # noqa: E402
from agentlab.evalresume import evaluated, select_targets  # noqa: E402
from agentlab.spans import assert_text_only, missing_events, strip_images  # noqa: E402

REGION = "ap-northeast-1"
RUNTIME = "vlmlab_ci_inspector"
DEFAULT_EVALUATOR = "vlmlab_reason_consistency"
JUDGE_PRICES = {  # カスタムの評価器の判定のモデル（pricing.yaml と同じ値。2026-11-21 確認）
    "jp.anthropic.claude-haiku-4-5-20251001-v1:0": (1.10, 5.50),
}


def runtime_log_group(ctl) -> str:
    for page in ctl.get_paginator("list_agent_runtimes").paginate():
        for r in page.get("agentRuntimes", []):
            if r["agentRuntimeName"] == RUNTIME:
                return f"/aws/bedrock-agentcore/runtimes/{r['agentRuntimeId']}-DEFAULT"
    sys.exit(f"{RUNTIME} が見つからない（ロググループを決められない）")


def resolve_evaluator(ctl, name: str) -> tuple[str, str, str | None]:
    """(evaluatorId, kind, 判定のモデル ID)。Builtin.* はそのまま。"""
    if name.startswith("Builtin."):
        return name, "builtin", None
    for page in ctl.get_paginator("list_evaluators").paginate():
        for e in page.get("evaluators", []):
            if e.get("evaluatorName") == name:
                d = ctl.get_evaluator(evaluatorId=e["evaluatorId"])
                model = (d.get("evaluatorConfig", {}).get("llmAsAJudge", {}).get("modelConfig", {})
                         .get("bedrockEvaluatorModelConfig", {}).get("modelId"))
                return e["evaluatorId"], "custom", model
    sys.exit(f"評価器 {name} が見つからない")


def main() -> None:
    from bedrock_agentcore.evaluation import spans as span_helpers
    from bedrock_agentcore.evaluation.agent_span_collector.agent_span_collector import CloudWatchAgentSpanCollector

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="invoke_runtime.py の記録（runtime_session_id を読む）")
    ap.add_argument("--evaluator", default=DEFAULT_EVALUATOR)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--first-guard", type=float, default=0.48)
    ap.add_argument("--run-id", help="この run_id の記録だけを評価する")
    ap.add_argument("--max-new", type=int, help="この実行で新しく評価する件数の上限")
    a = ap.parse_args()

    recs = [json.loads(line) for line in Path(a.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = select_targets(recs, a.run_id)[: a.limit]
    ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    dp = boto3.client("bedrock-agentcore", region_name=REGION)
    evaluator_id, kind, judge_model = resolve_evaluator(ctl, a.evaluator)
    judge_prices = JUDGE_PRICES.get(judge_model) if kind == "custom" else (None, None)
    if kind == "custom" and judge_prices is None:
        sys.exit(f"判定のモデル {judge_model} の単価が無い（CI-C5）")
    collector = CloudWatchAgentSpanCollector(log_group_name=runtime_log_group(ctl), region=REGION,
                                             max_wait_seconds=120, poll_interval_seconds=20)

    out_dir = ROOT / "runs" / "agentcore" / "eval" / Path(a.records).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir.parent / f"{Path(a.records).stem}__{a.evaluator.replace('.', '_')}.jsonl"
    prev = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()] if out.exists() else []
    done, total, worst = evaluated(prev, a.evaluator)
    recs = [r for r in recs if r["runtime_session_id"] not in done]
    if a.max_new is not None:
        recs = recs[: a.max_new]
    print(f"done={len(done)} todo_now={len(recs)} prior_cost={total:.4f}")
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for r in recs:
            guard = worst if worst > 0 else a.first_guard
            if total + guard > a.budget:
                print(f"STOP before {r['image_id']}: total {total:.4f} + guard {guard:.4f} > budget {a.budget}")
                break
            sid = r["runtime_session_id"]
            t = datetime.fromisoformat(r["timestamp"])
            spans = collector.collect(session_id=sid, start_time=t - timedelta(minutes=10),
                                      end_time=t + timedelta(minutes=30))
            if not spans:
                print("NO SPANS", r["image_id"], sid)
                continue
            raw_bytes = len(json.dumps(spans, ensure_ascii=False).encode("utf-8"))
            stripped, n_removed = strip_images(spans)
            sent_bytes = assert_text_only(stripped)  # 画像が残っていれば ValueError で止まる（評価は呼ばない）
            (out_dir / f"{sid}.spans.json").write_text(json.dumps(stripped, ensure_ascii=False, indent=1),
                                                       encoding="utf-8", newline="\n")
            lacking = missing_events(stripped)
            if lacking:  # CI-AC-S7：イベントの欠けたスパンがあると評価は失敗するので、呼ばずに記録だけ残す
                row = {"image_id": r["image_id"], "label": r["label"], "pred": r.get("pred"), "session_id": sid,
                       "evaluator": a.evaluator, "kind": kind, "n_spans": len(spans), "images_removed": n_removed,
                       "raw_bytes": raw_bytes, "sent_bytes": sent_bytes, "value": None, "result_label": None,
                       "explanation": None, "input_tokens": None, "output_tokens": None, "cost_usd": 0.0,
                       "error": f"incomplete: {len(lacking)} spans lack log events (not evaluated): {lacking[:5]}",
                       "skipped": True}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print("SKIP incomplete", r["image_id"], lacking[:5])
                continue
            trace_ids = span_helpers.trace_ids(stripped)[:10]
            resp = dp.evaluate(evaluatorId=evaluator_id, evaluationInput={"sessionSpans": stripped},
                               evaluationTarget={"traceIds": trace_ids})
            for res in resp.get("evaluationResults", []):
                u = res.get("tokenUsage") or {}
                cost = eval_cost(kind, int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0)), *judge_prices)
                total += cost
                worst = max(worst, cost)
                row = {"image_id": r["image_id"], "label": r["label"], "pred": r.get("pred"), "session_id": sid,
                       "evaluator": a.evaluator, "kind": kind, "n_spans": len(spans), "images_removed": n_removed,
                       "raw_bytes": raw_bytes, "sent_bytes": sent_bytes, "value": res.get("value"),
                       "result_label": res.get("label"), "explanation": res.get("explanation"),
                       "input_tokens": u.get("inputTokens"), "output_tokens": u.get("outputTokens"),
                       "cost_usd": cost, "error": res.get("errorMessage")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({k: row[k] for k in ("image_id", "result_label", "input_tokens", "output_tokens",
                                                      "images_removed", "raw_bytes", "sent_bytes", "cost_usd",
                                                      "error")}, ensure_ascii=False))
    print(f"total_cost_usd={total:.5f} out={out}")


if __name__ == "__main__":
    main()
