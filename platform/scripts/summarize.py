"""表1（モデルごとの判定・時間・費用）を作る。11月の vlmlab report の指標に、基盤ならではの列を足す。

- 主の指標は 11月の規則（CI-P1。救済しない）。参考として「本文の先頭の JSON オブジェクトだけを読む」値も出す
  （書き手の判断 2026-11-20：Nova は JSON のあとに説明を続けるため、主の規則では全件が解析失敗になる。12/2 の記事の「参考の読み方」と同じ扱い）
- 1枚の処理時間 = S3 に置いた時刻（inbox のオブジェクトの LastModified）から、DynamoDB で DONE になった時刻まで（framing.md の MP3）

  AWS_PROFILE=<profile> uv run python scripts/summarize.py --runs p-nova p-qwen p-claude --out runs/summary_models.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from upload_run import REGION, outputs  # noqa: E402
from vlmlab.metrics import binary_metrics  # noqa: E402
from vlmlab.parse import ParseSpec, parse_response  # noqa: E402


def reference_read(text: str | None, spec: ParseSpec):
    """参考の読み方：本文（前後の空白を除く）の先頭から最初の JSON オブジェクトだけを読み、後ろの文章は捨てる。"""
    if not text:
        return None
    t = text.strip()
    if not t.startswith("{"):
        return parse_response(text, spec)  # 先頭が JSON でなければ、主の規則と同じ
    try:
        obj, end = json.JSONDecoder().raw_decode(t)
    except json.JSONDecodeError:
        return parse_response(text, spec)
    return parse_response(json.dumps(obj, ensure_ascii=False), spec)


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--collected", default=str(ROOT / "runs" / "collected"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = outputs()
    s3 = boto3.client("s3", region_name=REGION)
    table = boto3.resource("dynamodb", region_name=REGION).Table(out["StatusTable"])
    rows = []
    for run in a.runs:
        recs = [json.loads(l) for l in (Path(a.collected) / run / "calls.jsonl").open(encoding="utf-8")]
        cfg = json.loads(s3.get_object(Bucket=out["BucketName"], Key=f"run-configs/{run}.json")["Body"].read())
        spec = ParseSpec.from_dict(cfg["parse"])
        main_pred = [parse_response(r["raw_text"], spec).pred if r["api_error"] is None else None for r in recs]
        ref_pred = [(reference_read(r["raw_text"], spec) or parse_response(None, spec)).pred for r in recs]
        import pandas as pd
        df = pd.DataFrame({"label": [r["label"] for r in recs], "pred": main_pred, "repeat": 1})
        dfr = pd.DataFrame({"label": [r["label"] for r in recs], "pred": ref_pred, "repeat": 1})
        m, mr = binary_metrics(df), binary_metrics(dfr)
        e2e = []
        for r in recs:
            head = s3.head_object(Bucket=out["BucketName"], Key=r["path"])
            it = table.get_item(Key={"image_key": r["path"]}).get("Item") or {}
            if it.get("finished_at"):
                e2e.append((datetime.fromisoformat(it["finished_at"].replace("Z", "+00:00")) - head["LastModified"]).total_seconds())
        n = len(recs)
        mean = lambda k: sum(r[k] or 0 for r in recs) / n  # noqa: E731
        rows.append({
            "run_name": run, "condition": cfg["platform_condition"], "model_id": cfg["model_id"],
            "image_long_edge": cfg["image_long_edge"], "tools_enabled": cfg["tools_enabled"],
            "prompt_cache": cfg.get("prompt_cache", False), "n_images": n,
            "n_valid": m["n_valid"], "parse_fail_rate": m["parse_fail_rate"], "recall": m["recall"],
            "false_positive_rate": m["false_positive_rate"], "precision": m["precision"], "f1": m["f1"],
            "ref_n_valid": mr["n_valid"], "ref_recall": mr["recall"], "ref_false_positive_rate": mr["false_positive_rate"],
            "mean_input_tokens": mean("input_tokens"), "mean_cache_read_tokens": mean("cache_read_tokens"),
            "mean_cache_write_tokens": mean("cache_write_tokens"), "mean_output_tokens": mean("output_tokens"),
            "mean_cycles": mean("n_cycles"), "mean_tool_calls": mean("n_tool_calls"),
            "code_interpreter_rate": sum(bool(r["used_code_interpreter"]) for r in recs) / n,
            "agent_latency_p50_s": pct([r["latency_s"] for r in recs], 50),
            "agent_latency_p95_s": pct([r["latency_s"] for r in recs], 95),
            "e2e_latency_p50_s": pct(e2e, 50), "e2e_latency_p95_s": pct(e2e, 95), "e2e_n": len(e2e),
            "model_cost_usd": sum(r["cost_usd"] for r in recs), "model_cost_per_1000_usd": mean("cost_usd") * 1000,
            "n_api_error": sum(r["api_error"] is not None for r in recs),
        })
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
