"""表2：条件ごと・サービスごとの費用（量 × 単価）。docs/contracts.md の CI-PL-C1〜C5。

入力
- runs/summary_models.csv（scripts/summarize.py）: モデルの費用（トークン数 × 単価、記録ごとの合計）
- runs/runtime_metrics.csv（scripts/runtime_usage.py）: Runtime・Code Interpreter の CPU とメモリ（条件ごとの時間帯）
- AWS から読むもの: Athena のクエリごとのスキャン量、Lambda の呼び出し回数と時間、ロググループの保存量

単価は results/facts.md の F9（2026-11-20 確認、東京）。量が見積もりの行は source=estimated とし、根拠を basis に書く。

  AWS_PROFILE=<profile> uv run python scripts/cost_breakdown.py --out runs/cost_breakdown.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from platformlab.costbreak import CostLine, per_1000, runtime_lines, total  # noqa: E402
from upload_run import REGION, outputs  # noqa: E402

JST = timezone(timedelta(hours=9))
P = {  # facts.md F9（東京、2026-11-20）
    "sfn_transition": 0.000025, "lambda_request": 0.20 / 1e6, "lambda_gb_s_arm": 0.0000133334,
    "ddb_wru": 0.715 / 1e6, "s3_put": 0.0047 / 1000, "s3_get": 0.00037 / 1000, "athena_tb": 5.00,
    "gateway_call": 0.000005, "memory_stm_ingest_gb": 1.00, "logs_ingest_gb": 0.76, "spans_ingest_gb": 0.35,
    "eventbridge_s3_event": None,  # 未確認（facts.md F9）。0 にしない（CI-PL-C2）
}
TRANSITIONS_PER_EXECUTION = 11  # 1回の実行の StateEntered（2026-11-20、実行の履歴で確認。成功した実行はすべて同じ経路）
GB = 1024 ** 3
MB10 = 10 * 1024 * 1024


def read_csv(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open(encoding="utf-8")))


def pipeline_lines(cond: str, n: int, model_cost: float, rt: dict[str, dict]) -> list[CostLine]:
    """検査の経路（S3 → EventBridge → Step Functions → Runtime → S3・DynamoDB）の1条件ぶん。"""
    lines = [CostLine("Amazon Bedrock（モデル）", "tokens", model_cost, "USD", 1.0, "measured",
                      "記録のトークン数（入力・出力・キャッシュの読み書き）× 単価の合計")]
    r = rt.get(cond)
    if r:
        lines += runtime_lines("検査エージェント", float(r["vcpu_hours"]), float(r["gb_hours"]), "V1", "measured",
                               f"CloudWatch AWS/Bedrock-AgentCore（{r['start_jst']}〜{r['end_jst']}、1分ごとの合計）")
    ci = rt.get(cond + "#ci")
    if ci:
        src = "estimated" if cond.startswith("P-idle") else "measured"
        basis = "CloudWatch（Service=AgentCore.CodeInterpreter）" + ("。P-idle の2条件で同じ時間帯のため、枚数で半分ずつに按分" if src == "estimated" else "")
        lines += [CostLine("AgentCore Code Interpreter", "vCPU", float(ci["vcpu_hours"]), "vCPU-hour", 0.0895, src, basis),
                  CostLine("AgentCore Code Interpreter", "Memory", float(ci["gb_hours"]), "GB-hour", 0.00945, src, basis)]
    lines += [
        CostLine("AWS Step Functions（Standard）", "StateTransition", n * TRANSITIONS_PER_EXECUTION, "transition",
                 P["sfn_transition"], "measured", f"実行 {n} 回 × 状態遷移 {TRANSITIONS_PER_EXECUTION}（無料枠 月4,000 は考えない）"),
        CostLine("Amazon EventBridge", "S3 Object Created（opt-in data event）", n, "event", P["eventbridge_s3_event"],
                 "measured", "画像1枚につき1件。単価は未確認"),
        CostLine("Amazon S3", "PUT", n * 2, "request", P["s3_put"], "estimated", "1枚につき inbox への PUT と results への PUT"),
        CostLine("Amazon S3", "GET・HEAD", n * 4, "request", P["s3_get"], "estimated",
                 "1枚につき HeadObject（Step Functions）と GetObject 3回（画像・設定・見本）"),
        CostLine("Amazon DynamoDB", "WriteRequestUnits", n * 2, "WRU", P["ddb_wru"], "estimated",
                 "1枚につき PutItem と UpdateItem（どちらも 1 KB 未満）。検証のスクリプトの読み取りは含めない"),
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "runs" / "summary_models.csv"))
    ap.add_argument("--runtime", default=str(ROOT / "runs" / "runtime_metrics.csv"))
    ap.add_argument("--qa", default=str(ROOT / "runs" / "qa"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = outputs()

    # 条件ごとに、どの時間帯（window）のどの Runtime を数えるか。P-idle の2条件は同じ時間帯に別々の Runtime で動かした
    use = {"P-nova": ("P-nova", "vlmlab_inspector_main"), "P-qwen": ("P-qwen", "vlmlab_inspector_main"),
           "P-claude": ("P-claude", "vlmlab_inspector_main"), "P-idle-900": ("P-idle", "vlmlab_inspector_main"),
           "P-idle-60": ("P-idle", "vlmlab_inspector_idle60")}
    raw = {(r["window"], r["resource"]): r for r in read_csv(Path(a.runtime))}
    rt = {}
    for cond, (win, res) in use.items():
        if (win, res) in raw:
            rt[cond] = raw[(win, res)]
        ci = raw.get((win, "code_interpreter(aws.codeinterpreter.v1)"))
        if ci:
            share = 0.5 if win == "P-idle" else 1.0  # P-idle は2条件で同じ時間帯（10枚ずつ）なので半分ずつ
            rt[cond + "#ci"] = {**ci, "vcpu_hours": float(ci["vcpu_hours"]) * share,
                                "gb_hours": float(ci["gb_hours"]) * share}
    if ("Q", "vlmlab_analyst") in raw:
        rt["Q#analyst"] = raw[("Q", "vlmlab_analyst")]

    rows, per1000 = [], []
    for s in read_csv(Path(a.summary)):
        cond, n = s["condition"], int(s["n_images"])
        lines = pipeline_lines(cond, n, float(s["model_cost_usd"]), rt)
        for ln in lines:
            rows.append({"condition": cond, **ln.to_row()})
        known = [ln for ln in lines if ln.cost_usd is not None]
        per1000.append({"condition": cond, "n_images": n, "total_known_usd": total(known),
                        "per_1000_known_usd": per_1000(known, n),
                        "unknown_lines": "; ".join(f"{ln.service}/{ln.usage_type}" for ln in lines if ln.cost_usd is None)})

    # 分析（条件 Q）：分析エージェントの Runtime、Lambda、Athena、Gateway、Memory、モデル
    qa_cost = sum(float(r["cost_usd"] or 0) for f in ("qa_eval_v1.csv", "qa_eval.csv")
                  for r in read_csv(Path(a.qa) / f))
    athena = boto3.client("athena", region_name=REGION)
    ids = athena.list_query_executions(WorkGroup=out["AthenaWorkGroup"], MaxResults=50)["QueryExecutionIds"]
    scanned = billable = 0
    for i in range(0, len(ids), 50):
        for q in athena.batch_get_query_execution(QueryExecutionIds=ids[i:i + 50])["QueryExecutions"]:
            b = q.get("Statistics", {}).get("DataScannedInBytes", 0) or 0
            scanned += b
            billable += max(b, MB10)
    lam = boto3.client("lambda", region_name=REGION)
    fn = [f["FunctionName"] for f in lam.list_functions()["Functions"] if "AnalyticsTools" in f["FunctionName"]][0]
    cw = boto3.client("cloudwatch", region_name=REGION)
    start, end = datetime(2026, 11, 20, 11, 0, tzinfo=JST), datetime.now(JST)
    stat = lambda m: sum(p["Sum"] for p in cw.get_metric_statistics(  # noqa: E731
        Namespace="AWS/Lambda", MetricName=m, Dimensions=[{"Name": "FunctionName", "Value": fn}],
        StartTime=start, EndTime=end, Period=86400, Statistics=["Sum"])["Datapoints"])
    inv, dur_ms = stat("Invocations"), stat("Duration")
    n_q = len(read_csv(Path(a.qa) / "qa_eval_v1.csv")) + len(read_csv(Path(a.qa) / "qa_eval.csv"))
    q_lines = [CostLine("Amazon Bedrock（モデル）", "tokens", qa_cost, "USD", 1.0, "measured", f"分析エージェントの {n_q} 問（v1 と v2）")]
    an = rt.get("Q#analyst")
    if an:
        q_lines += runtime_lines("分析エージェント", float(an["vcpu_hours"]), float(an["gb_hours"]), "V1", "measured",
                                 f"CloudWatch（{an['start_jst']}〜{an['end_jst']}）")
    q_lines += [
        CostLine("AWS Lambda（arm64）", "Requests", inv, "request", P["lambda_request"], "measured", "CloudWatch AWS/Lambda Invocations（試運転を含む）"),
        CostLine("AWS Lambda（arm64）", "GB-seconds", dur_ms / 1000 * 0.25, "GB-second", P["lambda_gb_s_arm"], "measured", "Duration × 256 MB"),
        CostLine("Amazon Athena", "DataScanned（課金は1クエリ最小 10 MB）", billable / 1024 ** 4, "TB", P["athena_tb"], "measured",
                 f"クエリ {len(ids)} 回、実際のスキャン量 {scanned / 1e6:.2f} MB（正解を出す SQL と試運転を含む）"),
        CostLine("AgentCore Gateway", "API-Invocations", inv + n_q, "call", P["gateway_call"], "estimated",
                 "ツールの呼び出し（Lambda の回数）と、1問ごとのツールの一覧（ListTools）1回"),
        CostLine("AgentCore Memory", "STM-Ingest", n_q * 2 * 12 * 1024 / GB, "GB", P["memory_stm_ingest_gb"], "estimated",
                 "1問につき2イベント（質問と答え）× 最小 12 KB"),
    ]
    for ln in q_lines:
        rows.append({"condition": "Q", **ln.to_row()})
    per1000.append({"condition": "Q", "n_images": n_q, "total_known_usd": total(q_lines),
                    "per_1000_known_usd": per_1000(q_lines, n_q), "unknown_lines": ""})

    # ログとトレース（全条件の合計。ロググループの保存量で数える）
    logs = boto3.client("logs", region_name=REGION)
    stored = {"runtime": 0, "spans": 0, "other": 0}
    for page in logs.get_paginator("describe_log_groups").paginate():
        for g in page["logGroups"]:
            name, b = g["logGroupName"], g.get("storedBytes", 0)
            if name.startswith("/aws/bedrock-agentcore/runtimes/vlmlab"):
                stored["runtime"] += b
            elif name == "aws/spans":
                stored["spans"] += b
            elif "Vlmlab" in name or "vlmlab" in name:
                stored["other"] += b
    for ln in [CostLine("CloudWatch Logs", "取り込み（Runtime のログ）", stored["runtime"] / GB, "GB", P["logs_ingest_gb"], "measured", "ロググループの保存量（保持 7日）"),
               CostLine("CloudWatch Logs", "取り込み（Lambda・Step Functions）", stored["other"] / GB, "GB", P["logs_ingest_gb"], "measured", "同上"),
               CostLine("CloudWatch（Transaction Search）", "スパンの取り込み", stored["spans"] / GB, "GB", P["spans_ingest_gb"], "measured",
                        "aws/spans の保存量（アカウント全体。この記事の分がほぼすべて）")]:
        rows.append({"condition": "共通（ログ・トレース）", **ln.to_row()})

    p = Path(a.out)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with p.with_name(p.stem + "_per1000.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per1000[0]))
        w.writeheader()
        w.writerows(per1000)
    for r in per1000:
        print(r)


if __name__ == "__main__":
    main()
