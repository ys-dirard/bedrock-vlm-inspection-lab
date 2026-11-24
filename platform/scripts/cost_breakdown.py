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
    # S3 のイベント1件 = APN1-Event-64K-Chunks 1件。Cost Explorer の 2026-11-20 の実績（293件で 0.000293 USD）と
    # Pricing API（2026-11-25、1.00 USD／100万件）で確認（idle_cost.md）
    "eventbridge_s3_event": 1.00 / 1e6,
}
# 課金される状態遷移 = 状態に入った数（StateEntered）+ 2。セッションを止める経路は StateEntered 12、止めない経路は 11
# （2026-11-25、実行の履歴で確認）。297回の実行（止める 277・止めない 20）で 277×14 + 20×13 = 4,138 となり、
# Cost Explorer の 2026-11-20 の APN1-StateTransition（4,138）と一致した
TRANSITIONS_STOP, TRANSITIONS_NO_STOP = 14, 13
NO_STOP_CONDITIONS = {"P-idle-900", "P-idle-60"}
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
        src = "estimated" if cond.startswith("P-idle") or cond == "P-claude" else "measured"
        basis = {"P-claude": "CloudWatch（Service=AgentCore.CodeInterpreter）の 12:01〜12:11。閉じられなかったセッションの分を含む。P-idle の分（P-nova の1枚あたり × 20）を差し引いた",
                 "P-idle-900": "P-nova の1枚あたりの値 × 10枚（同じ時間帯の P-claude の残りと分けられないため）",
                 "P-idle-60": "P-nova の1枚あたりの値 × 10枚（同上）"}.get(cond, "CloudWatch（Service=AgentCore.CodeInterpreter）、条件の時間帯")
        lines += [CostLine("AgentCore Code Interpreter", "vCPU", float(ci["vcpu_hours"]), "vCPU-hour", 0.0895, src, basis),
                  CostLine("AgentCore Code Interpreter", "Memory", float(ci["gb_hours"]), "GB-hour", 0.00945, src, basis)]
    tpe = TRANSITIONS_NO_STOP if cond in NO_STOP_CONDITIONS else TRANSITIONS_STOP
    lines += [
        CostLine("AWS Step Functions（Standard）", "StateTransition", n * tpe, "transition",
                 P["sfn_transition"], "measured", f"実行 {n} 回 × 状態遷移 {tpe}（無料枠 月4,000 は考えない）"),
        CostLine("Amazon EventBridge", "S3 Object Created（opt-in data event）", n, "event", P["eventbridge_s3_event"],
                 "measured", "画像1枚につき1件（Cost Explorer の APN1-Event-64K-Chunks の単価）"),
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
    CI = "code_interpreter(aws.codeinterpreter.v1)"
    for cond, (win, res) in use.items():
        if (win, res) in raw:
            rt[cond] = raw[(win, res)]
    # Code Interpreter（アカウントで共有の組み込み。条件の時間帯で数える）
    # - P-claude は、並列のツール呼び出しで閉じられなかったセッションが 12:10 まで残ったため（deviations.md の 12）、
    #   12:01〜12:11 の窓（P-claude-ci）で数え、同じ時間帯に動いた P-idle の分（見積もり）を差し引く
    # - P-idle の2条件は、P-nova の1枚あたりの値 × 10枚（同じ Nova・同じツールの使い方。見積もり）
    nova_ci, n_nova = raw[("P-nova", CI)], 100
    per_img = {k: float(nova_ci[k]) / n_nova for k in ("vcpu_hours", "gb_hours")}
    for cond in ("P-nova", "P-qwen"):
        rt[cond + "#ci"] = raw[(cond, CI)]
    for cond in ("P-idle-900", "P-idle-60"):
        rt[cond + "#ci"] = {k: v * 10 for k, v in per_img.items()}
    cl = raw[("P-claude-ci", CI)]
    rt["P-claude#ci"] = {k: float(cl[k]) - per_img[k] * 20 for k in ("vcpu_hours", "gb_hours")}
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
    # 数える期間は試運転から本番・Q まで（直したあとの確認 fixcheck の 18:13 より前）。後から作られたほかの記事の
    # Runtime・スパンが混ざらないよう、終わりを固定する（2026-11-25 に直した。deviations.md の 17）
    start, end = datetime(2026, 11, 20, 11, 0, tzinfo=JST), datetime(2026, 11, 20, 18, 0, tzinfo=JST)
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

    # ログとトレース（全条件の合計）。ロググループの保存量（storedBytes）と IncomingBytes の指標は、Runtime のロググループで
    # 0 のままだったため（2026-11-20）、イベントを読み、本文のバイト数 + 1件 26 バイトで数える
    logs = boto3.client("logs", region_name=REGION)

    t0, t1 = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    def event_bytes(prefix: str) -> tuple[int, int]:
        tot = cnt = 0
        for g in [x["logGroupName"] for pg in logs.get_paginator("describe_log_groups").paginate(logGroupNamePrefix=prefix)
                  for x in pg["logGroups"]]:
            for page in logs.get_paginator("filter_log_events").paginate(logGroupName=g, startTime=t0, endTime=t1):
                for e in page["events"]:
                    tot += len(e["message"].encode("utf-8")) + 26
                    cnt += 1
        return tot, cnt

    rt_b, rt_n = (sum(x) for x in zip(*(event_bytes(f"/aws/bedrock-agentcore/runtimes/{r}")
                                        for r in ("vlmlab_inspector_", "vlmlab_analyst"))))
    ot_b, ot_n = event_bytes("VlmlabInspectionPlatform")
    sp_b, sp_n = event_bytes("aws/spans")
    for ln in [CostLine("CloudWatch Logs", "取り込み（Runtime のログ）", rt_b / GB, "GB", P["logs_ingest_gb"], "measured",
                        f"{rt_n} 件。うち約 95% は ADOT の gen_ai.user.message（送った画像が Base64 で入る）"),
               CostLine("CloudWatch Logs", "取り込み（Lambda・Step Functions）", ot_b / GB, "GB", P["logs_ingest_gb"], "measured", f"{ot_n} 件"),
               CostLine("CloudWatch（Transaction Search）", "スパンの取り込み", sp_b / GB, "GB", P["spans_ingest_gb"], "measured",
                        f"aws/spans の {sp_n} 件（11/20 11:00〜18:00 JST。この時間帯はこの記事の分だけ）")]:
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
