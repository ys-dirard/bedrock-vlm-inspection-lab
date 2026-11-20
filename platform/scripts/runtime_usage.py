"""Runtime と Code Interpreter の使用量（CloudWatch の AWS/Bedrock-AgentCore）を、時間帯ごとに集める。

- 指標: CPUUsed-vCPUHours・MemoryUsed-GBHours（5分ごとの合計）。指標は30〜40分ほど遅れて出る（2026-11-20 に確認）
- Runtime は Name の次元（<runtime_name>::DEFAULT）で分ける。Code Interpreter は Service=AgentCore.CodeInterpreter
  （組み込みの aws.codeinterpreter.v1。アカウントで共有なので、ほかの利用と混ざらない時間帯で見る）
- 費用は platformlab.costbreak の単価（V1）で計算する（CI-PL-C5）

  uv run python scripts/runtime_usage.py --window p-nova 2026-11-20T12:00 2026-11-20T12:20 --out runs/runtime_metrics.csv
（時刻は JST。--window は何度でも指定できる）
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
from platformlab.costbreak import RUNTIME_PRICES  # noqa: E402

JST = timezone(timedelta(hours=9))
NS = "AWS/Bedrock-AgentCore"
RUNTIMES = ["vlmlab_inspector_main", "vlmlab_inspector_idle60", "vlmlab_analyst"]
cw = boto3.client("cloudwatch", region_name="ap-northeast-1")


def dims_for(name: str | None) -> list[list[dict]]:
    if name is None:
        flt = [{"Name": "Service", "Value": "AgentCore.CodeInterpreter"}]
    else:
        flt = [{"Name": "Name", "Value": f"{name}::DEFAULT"}]
    ms = cw.list_metrics(Namespace=NS, MetricName="MemoryUsed-GBHours", Dimensions=flt)["Metrics"]
    if name is None:  # Resource=aws.codeinterpreter.v1 の行（Service だけの集計行は除く）
        ms = [m for m in ms if any(d["Name"] == "Resource" for d in m["Dimensions"])]
    return [m["Dimensions"] for m in ms]


def total(metric: str, dims: list[dict], start: datetime, end: datetime) -> float:
    pts = cw.get_metric_statistics(Namespace=NS, MetricName=metric, Dimensions=dims, StartTime=start, EndTime=end,
                                   Period=300, Statistics=["Sum"])["Datapoints"]
    return sum(p["Sum"] for p in pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", nargs=3, action="append", metavar=("LABEL", "START", "END"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = RUNTIME_PRICES["V1"]
    rows = []
    for label, s, e in a.window:
        start = datetime.fromisoformat(s).replace(tzinfo=JST)
        end = datetime.fromisoformat(e).replace(tzinfo=JST)
        for name in RUNTIMES + [None]:
            for dims in dims_for(name):
                cpu = total("CPUUsed-vCPUHours", dims, start, end)
                mem = total("MemoryUsed-GBHours", dims, start, end)
                rows.append({"window": label, "start_jst": s, "end_jst": e,
                             "resource": name or "code_interpreter(aws.codeinterpreter.v1)",
                             "vcpu_hours": round(cpu, 9), "gb_hours": round(mem, 9),
                             "cost_usd_v1": round(cpu * p["vcpu_hour"] + mem * p["gb_hour"], 9)})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
