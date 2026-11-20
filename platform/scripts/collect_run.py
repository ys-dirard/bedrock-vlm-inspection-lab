"""S3 の判定の記録を、11月と同じ実験のディレクトリ（calls.jsonl と config.yaml）に集める（S2）。

集めたあとは、11月の集計をそのまま使う:  uv run vlmlab report runs/<run_name> --out <dir>

  AWS_PROFILE=<profile> uv run python scripts/collect_run.py --run-name p-nova --out ../runs/platform
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
import yaml

from upload_run import REGION, outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bucket = outputs()["BucketName"]
    s3 = boto3.client("s3", region_name=REGION)

    cfg = json.loads(s3.get_object(Bucket=bucket, Key=f"run-configs/{a.run_name}.json")["Body"].read())
    records = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="results/"):
        for obj in page.get("Contents", []):
            if not Path(obj["Key"]).name.startswith(f"{a.run_name}__"):
                continue
            rec = json.loads(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
            if isinstance(rec.get("parsed"), str):
                rec["parsed"] = json.loads(rec["parsed"])  # to_line で文字列にしたものを戻す
            records.append(rec)
    if not records:
        raise SystemExit(f"no records for {a.run_name}")
    if len({r["run_id"] for r in records}) != 1:
        raise SystemExit("records of one run must share one run_id (CI-PL-G3)")
    records.sort(key=lambda r: (r["timestamp"], r["image_id"]))
    run_dir = Path(a.out) / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "calls.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(
        {"name": a.run_name, "model_id": cfg["model_id"], "prompt": cfg["prompt_version"], "parse": cfg["parse"],
         "platform_condition": cfg["platform_condition"], "tools_enabled": cfg["tools_enabled"]},
        allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"{len(records)} records -> {run_dir}")


if __name__ == "__main__":
    main()
