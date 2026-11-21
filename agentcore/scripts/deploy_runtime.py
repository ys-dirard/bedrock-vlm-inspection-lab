"""12/16 の検査エージェントの Runtime を作る・確かめる・消す（boto3。CDK は使わない）。

作るもの（記事の results/resources_plan.md の案A。2026-11-21 書き手が承認）:
- IAM ロール vlmlab-ci-runtime-role（Runtime の実行ロール。インラインポリシー vlmlab-ci-runtime-policy）
- コードの ZIP: CDK の bootstrap のバケット（既存）の vlmlab-ci/inspector.zip
- AgentCore Runtime vlmlab_ci_inspector（ZIP の直接のデプロイ、Python 3.12、V1、idle 60秒、最長 1時間、PUBLIC）

  uv run python scripts/deploy_runtime.py create      # dist/inspector.zip を先に作っておく（scripts/build_zip.py）
  uv run python scripts/deploy_runtime.py status
  uv run python scripts/deploy_runtime.py delete
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3

REGION = "ap-northeast-1"
ROLE, POLICY = "vlmlab-ci-runtime-role", "vlmlab-ci-runtime-policy"
RUNTIME = "vlmlab_ci_inspector"
KEY = "vlmlab-ci/inspector.zip"
ZIP = Path(__file__).resolve().parents[1] / "dist" / "inspector.zip"
TAGS = {"project": "advent-2026", "article": "2026-12-16"}
# 判定に使うモデル（推論プロファイル → 基盤モデル）。jp. は東京と大阪に振り分ける
MODELS = {"jp.anthropic.claude-haiku-4-5-20251001-v1:0": "anthropic.claude-haiku-4-5-20251001-v1:0"}


def trust(acct: str) -> dict:
    return {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": acct},
                      "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock-agentcore:{REGION}:{acct}:*"}}}]}


def policy(acct: str, bucket: str) -> dict:
    models = []
    for profile, fm in MODELS.items():
        models.append(f"arn:aws:bedrock:{REGION}:{acct}:inference-profile/{profile}")
        models += [f"arn:aws:bedrock:{r}::foundation-model/{fm}" for r in ("ap-northeast-1", "ap-northeast-3")]
    logs = f"arn:aws:logs:{REGION}:{acct}:log-group:/aws/bedrock-agentcore/runtimes/*"
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "InvokeModels", "Effect": "Allow",
         "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], "Resource": models},
        {"Sid": "CodeInterpreterBuiltin", "Effect": "Allow",
         "Action": ["bedrock-agentcore:StartCodeInterpreterSession", "bedrock-agentcore:InvokeCodeInterpreter",
                    "bedrock-agentcore:StopCodeInterpreterSession", "bedrock-agentcore:GetCodeInterpreterSession"],
         "Resource": f"arn:aws:bedrock-agentcore:{REGION}:aws:code-interpreter/aws.codeinterpreter.v1"},
        {"Sid": "CodeZip", "Effect": "Allow", "Action": "s3:GetObject",
         "Resource": f"arn:aws:s3:::{bucket}/{KEY}"},
        {"Sid": "RuntimeLogs", "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:DescribeLogStreams", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": [logs, f"{logs}:log-stream:*"]},
        {"Sid": "DescribeLogGroups", "Effect": "Allow", "Action": "logs:DescribeLogGroups",
         "Resource": f"arn:aws:logs:{REGION}:{acct}:log-group:*"},
        {"Sid": "Traces", "Effect": "Allow",
         "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules",
                    "xray:GetSamplingTargets"], "Resource": "*"},
        {"Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
         "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}}},
        {"Sid": "WorkloadIdentity", "Effect": "Allow",
         "Action": ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId"],
         "Resource": [f"arn:aws:bedrock-agentcore:{REGION}:{acct}:workload-identity-directory/default",
                      f"arn:aws:bedrock-agentcore:{REGION}:{acct}:workload-identity-directory/default/"
                      f"workload-identity/{RUNTIME}-*"]},
    ]}


def find_runtime(ctl) -> dict | None:
    for page in ctl.get_paginator("list_agent_runtimes").paginate():
        for r in page.get("agentRuntimes", []):
            if r["agentRuntimeName"] == RUNTIME:
                return r
    return None


def create() -> None:
    if not ZIP.exists():
        sys.exit(f"{ZIP} がありません。scripts/build_zip.py を先に実行してください")
    acct = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"cdk-hnb659fds-assets-{acct}-{REGION}"
    iam = boto3.client("iam")
    try:
        role_arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
        print("role exists")
    except iam.exceptions.NoSuchEntityException:
        role_arn = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust(acct)),
                                   Description="12/16 vlmlab_ci_inspector execution role",
                                   Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()])["Role"]["Arn"]
        print("role created")
    iam.put_role_policy(RoleName=ROLE, PolicyName=POLICY, PolicyDocument=json.dumps(policy(acct, bucket)))
    boto3.client("s3", region_name=REGION).upload_file(str(ZIP), bucket, KEY)
    print(f"uploaded s3://<cdk-assets>/{KEY}")
    ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    if find_runtime(ctl):
        sys.exit(f"{RUNTIME} はすでにあります（更新はこのスクリプトでは扱わない）")
    time.sleep(10)  # 作ったばかりのロールが Runtime から見えるまで待つ
    r = ctl.create_agent_runtime(
        agentRuntimeName=RUNTIME, roleArn=role_arn, description="12/16 検査エージェント（Strands、Code Interpreter）",
        agentRuntimeArtifact={"codeConfiguration": {
            "code": {"s3": {"bucket": bucket, "prefix": KEY}}, "runtime": "PYTHON_3_12",
            "entryPoint": ["opentelemetry-instrument", "main.py"]}},
        networkConfiguration={"networkMode": "PUBLIC"}, protocolConfiguration={"serverProtocol": "HTTP"},
        lifecycleConfiguration={"idleRuntimeSessionTimeout": 60, "maxLifetime": 3600},
        platformVersion="V1", tags=TAGS)
    print("runtime", r["agentRuntimeArn"].replace(acct, "<acct>"), r["status"])
    wait(ctl)


def wait(ctl) -> None:
    for _ in range(60):
        r = find_runtime(ctl)
        print(r["status"] if r else "missing")
        if r is None or r["status"] in ("READY", "CREATE_FAILED", "UPDATE_FAILED"):
            if r and r["status"] != "READY":
                d = ctl.get_agent_runtime(agentRuntimeId=r["agentRuntimeId"])
                print(d.get("failureReason"))
            return
        time.sleep(10)


def status() -> None:
    ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    r = find_runtime(ctl)
    print(json.dumps(r, default=str) if r else "no runtime")


def delete() -> None:
    acct = boto3.client("sts").get_caller_identity()["Account"]
    ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    r = find_runtime(ctl)
    if r:
        ctl.delete_agent_runtime(agentRuntimeId=r["agentRuntimeId"])
        print("runtime deleted", r["agentRuntimeId"])
    iam = boto3.client("iam")
    try:
        iam.delete_role_policy(RoleName=ROLE, PolicyName=POLICY)
        iam.delete_role(RoleName=ROLE)
        print("role deleted")
    except iam.exceptions.NoSuchEntityException:
        print("no role")
    boto3.client("s3", region_name=REGION).delete_object(Bucket=f"cdk-hnb659fds-assets-{acct}-{REGION}", Key=KEY)
    print("zip deleted. ロググループ /aws/bedrock-agentcore/runtimes/<id>-DEFAULT は別に消す")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["create", "status", "delete"])
    {"create": create, "status": status, "delete": delete}[ap.parse_args().action]()
