"""CDK の構成のテスト（synth だけ。AWS は呼ばない）。契約は docs/human_review.md の CI-IR1〜IR6。"""

from __future__ import annotations

import json

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from human_review_stack import HumanReviewConfig, HumanReviewStack

ALLOWED_TYPES = {
    "AWS::S3::Bucket", "AWS::S3::BucketPolicy", "AWS::DynamoDB::Table", "AWS::Logs::LogGroup",
    "AWS::IAM::Role", "AWS::IAM::Policy", "AWS::Lambda::Function", "AWS::Lambda::Url",
    "AWS::StepFunctions::StateMachine",
}


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App()
    stack = HumanReviewStack(app, "T", env=cdk.Environment(region="ap-northeast-1"))
    return Template.from_stack(stack)


def _actions(template: Template) -> list[tuple[str, str]]:
    out = []
    for lid, res in template.find_resources("AWS::IAM::Policy").items():
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            out += [(lid, a) for a in acts]
    return out


def test_only_expected_resource_types(template):
    """CI-IR1：作るのは決めた種類だけ（カスタムリソース、VPC、NAT Gateway、EventBridge、SNS は作らない）。"""
    types = {r["Type"] for r in template.to_json()["Resources"].values()}
    assert types <= ALLOWED_TYPES, types - ALLOWED_TYPES
    template.resource_count_is("AWS::Lambda::Function", 2)


def test_no_wildcard_actions_or_resources(template):
    """CI-IR2：アクションに * を使わない。リソースが "*" の文は無い。"""
    for lid, a in _actions(template):
        assert "*" not in a, (lid, a)
    for res in template.find_resources("AWS::IAM::Policy").values():
        for st in res["Properties"]["PolicyDocument"]["Statement"]:
            assert st["Resource"] != "*"


def test_bedrock_permission_limited_to_profile(template):
    """CI-IR3：Bedrock は InvokeModel だけ。基盤モデルへの許可は jp の推論プロファイル経由に限る。"""
    acts = [a for _, a in _actions(template) if a.startswith("bedrock:")]
    assert set(acts) == {"bedrock:InvokeModel"}
    template.has_resource_properties("AWS::IAM::Policy", {"PolicyDocument": {"Statement": Match.array_with([
        Match.object_like({"Sid": "InvokeModelViaProfile", "Resource": [
            "arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.nova-2-lite-v1:0",
            "arn:aws:bedrock:ap-northeast-3::foundation-model/amazon.nova-2-lite-v1:0"],
            "Condition": {"StringEquals": {"bedrock:InferenceProfileArn": Match.any_value()}}})])}})


def test_s3_read_only_getobject(template):
    """CI-IR2：Lambda の S3 の権限は GetObject だけ（一覧・書き込みは無い）。"""
    s3_acts = {a for _, a in _actions(template) if a.startswith("s3:")}
    assert s3_acts == {"s3:GetObject"}


def test_function_url_iam_and_private_bucket(template):
    """CI-IR4：関数 URL は AWS_IAM。バケットは公開をすべて止め、TLS 以外を拒否する。"""
    template.has_resource_properties("AWS::Lambda::Url", {"AuthType": "AWS_IAM"})
    template.resource_count_is("AWS::Lambda::Permission", 0)  # 公開の許可（Principal: *）を作らない
    template.has_resource_properties("AWS::S3::Bucket", {"PublicAccessBlockConfiguration": {
        "BlockPublicAcls": True, "BlockPublicPolicy": True, "IgnorePublicAcls": True, "RestrictPublicBuckets": True}})


def test_free_tier_friendly_table_and_logs(template):
    """CI-IR5：DynamoDB はプロビジョンド 1/1（無料枠の対象）。ロググループは7日で消え、cdk destroy で消える。"""
    template.has_resource_properties("AWS::DynamoDB::Table", {
        "ProvisionedThroughput": {"ReadCapacityUnits": 1, "WriteCapacityUnits": 1}})
    for res in template.find_resources("AWS::Logs::LogGroup").values():
        assert res["Properties"]["RetentionInDays"] == 7 and res["DeletionPolicy"] == "Delete"
    for t in ("AWS::DynamoDB::Table", "AWS::S3::Bucket"):
        for res in template.find_resources(t).values():
            assert res["DeletionPolicy"] == "Delete"


def test_state_machine_waits_for_token_with_timeout(template):
    """CI-IR6：Standard。人の確認は putItem.waitForTaskToken で待ち、TimeoutSeconds があり、
    States.Timeout は RecordTimeout へ、判定の失敗は RecordJudgeError（自動で OK にしない）へ行く。"""
    sm = list(template.find_resources("AWS::StepFunctions::StateMachine").values())[0]
    parts = sm["Properties"]["DefinitionString"]["Fn::Join"][1]
    text = "".join(p if isinstance(p, str) else "X" for p in parts)
    d = json.loads(text)
    assert sm["Properties"].get("StateMachineType", "STANDARD") == "STANDARD"
    ask = d["States"]["AskHuman"]
    assert ask["Resource"].endswith(":states:::aws-sdk:dynamodb:putItem.waitForTaskToken")
    assert ask["TimeoutSeconds"] == 3600
    assert ask["Parameters"]["Item"]["task_token"] == {"S.$": "$$.Task.Token"}
    assert ask["Catch"] == [{"ErrorEquals": ["States.Timeout"], "ResultPath": "$.error", "Next": "RecordTimeout"}]
    assert d["States"]["Judge"]["Catch"][0]["Next"] == "RecordJudgeError"
    route = d["States"]["Route"]
    assert route["Default"] == "UnknownRoute"
    assert {c["StringEquals"]: c["Next"] for c in route["Choices"]} == {"human": "AskHuman", "auto": "RecordAuto"}
    assert d["States"]["ValidDecision"]["Default"] == "InvalidDecision"


def test_invalid_route_policy_fails_at_synth():
    """CI-IR6：回し方の設定が壊れていれば synth で止まる（Lambda と同じ検証）。"""
    with pytest.raises(ValueError):
        HumanReviewStack(cdk.App(), "Bad", config=HumanReviewConfig(route_policy={"kind": "confidence_band", "lo": 0.9, "hi": 0.1}))
    with pytest.raises(ValueError):
        HumanReviewConfig.from_context({"function_url_auth": "OPEN"})


def test_lambda_asset_contains_only_runtime_files(tmp_path):
    """CI-IR7：Lambda に載せるのは vlmlab/__init__.py・parse.py と review の4つの .py だけ（pandas などを使うものは載せない）。"""
    from pathlib import Path

    app = cdk.App(outdir=str(tmp_path))
    HumanReviewStack(app, "A", env=cdk.Environment(region="ap-northeast-1"))
    asm = app.synth()
    dirs = [p for p in Path(asm.directory).iterdir() if p.is_dir() and p.name.startswith("asset.")]
    assert len(dirs) == 1  # 2つの Lambda は同じ asset を使う
    files = sorted(str(p.relative_to(dirs[0])).replace("\\", "/") for p in dirs[0].rglob("*") if p.is_file())
    assert files == ["vlmlab/__init__.py", "vlmlab/parse.py", "vlmlab/review/__init__.py",
                     "vlmlab/review/judge_handler.py", "vlmlab/review/review_handler.py", "vlmlab/review/route.py"]
