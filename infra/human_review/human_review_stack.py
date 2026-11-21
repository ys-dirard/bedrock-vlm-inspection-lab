"""12/14 の記事：グレーな画像だけ人の確認を待つ構成（CDK、Python）。

S3（画像）→ Step Functions（Standard）→ 判定の Lambda（Bedrock Converse）→ Choice
  - 自動：DynamoDB に確定の判定を書く
  - 人へ：DynamoDB に確認待ちを書き、タスクトークンが返るまで待つ（aws-sdk:dynamodb:putItem.waitForTaskToken）
         → 人の判定を書く。時間切れ（States.Timeout）なら timed_out を書く（エスカレーション）
確認画面：Lambda の関数 URL（既定は AWS_IAM 認証）

作らないもの：VPC、NAT Gateway、S3 のイベント通知（実行はローカルのスクリプトが StartExecution で始める）、
自動削除のカスタムリソース（バケットは削除の前に手で空にする。docs/human_review.md）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

LAB_ROOT = Path(__file__).resolve().parents[2]
# Lambda に載せるのは vlmlab/__init__.py・parse.py・review の3つの .py と __init__.py だけ（標準ライブラリと boto3 で動く）
ASSET_EXCLUDE = ["**", "!vlmlab", "!vlmlab/__init__.py", "!vlmlab/parse.py", "!vlmlab/review",
                 "!vlmlab/review/__init__.py", "!vlmlab/review/route.py", "!vlmlab/review/judge_handler.py",
                 "!vlmlab/review/review_handler.py"]


@dataclass(frozen=True)
class HumanReviewConfig:
    model_id: str = "jp.amazon.nova-2-lite-v1:0"
    # 推論プロファイルが振り分ける先のリージョン（aws bedrock get-inference-profile で確認、2026-11-21）
    model_regions: tuple[str, ...] = ("ap-northeast-1", "ap-northeast-3")
    foundation_model_id: str = "amazon.nova-2-lite-v1:0"
    prompt_path: str = "prompts/v1_judge.txt"
    inference: dict = field(default_factory=lambda: {"temperature": 0, "maxTokens": 300})
    parse_spec: dict = field(default_factory=lambda: {"allow_code_fence": True})  # 11/11 の条件 B と同じ
    # 回し方：確信度を使わず、NG と解析失敗は人へ、OK は 50% を抜き取って人へ（2026-11-21 書き手が決定。記事の results/deviations.md の D1-A）
    route_policy: dict = field(default_factory=lambda: {"kind": "sample", "rate": 0.5, "seed": "ac14"})
    human_timeout_seconds: int = 3600
    function_url_auth: str = "AWS_IAM"  # NONE にするなら書き手と相談（claude-code.md）
    log_retention: logs.RetentionDays = logs.RetentionDays.ONE_WEEK

    @classmethod
    def from_context(cls, ctx: dict) -> "HumanReviewConfig":
        kw = {}
        for k in ("model_id", "foundation_model_id", "prompt_path", "function_url_auth"):
            if k in ctx:
                kw[k] = str(ctx[k])
        for k in ("inference", "parse_spec", "route_policy"):
            if k in ctx:
                kw[k] = json.loads(ctx[k]) if isinstance(ctx[k], str) else ctx[k]
        if "model_regions" in ctx:
            v = ctx["model_regions"]
            kw["model_regions"] = tuple(v.split(",") if isinstance(v, str) else v)
        if "human_timeout_seconds" in ctx:
            kw["human_timeout_seconds"] = int(ctx["human_timeout_seconds"])
        cfg = cls(**kw)
        if cfg.function_url_auth not in ("AWS_IAM", "NONE"):
            raise ValueError("function_url_auth must be AWS_IAM or NONE")
        return cfg


class HumanReviewStack(Stack):
    def __init__(self, scope: Construct, cid: str, config: HumanReviewConfig | None = None, **kwargs):
        super().__init__(scope, cid, **kwargs)
        cfg = config or HumanReviewConfig()
        # 設定の誤りは synth の時点で止める（Lambda と同じ検証を使う）
        import sys

        sys.path.insert(0, str(LAB_ROOT / "src"))
        from vlmlab.review.route import RoutePolicy

        RoutePolicy.from_dict(cfg.route_policy)

        bucket = s3.Bucket(
            self, "Images",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,  # 空でないと消えない。削除の前に aws s3 rm で空にする
        )
        table = ddb.Table(
            self, "Reviews",
            partition_key=ddb.Attribute(name="review_id", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PROVISIONED,  # 無料枠（25 RCU / 25 WCU）はプロビジョンドだけが対象
            read_capacity=1, write_capacity=1,
            removal_policy=RemovalPolicy.DESTROY,
        )

        code = lambda_.Code.from_asset(str(LAB_ROOT / "src"), exclude=ASSET_EXCLUDE)
        prompt_text = (LAB_ROOT / cfg.prompt_path).read_text(encoding="utf-8")

        judge_logs = logs.LogGroup(self, "JudgeLogs", retention=cfg.log_retention, removal_policy=RemovalPolicy.DESTROY)
        judge_fn = lambda_.Function(
            self, "JudgeFn",
            runtime=lambda_.Runtime.PYTHON_3_12, architecture=lambda_.Architecture.ARM_64,
            handler="vlmlab.review.judge_handler.handler", code=code,
            memory_size=256, timeout=Duration.seconds(120), log_group=judge_logs,
            environment={
                "MODEL_ID": cfg.model_id,
                "PROMPT_TEXT": prompt_text,
                "INFERENCE_JSON": json.dumps(cfg.inference),
                "PARSE_SPEC_JSON": json.dumps(cfg.parse_spec),
                "ROUTE_POLICY_JSON": json.dumps(cfg.route_policy),
            },
        )
        judge_fn.add_to_role_policy(iam.PolicyStatement(
            sid="ReadImages", actions=["s3:GetObject"], resources=[bucket.arn_for_objects("*")]))
        profile_arn = f"arn:aws:bedrock:{self.region}:{self.account}:inference-profile/{cfg.model_id}"
        judge_fn.add_to_role_policy(iam.PolicyStatement(
            sid="InvokeGeoProfile", actions=["bedrock:InvokeModel"], resources=[profile_arn]))
        judge_fn.add_to_role_policy(iam.PolicyStatement(
            sid="InvokeModelViaProfile", actions=["bedrock:InvokeModel"],
            resources=[f"arn:aws:bedrock:{r}::foundation-model/{cfg.foundation_model_id}" for r in cfg.model_regions],
            conditions={"StringEquals": {"bedrock:InferenceProfileArn": profile_arn}},
        ))

        # ---- 状態機械 ----
        exec_name = sfn.JsonPath.string_at("$$.Execution.Name")
        common_item = {
            "review_id": tasks.DynamoAttributeValue.from_string(exec_name),
            "image_id": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.image_id")),
            "s3_bucket": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.bucket")),
            "s3_key": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.key")),
            "vlm_result": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.json_to_string(sfn.JsonPath.object_at("$.judge"))),
            "started_at": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.Execution.StartTime")),
        }

        judge = tasks.LambdaInvoke(
            self, "Judge", lambda_function=judge_fn, payload_response_only=True, result_path="$.judge",
            payload=sfn.TaskInput.from_object({
                "image_id": sfn.JsonPath.string_at("$.image_id"),
                "bucket": sfn.JsonPath.string_at("$.bucket"),
                "key": sfn.JsonPath.string_at("$.key"),
                "ref_keys": sfn.JsonPath.list_at("$.ref_keys"),
            }),
            retry_on_service_exceptions=True,
        )
        record_auto = tasks.DynamoPutItem(
            self, "RecordAuto", table=table, result_path=sfn.JsonPath.DISCARD,
            item={**common_item,
                  "status": tasks.DynamoAttributeValue.from_string("auto_decided"),
                  "decided_by": tasks.DynamoAttributeValue.from_string("vlm"),
                  "final_judgement": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.judge.auto_judgement")),
                  "decided_at": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.State.EnteredTime"))},
        )
        record_judge_error = tasks.DynamoPutItem(
            self, "RecordJudgeError", table=table, result_path=sfn.JsonPath.DISCARD,
            item={"review_id": tasks.DynamoAttributeValue.from_string(exec_name),
                  "image_id": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.image_id")),
                  "status": tasks.DynamoAttributeValue.from_string("judge_failed"),
                  "error": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.json_to_string(sfn.JsonPath.object_at("$.error"))),
                  "started_at": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.Execution.StartTime"))},
        ).next(sfn.Fail(self, "JudgeFailed", error="JudgeFailed", cause="judge Lambda failed (never auto-OK on errors)"))

        ask_human = tasks.CallAwsService(
            self, "AskHuman", service="dynamodb", action="putItem",
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            iam_resources=[table.table_arn], iam_action="dynamodb:PutItem",
            parameters={"TableName": table.table_name, "Item": {
                "review_id": {"S": exec_name},
                "image_id": {"S": sfn.JsonPath.string_at("$.image_id")},
                "s3_bucket": {"S": sfn.JsonPath.string_at("$.bucket")},
                "s3_key": {"S": sfn.JsonPath.string_at("$.key")},
                "vlm_result": {"S": sfn.JsonPath.json_to_string(sfn.JsonPath.object_at("$.judge"))},
                "started_at": {"S": sfn.JsonPath.string_at("$$.Execution.StartTime")},
                "requested_at": {"S": sfn.JsonPath.string_at("$$.State.EnteredTime")},
                "status": {"S": "pending"},
                "task_token": {"S": sfn.JsonPath.task_token},
            }},
            task_timeout=sfn.Timeout.duration(Duration.seconds(cfg.human_timeout_seconds)),
            result_path="$.human",
        )
        record_human = tasks.DynamoUpdateItem(
            self, "RecordHuman", table=table, result_path=sfn.JsonPath.DISCARD,
            key={"review_id": tasks.DynamoAttributeValue.from_string(exec_name)},
            update_expression="SET #s = :s, decided_by = :by, final_judgement = :f, decided_at = :t REMOVE task_token",
            expression_attribute_names={"#s": "status"},
            expression_attribute_values={
                ":s": tasks.DynamoAttributeValue.from_string("human_decided"),
                ":by": tasks.DynamoAttributeValue.from_string("human"),
                ":f": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.human.decision")),
                ":t": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.State.EnteredTime")),
            },
        )
        record_timeout = tasks.DynamoUpdateItem(
            self, "RecordTimeout", table=table, result_path=sfn.JsonPath.DISCARD,
            key={"review_id": tasks.DynamoAttributeValue.from_string(exec_name)},
            update_expression="SET #s = :s, decided_by = :by, decided_at = :t REMOVE task_token",
            expression_attribute_names={"#s": "status"},
            expression_attribute_values={
                ":s": tasks.DynamoAttributeValue.from_string("timed_out"),
                ":by": tasks.DynamoAttributeValue.from_string("none"),
                ":t": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.State.EnteredTime")),
            },
        )
        valid_decision = sfn.Choice(self, "ValidDecision").when(
            sfn.Condition.or_(sfn.Condition.string_equals("$.human.decision", "OK"),
                              sfn.Condition.string_equals("$.human.decision", "NG")),
            record_human,
        ).otherwise(sfn.Fail(self, "InvalidDecision", error="InvalidDecision", cause="human decision is not OK or NG"))

        ask_human.add_catch(record_timeout, errors=["States.Timeout"], result_path="$.error")
        judge.add_catch(record_judge_error, errors=["States.ALL"], result_path="$.error")
        ask_human.next(valid_decision)
        definition = judge.next(
            sfn.Choice(self, "Route")
            .when(sfn.Condition.string_equals("$.judge.route", "human"), ask_human)
            .when(sfn.Condition.string_equals("$.judge.route", "auto"), record_auto)
            .otherwise(sfn.Fail(self, "UnknownRoute", error="UnknownRoute"))
        )
        machine = sfn.StateMachine(
            self, "Workflow", definition_body=sfn.DefinitionBody.from_chainable(definition),
            state_machine_type=sfn.StateMachineType.STANDARD,
            timeout=Duration.seconds(cfg.human_timeout_seconds + 900),
        )

        # ---- 確認画面 ----
        review_logs = logs.LogGroup(self, "ReviewLogs", retention=cfg.log_retention, removal_policy=RemovalPolicy.DESTROY)
        review_fn = lambda_.Function(
            self, "ReviewFn",
            runtime=lambda_.Runtime.PYTHON_3_12, architecture=lambda_.Architecture.ARM_64,
            handler="vlmlab.review.review_handler.handler", code=code,
            memory_size=256, timeout=Duration.seconds(15), log_group=review_logs,
            environment={"TABLE_NAME": table.table_name, "PRESIGN_SECONDS": "900"},
        )
        review_fn.add_to_role_policy(iam.PolicyStatement(
            sid="ReviewTable", actions=["dynamodb:GetItem", "dynamodb:UpdateItem", "dynamodb:Scan"],
            resources=[table.table_arn]))
        machine.grant_task_response(review_fn)
        review_fn.add_to_role_policy(iam.PolicyStatement(  # 署名付き URL で画像を見せる
            sid="ReadImages", actions=["s3:GetObject"], resources=[bucket.arn_for_objects("*")]))
        url = review_fn.add_function_url(auth_type=lambda_.FunctionUrlAuthType[cfg.function_url_auth])

        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "TableName", value=table.table_name)
        CfnOutput(self, "StateMachineArn", value=machine.state_machine_arn)
        CfnOutput(self, "ReviewUrl", value=url.url)
