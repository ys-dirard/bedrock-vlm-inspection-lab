"""11/30 の記事：検査画像のフルサーバーレス分析基盤（ap-northeast-1）。

作らないもの: VPC、NAT Gateway、OpenSearch Serverless、SageMaker のエンドポイント、Cognito、harness。
すべてのリソースにタグ project=advent-2026、article=2026-11-30 を付ける（app.py）。
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_athena as athena
from aws_cdk import aws_bedrockagentcore as ac
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_glue as glue
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
RETENTION = logs.RetentionDays.ONE_WEEK
DB_NAME, TABLE_NAME = "vlmlab_platform", "results"
ENTRY = ["opentelemetry-instrument", "main.py"]

# 判定に使うモデル（facts.md F3・F9、pricing.yaml）。jp. の推論プロファイルは東京と大阪に振り分ける
MODELS = {
    "jp.amazon.nova-2-lite-v1:0": "amazon.nova-2-lite-v1:0",
    "jp.anthropic.claude-sonnet-4-6": "anthropic.claude-sonnet-4-6",
}
IN_REGION_MODELS = ["qwen.qwen3-vl-235b-a22b"]  # 東京のリージョンの中だけ（プロファイルなし）


def bedrock_policy(stack: Stack, profiles: dict[str, str], in_region: list[str]) -> iam.PolicyStatement:
    res = []
    for profile, fm in profiles.items():
        res.append(f"arn:aws:bedrock:{stack.region}:{stack.account}:inference-profile/{profile}")
        res += [f"arn:aws:bedrock:{r}::foundation-model/{fm}" for r in ("ap-northeast-1", "ap-northeast-3")]
    res += [f"arn:aws:bedrock:{stack.region}::foundation-model/{m}" for m in in_region]
    return iam.PolicyStatement(actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                               resources=res)


class InspectionPlatformStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kw) -> None:
        super().__init__(scope, cid, **kw)

        # ---- 1. 取り込みと記録の置き場所 ----
        bucket = s3.Bucket(self, "Data", block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                           enforce_ssl=True, encryption=s3.BucketEncryption.S3_MANAGED,
                           event_bridge_enabled=True, removal_policy=RemovalPolicy.DESTROY,
                           auto_delete_objects=True)
        table = ddb.Table(self, "Status", partition_key=ddb.Attribute(name="image_key", type=ddb.AttributeType.STRING),
                          billing_mode=ddb.BillingMode.PAY_PER_REQUEST, removal_policy=RemovalPolicy.DESTROY)

        # ---- 3. Glue のテーブル（パーティション射影）と Athena のワークグループ ----
        db = glue.CfnDatabase(self, "Db", catalog_id=self.account,
                              database_input=glue.CfnDatabase.DatabaseInputProperty(name=DB_NAME))
        cols = [("run_name", "string"), ("run_id", "string"), ("image_id", "string"), ("category", "string"),
                ("label", "string"), ("model_id", "string"), ("prompt_version", "string"), ("pred", "string"),
                ("confidence", "double"), ("defect", "string"), ("parse_error", "string"), ("api_error", "string"),
                ("input_tokens", "bigint"), ("output_tokens", "bigint"), ("cost_usd", "double"),
                ("latency_s", "double"), ("n_tool_calls", "int"), ("used_code_interpreter", "boolean"),
                ("tools_enabled", "boolean"), ("platform_condition", "string"), ("timestamp", "string")]
        results_loc = f"s3://{bucket.bucket_name}/results/"
        tbl = glue.CfnTable(
            self, "ResultsTable", catalog_id=self.account, database_name=DB_NAME,
            table_input=glue.CfnTable.TableInputProperty(
                name=TABLE_NAME, table_type="EXTERNAL_TABLE",
                partition_keys=[glue.CfnTable.ColumnProperty(name="dt", type="string")],
                parameters={"projection.enabled": "true", "projection.dt.type": "date",
                            "projection.dt.format": "yyyy-MM-dd", "projection.dt.range": "2026-11-01,NOW",
                            "storage.location.template": results_loc + "dt=${dt}/", "classification": "json"},
                storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                    location=results_loc,
                    columns=[glue.CfnTable.ColumnProperty(name=n, type=t) for n, t in cols],
                    input_format="org.apache.hadoop.mapred.TextInputFormat",
                    output_format="org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                    serde_info=glue.CfnTable.SerdeInfoProperty(
                        serialization_library="org.openx.data.jsonserde.JsonSerDe",
                        parameters={"ignore.malformed.json": "true"}))))
        tbl.node.add_dependency(db)
        wg = athena.CfnWorkGroup(
            self, "WorkGroup", name="vlmlab-platform", recursive_delete_option=True, state="ENABLED",
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                bytes_scanned_cutoff_per_query=100 * 1024 * 1024,  # 1クエリ 100 MB まで
                enforce_work_group_configuration=True, publish_cloud_watch_metrics_enabled=True,
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=f"s3://{bucket.bucket_name}/athena-results/")))

        # ---- 2. 検査エージェント（Runtime 2つ：idle timeout 900秒と60秒。P-idle 用） ----
        inspector_code = ac.AgentRuntimeArtifact.from_code_asset(
            path=str(DIST / "inspector.zip"), runtime=ac.AgentCoreRuntime.PYTHON_3_12, entrypoint=ENTRY)
        inspectors = {}
        for name, idle in (("main", 900), ("idle60", 60)):
            rt = ac.Runtime(self, f"Inspector-{name}", runtime_name=f"vlmlab_inspector_{name}",
                            agent_runtime_artifact=inspector_code,
                            description=f"検査エージェント（Strands）。idle timeout {idle}秒",
                            lifecycle_configuration=ac.LifecycleConfiguration(
                                idle_runtime_session_timeout=Duration.seconds(idle),
                                max_lifetime=Duration.hours(1)),
                            platform_version=ac.PlatformVersion.V1)
            bucket.grant_read(rt, "inbox/*")
            bucket.grant_read(rt, "refs/*")
            bucket.grant_read(rt, "run-configs/*")
            rt.add_to_role_policy(bedrock_policy(self, MODELS, IN_REGION_MODELS))
            rt.add_to_role_policy(iam.PolicyStatement(
                actions=["bedrock-agentcore:StartCodeInterpreterSession", "bedrock-agentcore:InvokeCodeInterpreter",
                         "bedrock-agentcore:StopCodeInterpreterSession", "bedrock-agentcore:GetCodeInterpreterSession"],
                resources=[f"arn:aws:bedrock-agentcore:{self.region}:aws:code-interpreter/aws.codeinterpreter.v1"]))
            logs.LogGroup(self, f"InspectorLogs-{name}", retention=RETENTION, removal_policy=RemovalPolicy.DESTROY,
                          log_group_name=f"/aws/bedrock-agentcore/runtimes/{rt.agent_runtime_id}-DEFAULT"
                          ).node.add_dependency(rt)
            inspectors[name] = rt

        # ---- 4. 分析：Lambda（決め打ちの SQL）→ Gateway（IAM）→ 分析エージェント、Memory（短期） ----
        fn_logs = logs.LogGroup(self, "AnalyticsToolsLogs", retention=RETENTION, removal_policy=RemovalPolicy.DESTROY)
        fn = lambda_.Function(self, "AnalyticsTools", runtime=lambda_.Runtime.PYTHON_3_12,
                              architecture=lambda_.Architecture.ARM_64, handler="handler.handler",
                              code=lambda_.Code.from_asset(str(DIST / "lambda_analytics")),
                              timeout=Duration.seconds(30), memory_size=256, log_group=fn_logs,
                              environment={"RESULTS_TABLE": f"{DB_NAME}.{TABLE_NAME}",
                                           "ATHENA_WORKGROUP": wg.name})
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults",
                     "athena:StopQueryExecution"],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/{wg.name}"]))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["glue:GetDatabase", "glue:GetTable", "glue:GetPartitions"],
            resources=[f"arn:aws:glue:{self.region}:{self.account}:catalog",
                       f"arn:aws:glue:{self.region}:{self.account}:database/{DB_NAME}",
                       f"arn:aws:glue:{self.region}:{self.account}:table/{DB_NAME}/{TABLE_NAME}"]))
        bucket.grant_read(fn, "results/*")
        bucket.grant_read_write(fn, "athena-results/*")
        fn.add_to_role_policy(iam.PolicyStatement(actions=["s3:GetBucketLocation", "s3:ListBucket"],
                                                  resources=[bucket.bucket_arn]))

        gateway = ac.Gateway(self, "Gateway", gateway_name="vlmlab-analytics",
                             authorizer_configuration=ac.GatewayAuthorizer.using_aws_iam(),  # Cognito を作らない
                             description="分析ツール（読み取りだけ）。受信は IAM（SigV4）")
        S = ac.SchemaDefinitionType
        filt = ac.SchemaDefinition(type=S.OBJECT, description="絞り込み（列の名前: 値）",
                                   properties={c: ac.SchemaDefinition(type=S.STRING)
                                               for c in ("run_name", "model_id", "category", "label", "pred")})
        gateway.add_lambda_target(
            "AnalyticsTarget", gateway_target_name="analytics", lambda_function=fn,
            tool_schema=ac.ToolSchema.from_inline([
                ac.ToolDefinition(name="count_by", description="軸ごとの件数・再現率・誤検知率・費用",
                                  input_schema=ac.SchemaDefinition(type=S.OBJECT, required=["group_by"], properties={
                                      "group_by": ac.SchemaDefinition(type=S.ARRAY, items=ac.SchemaDefinition(type=S.STRING),
                                                                      description="run_name, model_id, category, label, pred から選ぶ"),
                                      "filters": filt})),
                ac.ToolDefinition(name="list_images", description="条件に合う画像の一覧（最大50件）",
                                  input_schema=ac.SchemaDefinition(type=S.OBJECT, properties={
                                      "filters": filt, "limit": ac.SchemaDefinition(type=S.INTEGER)})),
                ac.ToolDefinition(name="compare_models", description="実験（run_name）ごとの指標を並べる（2〜5個）",
                                  input_schema=ac.SchemaDefinition(type=S.OBJECT, required=["run_names"], properties={
                                      "run_names": ac.SchemaDefinition(type=S.ARRAY, items=ac.SchemaDefinition(type=S.STRING))})),
            ]))

        memory = ac.Memory(self, "Memory", memory_name="vlmlab_analyst_memory",
                           expiration_duration=Duration.days(7), description="分析エージェントの短期記憶")
        analyst = ac.Runtime(self, "Analyst", runtime_name="vlmlab_analyst",
                             agent_runtime_artifact=ac.AgentRuntimeArtifact.from_code_asset(
                                 path=str(DIST / "analyst.zip"), runtime=ac.AgentCoreRuntime.PYTHON_3_12,
                                 entrypoint=ENTRY),
                             description="分析エージェント（Strands）。ツールは Gateway の向こう",
                             environment_variables={"GATEWAY_URL": gateway.gateway_url or "",
                                                    "MEMORY_ID": memory.memory_id},
                             lifecycle_configuration=ac.LifecycleConfiguration(
                                 idle_runtime_session_timeout=Duration.seconds(300), max_lifetime=Duration.hours(1)),
                             platform_version=ac.PlatformVersion.V1)
        gateway.grant_invoke(analyst)
        memory.grant_read_short_term_memory(analyst)
        memory.grant_write(analyst)
        analyst.add_to_role_policy(bedrock_policy(self, {"jp.amazon.nova-2-lite-v1:0": "amazon.nova-2-lite-v1:0"}, []))
        logs.LogGroup(self, "AnalystLogs", retention=RETENTION, removal_policy=RemovalPolicy.DESTROY,
                      log_group_name=f"/aws/bedrock-agentcore/runtimes/{analyst.agent_runtime_id}-DEFAULT"
                      ).node.add_dependency(analyst)

        # ---- 2. Step Functions（1枚ごとに1回。PL-D1） ----
        head = tasks.CallAwsService(self, "ReadImageMetadata", service="s3", action="headObject",
                                    parameters={"Bucket.$": "$.bucket", "Key.$": "$.key"},
                                    iam_resources=[bucket.arn_for_objects("inbox/*")],
                                    iam_action="s3:GetObject", result_selector={"Metadata.$": "$.Metadata"},
                                    result_path="$.head")
        mark_running = tasks.DynamoPutItem(self, "MarkRunning", table=table, result_path=sfn.JsonPath.DISCARD, item={
            "image_key": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.key")),
            "status": tasks.DynamoAttributeValue.from_string("RUNNING"),
            "run_name": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.head.Metadata.runname")),
            "execution": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.Execution.Name")),
            "started_at": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.Execution.StartTime")),
        })
        new_session = sfn.Pass(self, "NewSessionId", parameters={"id.$": "States.UUID()"}, result_path="$.session")

        def invoke(name: str, rt: ac.Runtime) -> tasks.CallAwsService:
            t = tasks.CallAwsService(
                self, f"Invoke-{name}", service="bedrockagentcore", action="invokeAgentRuntime",
                parameters={"AgentRuntimeArn": rt.agent_runtime_arn, "RuntimeSessionId.$": "$.session.id",
                            "Payload.$": "States.JsonToString($.payload)", "ContentType": "application/json",
                            "Accept": "application/json"},
                iam_resources=[rt.agent_runtime_arn, f"{rt.agent_runtime_arn}/*"],
                iam_action="bedrock-agentcore:InvokeAgentRuntime", result_path="$.invoke",
                task_timeout=sfn.Timeout.duration(Duration.minutes(5)))
            t.add_retry(errors=["BedrockAgentCore.ThrottlingException", "BedrockAgentCore.ServiceQuotaExceededException",
                                "BedrockAgentCore.InternalServerException"],
                        interval=Duration.seconds(5), max_attempts=4, backoff_rate=2.0)
            return t

        payload = sfn.Pass(self, "MakePayload", parameters={"bucket.$": "$.bucket", "key.$": "$.key"},
                           result_path="$.payload")
        # 応答（Response）の形は試運転で確かめる（framing.md U4）。JSON の文字列として読む
        parse = sfn.Pass(self, "ParseResponse", parameters={"out.$": "States.StringToJson($.invoke.Response)"},
                         result_path="$.parsed")
        put = tasks.CallAwsService(self, "WriteRecord", service="s3", action="putObject",
                                   parameters={"Bucket": bucket.bucket_name, "Key.$": "$.parsed.out.results_key",
                                               "Body.$": "$.parsed.out.record_line", "ContentType": "application/json"},
                                   iam_resources=[bucket.arn_for_objects("results/*")], iam_action="s3:PutObject",
                                   result_path=sfn.JsonPath.DISCARD)
        mark_done = tasks.DynamoUpdateItem(
            self, "MarkDone", table=table, result_path=sfn.JsonPath.DISCARD,
            key={"image_key": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.key"))},
            update_expression="SET #s = :s, results_key = :r, summary = :m, session_id = :sid, finished_at = :t",
            expression_attribute_names={"#s": "status"},
            expression_attribute_values={
                ":s": tasks.DynamoAttributeValue.from_string("DONE"),
                ":r": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.parsed.out.results_key")),
                ":m": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.json_to_string(sfn.JsonPath.object_at("$.parsed.out.summary"))),
                ":sid": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.session.id")),
                ":t": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$$.State.EnteredTime")),
            })
        mark_failed = tasks.DynamoUpdateItem(
            self, "MarkFailed", table=table, result_path=sfn.JsonPath.DISCARD,
            key={"image_key": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.key"))},
            update_expression="SET #s = :s, error_info = :e",
            expression_attribute_names={"#s": "status"},
            expression_attribute_values={
                ":s": tasks.DynamoAttributeValue.from_string("FAILED"),
                ":e": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.json_to_string(sfn.JsonPath.object_at("$.error"))),
            }).next(sfn.Fail(self, "Failed"))

        def stop(name: str, rt: ac.Runtime) -> tasks.CallAwsService:
            return tasks.CallAwsService(
                self, f"StopSession-{name}", service="bedrockagentcore", action="stopRuntimeSession",
                parameters={"AgentRuntimeArn": rt.agent_runtime_arn, "RuntimeSessionId.$": "$.session.id"},
                iam_resources=[rt.agent_runtime_arn, f"{rt.agent_runtime_arn}/*"],
                iam_action="bedrock-agentcore:StopRuntimeSession", result_path=sfn.JsonPath.DISCARD)

        # 流れ: メタデータ → RUNNING → セッション ID → どの Runtime か → 判定 → 応答を読む → S3 に記録 → DONE
        #       → （stopsession=1 なら）セッションを終わらせる。どこかで失敗したら FAILED
        done = sfn.Succeed(self, "Done")
        for t in (head, mark_running, put, mark_done):
            t.add_catch(mark_failed, result_path="$.error")
        parse.next(put).next(mark_done)

        which = sfn.Choice(self, "WhichRuntime?")
        stop_choice = sfn.Choice(self, "StopSession?")
        for name, rt in inspectors.items():
            inv = invoke(name, rt)
            inv.add_catch(mark_failed, result_path="$.error")
            which.when(sfn.Condition.string_equals("$.head.Metadata.runtime", name), inv.next(parse))
            st = stop(name, rt)
            st.add_catch(done, result_path="$.stop_error")  # 終わらせられなくても、判定は記録済み
            stop_choice.when(sfn.Condition.and_(
                sfn.Condition.string_equals("$.head.Metadata.stopsession", "1"),
                sfn.Condition.string_equals("$.head.Metadata.runtime", name)), st.next(done))
        which.otherwise(sfn.Fail(self, "UnknownRuntime", cause="metadata 'runtime' must be main or idle60"))
        stop_choice.otherwise(done)  # P-idle（stopsession=0）は終わらせず、idle timeout に任せる
        mark_done.next(stop_choice)
        definition = head.next(mark_running).next(new_session).next(payload).next(which)

        sm_logs = logs.LogGroup(self, "StateMachineLogs", retention=RETENTION, removal_policy=RemovalPolicy.DESTROY)
        sm = sfn.StateMachine(self, "Inspect", state_machine_name="vlmlab-inspect",
                              definition_body=sfn.DefinitionBody.from_chainable(definition),
                              state_machine_type=sfn.StateMachineType.STANDARD, timeout=Duration.minutes(15),
                              logs=sfn.LogOptions(destination=sm_logs, level=sfn.LogLevel.ERROR),
                              tracing_enabled=False)

        # ---- 1. EventBridge：inbox/ に置かれたら1回起動 ----
        events.Rule(self, "OnImage", event_pattern=events.EventPattern(
            source=["aws.s3"], detail_type=["Object Created"],
            detail={"bucket": {"name": [bucket.bucket_name]}, "object": {"key": [{"prefix": "inbox/"}]}}),
            targets=[targets.SfnStateMachine(sm, input=events.RuleTargetInput.from_object({
                "bucket": events.EventField.from_path("$.detail.bucket.name"),
                "key": events.EventField.from_path("$.detail.object.key")}))])

        for k, v in {"BucketName": bucket.bucket_name, "StatusTable": table.table_name,
                     "StateMachineArn": sm.state_machine_arn, "InspectorMainArn": inspectors["main"].agent_runtime_arn,
                     "InspectorIdle60Arn": inspectors["idle60"].agent_runtime_arn,
                     "AnalystArn": analyst.agent_runtime_arn, "GatewayUrl": gateway.gateway_url or "",
                     "MemoryId": memory.memory_id, "AthenaWorkGroup": wg.name}.items():
            CfnOutput(self, k, value=v)
