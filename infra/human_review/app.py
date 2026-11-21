"""cdk synth / cdk deploy の入口。設定は -c で上書きできる（例 -c route_policy='{"kind":"all"}'）。"""

import aws_cdk as cdk

from human_review_stack import HumanReviewConfig, HumanReviewStack

app = cdk.App()
ctx = {k: app.node.try_get_context(k) for k in (
    "model_id", "foundation_model_id", "model_regions", "prompt_path", "inference", "parse_spec",
    "route_policy", "human_timeout_seconds", "function_url_auth")}
cfg = HumanReviewConfig.from_context({k: v for k, v in ctx.items() if v is not None})
HumanReviewStack(app, "VlmHumanReview1214", config=cfg, env=cdk.Environment(region="ap-northeast-1"),
                 description="12/14 article: human review with Step Functions task tokens (delete after use)")
cdk.Tags.of(app).add("project", "advent-2026-12-14")
app.synth()
