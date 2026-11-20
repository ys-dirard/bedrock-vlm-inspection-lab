"""CDK のアプリ。cdk.json から `uv run python cdk/app.py` で呼ばれる。"""

import aws_cdk as cdk

from stack import InspectionPlatformStack

app = cdk.App()
stack = InspectionPlatformStack(app, "VlmlabInspectionPlatform",
                                env=cdk.Environment(region="ap-northeast-1"),
                                description="11/30 の記事：検査画像のフルサーバーレス分析基盤（bedrock-vlm-inspection-lab/platform）")
cdk.Tags.of(app).add("project", "advent-2026")
cdk.Tags.of(app).add("article", "2026-11-30")
app.synth()
