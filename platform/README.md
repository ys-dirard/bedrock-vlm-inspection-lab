# platform：検査画像のフルサーバーレス分析基盤

画像を S3 の `inbox/` に置くと、AgentCore Runtime の検査エージェント（Strands Agents）が判定し、結果を S3 と DynamoDB に記録します。記録は Athena で読め、分析エージェントが自然文の質問に答えます。費用を測るための検証基盤で、本番の運用を想定したものではありません。

契約は `../docs/contracts.md` の「platform」（CI-PL-*）にあります。

## 構成

```
platform/
├── cdk/                      CDK（Python）のアプリ。stack.py に全部のリソース
├── agents/inspector/         検査エージェント（main.py が Runtime の入口、judge.py が本体、preprocess.py が Code Interpreter のツール）
├── agents/analyst/           分析エージェント（Gateway の MCP のツールと、Memory の短期記憶）
├── lambdas/analytics_tools/  分析のツール（count_by・list_images・compare_models。SQL は決め打ちで、値は Athena のパラメーター）
├── src/platformlab/          AWS に依存しない部品（記録の形、実験の設定、SQL、費用の内訳、画像の縮小）
├── scripts/                  ZIP を作る、手元で1枚判定する、条件を流す、記録を集める
└── tests/                    単体テスト（AWS は呼ばない）
```

流れ（1枚ごと）: S3 `inbox/` → EventBridge → Step Functions →（SDK 統合 `invokeAgentRuntime`）検査エージェント → S3 `results/dt=YYYY-MM-DD/` と DynamoDB →（`stopRuntimeSession`）。分析: 分析エージェント → Gateway（IAM 認証）→ Lambda → Athena。

作らないもの: VPC、NAT Gateway、OpenSearch Serverless、SageMaker のエンドポイント、Cognito。

## 必要なもの

- uv、Node.js（CDK の CLI は `package.json` の開発用の依存として入れ、`npx cdk` で使います）
- ap-northeast-1 で、Nova 2 Lite・Claude Sonnet 4.6（JP の推論プロファイル）・Qwen3 VL のモデルを使えること

## 使い方

```bash
cd platform
uv sync
npm install
uv run pytest                                   # 単体テスト
uv run python scripts/make_subset.py            # ../data/splits/visa_platform_100.csv と _claude50.csv
uv run python scripts/build_zips.py             # dist/ に arm64 の ZIP（Runtime の直接のデプロイ用）と Lambda のコード
npx cdk synth

# 手元で1枚（Runtime は使わない。Bedrock と Code Interpreter は本物を呼ぶ）
AWS_PROFILE=<profile> uv run python scripts/local_judge.py --image-id pcb1/anomaly/033 --visa-root <VisA のフォルダ>

# デプロイ（初回は npx cdk bootstrap aws://<account>/ap-northeast-1 が要る）
AWS_PROFILE=<profile> npx cdk deploy

# 条件を流す（5枚ずつ置いて、終わるのを待つ。--budget を超えそうなら止まる）
AWS_PROFILE=<profile> uv run python scripts/upload_run.py --run-name p-nova --split ../data/splits/visa_platform_100.csv \
    --model jp.amazon.nova-2-lite-v1:0 --visa-root <VisA のフォルダ> --budget 0.5 --condition P-nova
AWS_PROFILE=<profile> uv run python scripts/collect_run.py --run-name p-nova --out ../runs/platform
cd .. && uv run vlmlab report runs/platform/p-nova --out runs/platform/report
```

## 消し方

```bash
AWS_PROFILE=<profile> npx cdk destroy
```

S3 のバケットは中身ごと消えます（`auto_delete_objects`）。Runtime が自動で作るロググループ以外のロググループは、スタックで作って保持期間を7日にしています。消したあと、`/aws/bedrock-agentcore/` と `/aws/lambda/` に残りがないか確かめてください。

## 注意

- Runtime の ZIP は Windows でも作れるように、依存は `uv pip install --python-platform aarch64-manylinux2014` で集め、ZIP の中の権限（644／755）を明示しています。Pillow は 13 に manylinux2014 の arm64 の wheel がないため、12 系になります（手元のテストは 13 で実行）
- Step Functions が受け取る `invokeAgentRuntime` の応答（`Response`）は JSON の文字列として読んでいます。形が違えば、`cdk/stack.py` の `ParseResponse` を直します
