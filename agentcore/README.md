# agentcore：12/16〜12/18 の検査エージェント（AgentCore）

判定の本体は 11/30 の `platform/agents/inspector`（lab `ac/11-30`）から持ってきた（docs/contracts.md の「agentcore」）。12/16 では Runtime に載せ、手元のスクリプトから `InvokeAgentRuntime` で直接呼ぶ。

```
agentcore/
├── agents/inspector/   main.py（Runtime の入口。ペイロードで画像を受け取る）、judge.py（本体）、preprocess.py（Code Interpreter のツール）
├── src/agentlab/       AWS に依存しない部品（payload・record・runcfg・images）
├── scripts/            build_zip.py、deploy_runtime.py（作る・確かめる・消す）、invoke_runtime.py（画像を流す）
└── tests/              単体テスト（AWS は呼ばない）
```

## 使い方

```bash
cd agentcore
uv sync
uv run pytest
uv run python scripts/build_zip.py                   # dist/inspector.zip（arm64）
uv run python scripts/deploy_runtime.py create        # ロール・ZIP・Runtime（作る前に名前と権限を書き手に見せる）
uv run python scripts/invoke_runtime.py --run-name ci-haiku-tools-trial --split ../data/splits/visa_platform_claude50.csv \
    --limit 5 --model jp.anthropic.claude-haiku-4-5-20251001-v1:0 --long-edge 768 --prompt-cache \
    --visa-root <VisA のフォルダ> --budget 0.12
uv run python scripts/deploy_runtime.py delete        # Runtime・ロール・ZIP を消す（ロググループは別に消す）
```

記録は `runs/agentcore/<run_name>.jsonl`（コミットしない）。集計は `uv run vlmlab report`（lab の main と同じ）。
