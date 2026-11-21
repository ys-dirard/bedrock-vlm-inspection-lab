"""12/18：カスタムの評価器 vlmlab_reason_consistency を作る・確かめる・消す（評価はまだ流さない）。

作るもの（12/18 の results/resources_plan.md の既定の案。2026-11-21 書き手が承認）:
- LLM-as-a-judge、TRACE レベル。判定理由（assistant_turn）が、前処理の結果（context のツールの出力）と矛盾しないかを見る
- 判定のモデルは Claude Haiku 4.5（jp.）、temperature 0、maxTokens 500
- 尺度は categorical（合う・合わない・判断できない）
評価器を作るだけでは課金されない（課金は評価1回ごと。料金ページ）。何を確かめるか（項目）は書き手が未決で、決まったら update する。

  uv run python scripts/create_evaluator.py create
  uv run python scripts/create_evaluator.py status
  uv run python scripts/create_evaluator.py delete
"""

from __future__ import annotations

import argparse
import json

import boto3

REGION = "ap-northeast-1"
NAME = "vlmlab_reason_consistency"
MODEL = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"
TAGS = {"project": "advent-2026", "article": "2026-12-18"}

INSTRUCTIONS = """あなたは外観検査エージェントの記録を点検する担当者です。
エージェントは、検査対象の画像を見て OK／NG を判定し、必要なら前処理のツール（zoom・contrast・diff）の結果を見ています。
画像そのものはここには含まれません。文字で書かれた記録だけを根拠にしてください。

次の点を確かめてください。
- 最後の応答に書かれた判定の理由が、ツールの呼び出しとその結果（文字の部分）と矛盾していないか
- 理由の中で、ツールを使っていないのに使ったかのように書いていないか
- 記録だけでは判断できないときは、無理に決めずに「判断できない」を選ぶ

これまでのやり取りとツールの呼び出し: {context}
評価する応答: {assistant_turn}"""

CONFIG = {"llmAsAJudge": {
    "instructions": INSTRUCTIONS,
    "ratingScale": {"categorical": [
        {"label": "合う", "definition": "判定の理由が、ツールの呼び出しと結果の記録と矛盾しない"},
        {"label": "合わない", "definition": "判定の理由が、ツールの呼び出しや結果の記録と矛盾する、または使っていないツールの結果を根拠にしている"},
        {"label": "判断できない", "definition": "記録の文字だけでは、矛盾があるかを決められない"},
    ]},
    "modelConfig": {"bedrockEvaluatorModelConfig": {
        "modelId": MODEL, "inferenceConfig": {"maxTokens": 500, "temperature": 0.0}}},
}}


def ctl():
    return boto3.client("bedrock-agentcore-control", region_name=REGION)


def find(c) -> dict | None:
    for page in c.get_paginator("list_evaluators").paginate():
        for e in page.get("evaluators", []):
            if e.get("evaluatorName") == NAME:
                return e
    return None


def create() -> None:
    c = ctl()
    if find(c):
        print("evaluator exists")
        return
    r = c.create_evaluator(evaluatorName=NAME, level="TRACE", evaluatorConfig=CONFIG,
                           description="12/18 reason vs tool-output consistency (text only)", tags=TAGS)
    print(json.dumps({k: v for k, v in r.items() if k != "ResponseMetadata"}, default=str))


def status() -> None:
    e = find(ctl())
    print(json.dumps(e, default=str, ensure_ascii=False) if e else "no evaluator")


def delete() -> None:
    c = ctl()
    e = find(c)
    if e:
        c.delete_evaluator(evaluatorId=e["evaluatorId"])
        print("deleted", e["evaluatorId"])
    else:
        print("no evaluator")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["create", "status", "delete"])
    {"create": create, "status": status, "delete": delete}[ap.parse_args().action]()
