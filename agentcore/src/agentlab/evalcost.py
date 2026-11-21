"""12/18：評価1回の費用（docs/contracts.md の CI-AC-S4）。

単価は東京（ap-northeast-1）、AWS Price List API（AmazonBedrockAgentCore、publicationDate 2026-11-20T19:11:34Z）と
料金ページ https://aws.amazon.com/bedrock/agentcore/pricing/ （2026-11-21 確認、一致）。
"""

from __future__ import annotations

BUILTIN_IN_PER_MTOK = 2.40    # APN1-Evaluations:Consumption-based:BuiltIn-Input:Tier1
BUILTIN_OUT_PER_MTOK = 12.00  # APN1-Evaluations:Consumption-based:BuiltIn-Output:Tier1
CUSTOM_PER_EVAL = 0.0015      # APN1-Evaluations:Consumption-based:CustomEvaluators:Tier1


def eval_cost(kind: str, input_tokens: int, output_tokens: int, judge_in_per_mtok: float | None = None,
              judge_out_per_mtok: float | None = None) -> float:
    """組み込み: トークン数 × 組み込みの単価（モデル代込み）。カスタム: 1回の料金 ＋ 判定のモデルのトークン数 × モデルの単価。"""
    if kind == "builtin":
        return (input_tokens * BUILTIN_IN_PER_MTOK + output_tokens * BUILTIN_OUT_PER_MTOK) / 1e6
    if kind == "custom":
        if judge_in_per_mtok is None or judge_out_per_mtok is None:
            raise ValueError("judge model prices are required for a custom evaluator (CI-C5)")
        return CUSTOM_PER_EVAL + (input_tokens * judge_in_per_mtok + output_tokens * judge_out_per_mtok) / 1e6
    raise ValueError(f"kind must be builtin or custom: {kind!r}")
