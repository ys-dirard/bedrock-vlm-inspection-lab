"""BedrockConverse の再試行（CI-B1）。偽物のクライアントを渡すので AWS は呼ばない。"""

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from vlmlab.bedrock import BedrockConverse, ModelRequest

OK = {"output": {"message": {"content": [{"text": '{"judgement": "OK"}'}]}},
      "usage": {"inputTokens": 10, "outputTokens": 3}, "stopReason": "end_turn"}


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def converse(self, **kwargs):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


REQ = ModelRequest(model_id="m", images=[], text="t")


@pytest.mark.parametrize("err", [client_error("ThrottlingException"),
                                 ReadTimeoutError(endpoint_url="https://example")])
def test_retryable_errors_are_retried_then_succeed(err):
    c = FakeClient([err, err, OK])
    r = BedrockConverse("ap-northeast-1", base_delay_s=0, client=c).invoke(REQ)
    assert c.calls == 3 and r.attempts == 3 and r.api_error is None and r.input_tokens == 10


def test_gives_up_after_max_attempts_without_exception():
    c = FakeClient([client_error("ThrottlingException")] * 3)
    r = BedrockConverse("ap-northeast-1", max_attempts=3, base_delay_s=0, client=c).invoke(REQ)
    assert c.calls == 3 and r.raw_text is None and r.api_error.startswith("ThrottlingException")


def test_non_retryable_error_returns_immediately():
    c = FakeClient([client_error("ValidationException"), OK])
    r = BedrockConverse("ap-northeast-1", base_delay_s=0, client=c).invoke(REQ)
    assert c.calls == 1 and r.api_error.startswith("ValidationException") and r.attempts == 1


# ---- 推論（12/9 の記事：Nova 2 Lite の reasoningConfig）----
from vlmlab.bedrock import (build_converse_kwargs, check_reasoning_constraints, reasoning_effort,
                            response_from_converse, split_content)
from vlmlab.parse import ParseSpec, parse_response

REASONING = {"reasoningContent": {"reasoningText": {"text": "[REDACTED]"}}}
ANSWER = '```json\n{"judgement": "NG", "defect": "scratch", "confidence": 0.8}\n```'


def test_reasoning_block_is_not_part_of_text():
    """CI-B3: 推論のブロックが先にあっても、本文は text のブロックだけになり、判定が読める。"""
    resp = {"output": {"message": {"content": [REASONING, {"text": ANSWER}]}},
            "usage": {"inputTokens": 608, "outputTokens": 412}, "stopReason": "end_turn"}
    r = response_from_converse(resp, 1.0, 1)
    assert r.raw_text == ANSWER and r.output_tokens == 412
    pr = parse_response(r.raw_text, ParseSpec(allow_code_fence=True))
    assert pr.pred == "NG" and pr.confidence == 0.8


def test_reasoning_only_response_has_no_text():
    """CI-B3: 推論の途中で上限に達し text のブロックが無ければ、本文は None（推論の中身を判定として読まない）。"""
    resp = {"output": {"message": {"content": [REASONING]}},
            "usage": {"inputTokens": 608, "outputTokens": 8000}, "stopReason": "max_tokens"}
    r = response_from_converse(resp, 1.0, 1)
    assert r.raw_text is None and r.stop_reason == "max_tokens" and r.extra["reasoning_blocks"] == 1
    assert parse_response(r.raw_text).parse_error == "empty: no text"


def test_split_content_counts_reasoning_and_keeps_text_order():
    """CI-B3, CI-B4: text は出てきた順につなぐ。推論のブロックは数と中身（[REDACTED]）を別に返す。"""
    blocks = [REASONING, {"text": "a"}, {"reasoningContent": {"redactedContent": b"xx"}}, {"text": "b"},
              {"toolUse": {"toolUseId": "1", "name": "n", "input": {}}}]
    assert split_content(blocks) == ("ab", 2, "[REDACTED]<redactedContent>")
    assert split_content([{"text": "{}"}]) == ("{}", 0, None)
    assert split_content([]) == (None, 0, None)


def _req(effort, **kw):
    af = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}} if effort else None
    return ModelRequest(model_id="m", images=[], text="t", additional_fields=af, **kw)


def test_reasoning_low_sends_inference_config_and_additional_fields():
    """CI-B5: low・medium は temperature・maxTokens と一緒に送れる。reasoningConfig は additionalModelRequestFields に入る。"""
    k = build_converse_kwargs(_req("low", temperature=0, max_tokens=8000))
    assert k["inferenceConfig"] == {"temperature": 0, "maxTokens": 8000}
    assert k["additionalModelRequestFields"] == {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}


def test_reasoning_high_omits_inference_config():
    """CI-B5: high は temperature・maxTokens を None にすると inferenceConfig ごと送らない。"""
    k = build_converse_kwargs(_req("high", temperature=None, max_tokens=None))
    assert "inferenceConfig" not in k
    assert k["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"] == "high"


@pytest.mark.parametrize("kw", [dict(temperature=0, max_tokens=None), dict(temperature=None, max_tokens=300),
                                dict(temperature=None, max_tokens=None, top_p=0.9)])
def test_reasoning_high_rejects_sampling_params_before_call(kw):
    """CI-B5: high で temperature・maxTokens・topP のどれかを送ろうとしたら、呼ぶ前に ValueError。"""
    with pytest.raises(ValueError, match="high"):
        build_converse_kwargs(_req("high", **kw))
    c = FakeClient([OK])
    with pytest.raises(ValueError):
        BedrockConverse("ap-northeast-1", base_delay_s=0, client=c).invoke(_req("high", **kw))
    assert c.calls == 0


def test_reasoning_high_rejects_topk():
    """CI-B5: high で additionalModelRequestFields に topK があれば ValueError。"""
    af = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}, "topK": 20}
    with pytest.raises(ValueError, match="topK"):
        check_reasoning_constraints(None, None, None, af)


def test_reasoning_effort_reads_only_enabled():
    """CI-B5: type が enabled のときだけ推論の深さを返す（disabled や未設定は None で、制約もかけない）。"""
    assert reasoning_effort(None) is None
    assert reasoning_effort({"reasoningConfig": {"type": "disabled", "maxReasoningEffort": "high"}}) is None
    assert reasoning_effort({"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}}) == "medium"
    check_reasoning_constraints(0, 300, None, {"reasoningConfig": {"type": "disabled", "maxReasoningEffort": "high"}})
