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
