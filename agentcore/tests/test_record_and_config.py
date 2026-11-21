"""記録の形と実験の設定（CI-PL-R*、CI-PL-G*）。AWS は呼ばない。"""

import json

import pytest

from agentlab.record import AgentOutcome, ImageMeta, build_record, results_key, to_line
from agentlab.runcfg import PlatformRunConfig, new_run_config, prompt_sha
from vlmlab.run import make_record  # noqa: F401  （11月のキーと比べるため、モジュールが読めることも確かめる）

V1_JUDGE = "あなたは製品の外観検査員です。\n"

MONTH11_KEYS = {
    "run_name", "run_id", "repeat", "image_id", "category", "label", "path",
    "model_id", "service_tier_requested", "prompt_version", "prompt_sha", "n_refs", "image_long_edge",
    "temperature", "max_tokens", "top_p", "layout", "ref_note", "additional_fields",
    "service_tier_resolved", "raw_text", "parsed", "parse_error", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "latency_s", "stop_reason", "cost_usd", "attempts",
    "api_error", "server_latency_ms", "timestamp",
}


def cfg(**kw):
    base = dict(run_name="p-nova", model_id="jp.amazon.nova-2-lite-v1:0", prompt_version="v1_agent",
                prompt=V1_JUDGE, input_per_mtok=0.396, output_per_mtok=3.311,
                cache_read_per_mtok=0.099, cache_write_per_mtok=0.0)
    base.update(kw)
    return new_run_config(**base)


META = ImageMeta.from_s3_metadata(
    "inbox/p-nova/pcb1__anomaly__000.png",
    {"runname": "p-nova", "imageid": "pcb1/anomaly/000", "category": "pcb1", "label": "anomaly"})


def outcome(text='{"judgement": "NG", "defect": "傷", "confidence": 0.8}', **kw):
    base = dict(raw_text=text, input_tokens=2000, output_tokens=100, cache_read_tokens=0,
                cache_write_tokens=0, latency_s=3.2, stop_reason="end_turn", n_cycles=2,
                n_tool_calls=1, used_code_interpreter=True)
    base.update(kw)
    return AgentOutcome(**base)


def test_record_has_all_november_keys_and_extras():  # CI-PL-R1, R2
    rec = build_record(cfg(), META, outcome(), "s" * 36, "2026-11-22T10:00:00+09:00")
    assert MONTH11_KEYS <= set(rec)
    assert rec["repeat"] == 1 and rec["path"] == META.key
    assert rec["pred"] == "NG" and rec["confidence"] == 0.8 and rec["defect"] == "傷"
    assert rec["used_code_interpreter"] is True and rec["tools_enabled"] is True


def test_cost_uses_total_tokens():  # CI-PL-R3
    rec = build_record(cfg(), META, outcome(), None, "t")
    assert rec["cost_usd"] == pytest.approx((2000 * 0.396 + 100 * 3.311) / 1_000_000)


def test_agent_error_keeps_tokens_and_cost():  # CI-PL-R4
    rec = build_record(cfg(), META, outcome(text="partial", api_error="ThrottlingException: x"), None, "t")
    assert rec["raw_text"] is None and rec["parsed"] is None and rec["pred"] is None
    assert rec["parse_error"] is None and rec["api_error"].startswith("ThrottlingException")
    assert rec["cost_usd"] > 0


def test_parse_failure_is_recorded_not_rescued():  # CI-P1, CI-P3 をそのまま使う
    rec = build_record(cfg(), META, outcome(text='判定: {"judgement": "NG"}'), None, "t")
    assert rec["pred"] is None and rec["parse_error"]


def test_trailing_fence_only_when_configured():  # CI-P7
    text = '説明です。\n```json\n{"judgement": "OK", "confidence": 0.9}\n```'
    assert build_record(cfg(), META, outcome(text=text), None, "t")["pred"] is None
    c = cfg(parse={"allow_trailing_code_fence": True})
    assert build_record(c, META, outcome(text=text), None, "t")["pred"] == "OK"


def test_line_is_single_json_object():  # CI-PL-R6
    line = to_line(build_record(cfg(), META, outcome(), None, "t"))
    assert "\n" not in line
    obj = json.loads(line)
    assert isinstance(obj["parsed"], str) and json.loads(obj["parsed"])["judgement"] == "NG"


def test_run_name_mismatch_stops():
    with pytest.raises(ValueError):
        build_record(cfg(run_name="p-qwen"), META, outcome(), None, "t")


def test_label_must_be_known_and_metadata_complete():  # CI-PL-R5
    with pytest.raises(ValueError):
        ImageMeta.from_s3_metadata("k", {"runname": "r", "imageid": "i", "category": "c", "label": "x"})
    with pytest.raises(ValueError):
        ImageMeta.from_s3_metadata("k", {"runname": "r", "imageid": "i"})


def test_results_key():
    assert results_key("2026-11-22", "p-nova", "pcb1/anomaly/000") == \
        "results/dt=2026-11-22/p-nova__pcb1__anomaly__000.json"


def test_config_rejects_unknown_keys_bad_name_and_null_price():  # CI-PL-G1, G2
    good = json.loads(cfg().to_json())
    with pytest.raises(ValueError):
        PlatformRunConfig.from_dict({**good, "extra": 1})
    with pytest.raises(ValueError):
        PlatformRunConfig.from_dict({**good, "run_name": "a b"})
    with pytest.raises(ValueError):
        PlatformRunConfig.from_dict({**good, "input_per_mtok": None})


def test_run_id_fixed_once_and_roundtrip():  # CI-PL-G3
    c = cfg()
    assert PlatformRunConfig.from_json(c.to_json()).run_id == c.run_id
    with pytest.raises(ValueError):
        new_run_config(run_id="x", run_name="r", model_id="m", prompt_version="v", prompt="p",
                       input_per_mtok=1, output_per_mtok=1)


def test_prompt_sha_matches_november_definition():  # CI-PL-G4
    import hashlib
    expected = hashlib.sha256(json.dumps([V1_JUDGE, None, {}], ensure_ascii=False, sort_keys=True)
                              .encode("utf-8")).hexdigest()[:12]
    assert prompt_sha(V1_JUDGE) == expected
