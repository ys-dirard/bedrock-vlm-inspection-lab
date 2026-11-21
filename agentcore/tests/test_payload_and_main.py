"""Runtime の入口のペイロード（CI-AC-P1〜P5）と、入口 handle の流れ。AWS は呼ばない。"""

import base64
import json
from dataclasses import asdict

import pytest

import main
from agentlab.payload import build_payload, parse_payload
from agentlab.runcfg import new_run_config
from test_inspector import FakeSandbox, png


def cfg(**kw):
    base = dict(run_name="ci-haiku", model_id="m", prompt_version="v1_agent", prompt="判定して",
                input_per_mtok=1.10, output_per_mtok=5.50)
    base.update(kw)
    return new_run_config(**base)


def good(**kw):
    p = build_payload(cfg(), "pcb1/anomaly/000", "pcb1", "anomaly", png(), png(color=(1, 2, 3)))
    p.update(kw)
    return p


def test_roundtrip():
    c = cfg()
    req = parse_payload(build_payload(c, "pcb1/normal/001", "pcb1", "normal", b"T", b"R"))
    assert req.cfg == c and req.target == b"T" and req.ref == b"R"
    assert (req.meta.run_name, req.meta.image_id, req.meta.label, req.meta.key) == (
        "ci-haiku", "pcb1/normal/001", "normal", "payload:pcb1/normal/001")


def test_keys_must_match():  # CI-AC-P1
    with pytest.raises(ValueError):
        parse_payload([1, 2])
    p = good()
    del p["label"]
    with pytest.raises(ValueError, match="missing"):
        parse_payload(p)
    with pytest.raises(ValueError, match="unknown"):
        parse_payload(good(extra=1))


def test_run_config_is_validated():  # CI-AC-P2
    rc = asdict(cfg())
    rc["input_per_mtok"] = None
    with pytest.raises(ValueError):
        parse_payload(good(run_config=rc))
    rc = asdict(cfg())
    rc["unknown_key"] = 1
    with pytest.raises(ValueError):
        parse_payload(good(run_config=rc))


def test_label_and_ids():  # CI-AC-P3
    for bad in ({"label": "NG"}, {"image_id": ""}, {"category": 3}):
        with pytest.raises(ValueError):
            parse_payload(good(**bad))


def test_base64_is_strict():  # CI-AC-P4
    for bad in ("", "not base64!!", 123):
        with pytest.raises(ValueError):
            parse_payload(good(image_b64=bad))


def test_ref_required_only_with_refs():  # CI-AC-P5
    p = good()
    del p["ref_b64"]
    with pytest.raises(ValueError):
        parse_payload(p)
    c0 = cfg(n_refs=0)
    p0 = build_payload(c0, "i", "pcb1", "normal", b"T", None)
    assert parse_payload(p0).ref is None
    p0["ref_b64"] = base64.b64encode(b"R").decode()
    with pytest.raises(ValueError):
        parse_payload(p0)


def test_handle_returns_record_and_never_calls_model_on_bad_payload(monkeypatch):
    calls = []

    def fake_run_judge(c, target, ref, factory):
        calls.append((c.run_name, target[:4], ref is not None))
        from agentlab.record import AgentOutcome
        return AgentOutcome('{"judgement": "NG", "confidence": 0.7}', 100, 20, 0, 0, 1.5, "end_turn", 1, 0, False)

    monkeypatch.setattr(main, "run_judge", fake_run_judge)
    bad = main.handle({"run_config": {}}, "sid", sandbox_factory=FakeSandbox)
    assert "error" in bad and calls == []
    out = main.handle(good(), "sid-123", sandbox_factory=FakeSandbox)
    rec = json.loads(out["record_line"])
    assert rec["pred"] == "NG" and rec["label"] == "anomaly" and rec["runtime_session_id"] == "sid-123"
    assert out["summary"]["cost_usd"] == pytest.approx((100 * 1.10 + 20 * 5.50) / 1e6)
    assert len(calls) == 1 and calls[0][2] is True


def test_label_is_not_passed_to_judge():  # CI-PL-R5
    import inspect
    assert list(inspect.signature(main.run_judge).parameters)[:4] == ["cfg", "target_raw", "ref_raw",
                                                                       "sandbox_factory"]
