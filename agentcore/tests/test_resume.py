"""途中から流し直すときの決まり（CI-AC-R1〜R4）。AWS は呼ばない。"""

import json

import pytest

from agentlab.record import AgentOutcome, ImageMeta, build_record, to_line
from agentlab.resume import load_records, resume
from agentlab.runcfg import new_run_config


def cfg(**kw):
    base = dict(run_name="ci-haiku-notools", model_id="m", prompt_version="v1_judge", prompt="判定して",
                input_per_mtok=1.10, output_per_mtok=5.50, tools_enabled=False, image_long_edge=768,
                platform_condition="runtime-direct")
    base.update(kw)
    return new_run_config(**base)


def rec(c, image_id, error=None):
    out = AgentOutcome('{"judgement": "OK"}', 10, 2, 0, 0, 1.0, "end_turn", 1, 0, False, api_error=error)
    return json.loads(to_line(build_record(c, ImageMeta(c.run_name, image_id, "pcb1", "normal", "p"), out, None, "t")))


def test_no_records_keeps_config():  # CI-AC-R1
    c = cfg()
    c2, done = resume(c, [])
    assert c2 == c and done == set()


def test_successful_records_are_skipped_and_run_id_is_kept(tmp_path):  # CI-AC-R1, R3, R4
    old = cfg()
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [rec(old, "a"), rec(old, "b", error="Throttling")]) + "\n",
                 encoding="utf-8")
    new = cfg()  # 新しい run_id で作られる
    c2, done = resume(new, load_records(p))
    assert done == {("a", 1)}  # api_error のあった b は流し直す
    assert c2.run_id == old.run_id != new.run_id


def test_changed_settings_stop():  # CI-AC-R2
    old = cfg()
    for kw in ({"model_id": "other"}, {"prompt": "別の指示"}, {"image_long_edge": None}, {"tools_enabled": True},
               {"prompt_cache": True}, {"temperature": 0.5}):
        with pytest.raises(ValueError, match="CI-AC-R2"):
            resume(cfg(**kw), [rec(old, "a")])
    with pytest.raises(ValueError, match="run_name"):
        resume(cfg(run_name="other"), [rec(old, "a")])


def test_missing_file(tmp_path):
    assert load_records(tmp_path / "none.jsonl") == []
