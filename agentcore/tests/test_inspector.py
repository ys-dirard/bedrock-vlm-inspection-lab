"""検査エージェント（前処理ツール・判定の流れ・画像の符号化）。AWS は呼ばない。"""

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

import judge
from agentlab.images import encode_image
from agentlab.runcfg import new_run_config
from preprocess import Preprocessor, build_code, validate_params


def png(w=300, h=200, color=(120, 130, 140), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format=fmt)
    return buf.getvalue()


class FakeSandbox:
    def __init__(self, fail=False):
        self.uploaded, self.codes, self.stopped, self.fail = {}, [], False, fail

    def start(self, identifier=None, name=None, session_timeout_seconds=None):
        self.timeout = session_timeout_seconds
        return "sid"

    def upload_file(self, path, content, description=""):
        self.uploaded[path] = content
        return {}

    def execute_code(self, code, language="python", clear_context=False):
        self.codes.append(code)
        if self.fail:
            return {"stream": [{"result": {"isError": True, "structuredContent": {"stderr": "boom", "exitCode": 1}}}]}
        return {"stream": [{"result": {"structuredContent": {"stdout": '{"op": "zoom"}', "exitCode": 0}}}]}

    def download_file(self, path):
        return b"\x89PNG..."

    def stop(self):
        self.stopped = True
        return True


def test_validate_params():
    assert validate_params("contrast") == {"op": "contrast"}
    assert validate_params("zoom", 0.1, 0.2, 0.3, 0.4)["w"] == 0.3
    for bad in [("rotate",), ("zoom", 0.9, 0.0, 0.5, 0.5), ("zoom", 0.0, 0.0, 0.01, 0.5), ("zoom", "0", 0, 1, 1),
                ("zoom", True, 0, 1, 1)]:
        with pytest.raises(ValueError):
            validate_params(*bad)


def test_build_code_embeds_params_as_json_literal():
    code = build_code({"op": "zoom", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})
    first = code.splitlines()[0]
    assert first.startswith("PARAMS = ") and json.loads(json.loads(first[len("PARAMS = "):]))["op"] == "zoom"


def test_preprocessor_lazy_start_cap_and_close():
    sb = FakeSandbox()
    pre = Preprocessor(lambda: sb, b"T", b"R", max_calls=2)
    assert not pre.sandbox_used
    r = pre.run("zoom", 0.1, 0.1, 0.5, 0.5)
    assert r["status"] == "success" and r["content"][1]["image"]["format"] == "png"
    assert sb.uploaded == {"target.img": b"T", "ref.img": b"R"} and sb.timeout == 300
    assert pre.run("contrast")["status"] == "success"
    capped = pre.run("diff")
    assert capped["status"] == "error" and len(sb.codes) == 2  # 上限を超えたらコードを走らせない
    pre.close()
    assert sb.stopped and pre.sandbox_used


def test_preprocessor_reports_errors_without_raising():
    pre = Preprocessor(lambda: FakeSandbox(fail=True), b"T", None, max_calls=3)
    assert pre.run("contrast")["status"] == "error"
    assert pre.run("diff")["status"] == "error"  # 見本がないと diff はできない
    assert pre.run("zoom", 2, 0, 1, 1)["status"] == "error"


def cfg(**kw):
    base = dict(run_name="p-nova", model_id="m", prompt_version="v1_agent", prompt="判定して",
                input_per_mtok=0.396, output_per_mtok=3.311)
    base.update(kw)
    return new_run_config(**base)


class FakeAgent:
    def __init__(self, tools, fail=False):
        self.tools, self.fail, self.seen = tools, fail, None
        self.event_loop_metrics = SimpleNamespace(accumulated_usage={"inputTokens": 50, "outputTokens": 0},
                                                  cycle_count=1, tool_metrics={})

    def __call__(self, blocks):
        self.seen = blocks
        if self.fail:
            raise RuntimeError("ThrottlingException")
        metrics = SimpleNamespace(
            accumulated_usage={"inputTokens": 1500, "outputTokens": 80, "cacheReadInputTokens": 0},
            cycle_count=2, tool_metrics={"preprocess": SimpleNamespace(call_count=1)})
        return SimpleNamespace(message={"content": [{"text": '{"judgement": "OK", "confidence": 0.9}'}]},
                               metrics=metrics, stop_reason="end_turn")


def run(c, fail=False):
    holder = {}

    def factory(cfg_, tools):
        holder["agent"] = FakeAgent(tools, fail)
        return holder["agent"]

    out = judge.run_judge(c, png(), png(color=(1, 2, 3)), lambda: FakeSandbox(), agent_factory=factory)
    return out, holder["agent"]


def test_run_judge_message_order_and_usage():
    out, agent = run(cfg())
    kinds = [next(iter(b)) for b in agent.seen]
    assert kinds == ["image", "image", "text"]  # 見本 → 検査対象 → 指示
    assert len(agent.tools) == 1
    assert (out.input_tokens, out.output_tokens, out.n_cycles, out.n_tool_calls) == (1500, 80, 2, 1)
    assert out.raw_text.startswith('{"judgement"') and out.api_error is None


def test_run_judge_without_tools_and_refs():  # PL-D3
    out, agent = run(cfg(tools_enabled=False, n_refs=0))
    assert agent.tools == [] and [next(iter(b)) for b in agent.seen] == ["image", "text"]


def test_run_judge_error_keeps_partial_usage():  # CI-PL-R4
    out, _ = run(cfg(), fail=True)
    assert out.api_error.startswith("RuntimeError") and out.raw_text is None and out.input_tokens == 50


def test_encode_image_matches_november_loader(tmp_path):
    from vlmlab.data import load_image
    raw = png(1500, 1000, fmt="JPEG")
    p = tmp_path / "a.jpg"
    p.write_bytes(raw)
    for edge in (None, 768, 2000):
        a, b = encode_image(raw, edge), load_image(p, edge)
        assert (a.data, a.format, a.width, a.height) == (b.data, b.format, b.width, b.height)


def test_prompt_cache_only_when_configured():
    assert "cache_config" not in judge.model_kwargs(cfg())
    kw = judge.model_kwargs(cfg(prompt_cache=True))
    assert kw["cache_config"].strategy == "auto"


def test_cache_tokens_are_priced():  # CI-PL-R3（キャッシュの読み書きも費用に入れる）
    from agentlab.record import AgentOutcome, ImageMeta, build_record
    c = cfg(prompt_cache=True, cache_read_per_mtok=0.33, cache_write_per_mtok=4.125)
    meta = ImageMeta("p-nova", "i", "pcb1", "normal", "k")
    out = AgentOutcome('{"judgement": "OK"}', 1000, 100, 10000, 2000, 1.0, "end_turn", 3, 2, True)
    rec = build_record(c, meta, out, None, "t")
    assert rec["prompt_cache"] is True
    assert abs(rec["cost_usd"] - (1000 * 0.396 + 100 * 3.311 + 10000 * 0.33 + 2000 * 4.125) / 1e6) < 1e-12


def test_parallel_tool_calls_open_one_sandbox_and_close_it():  # deviations.md の 12（2026-11-20）
    import threading
    import time

    made = []

    class SlowSandbox(FakeSandbox):
        def start(self, identifier=None, name=None, session_timeout_seconds=None):
            time.sleep(0.05)  # 開くのに時間がかかる間に、2つ目の呼び出しが来る
            return super().start(identifier, name, session_timeout_seconds)

    def factory():
        sb = SlowSandbox()
        made.append(sb)
        return sb

    pre = Preprocessor(factory, b"T", b"R", max_calls=3)
    ts = [threading.Thread(target=pre.run, args=(op,)) for op in ("contrast", "diff")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    pre.close()
    assert len(made) == 1 and made[0].stopped and len(made[0].codes) == 2 and pre.calls == 2


def test_extra_text_goes_between_images_and_instruction():  # CI-AC-M8
    holder = {}

    def factory(cfg_, tools):
        holder["agent"] = FakeAgent(tools)
        return holder["agent"]

    judge.run_judge(cfg(tools_enabled=False), png(), png(color=(1, 2, 3)), lambda: FakeSandbox(),
                    agent_factory=factory, extra_text="参考の事例")
    seen = holder["agent"].seen
    assert [next(iter(b)) for b in seen] == ["image", "image", "text", "text"]
    assert seen[2]["text"] == "参考の事例" and seen[3]["text"] == "判定して"
    out, agent = run(cfg())
    assert [next(iter(b)) for b in agent.seen] == ["image", "image", "text"]  # 無ければ今までと同じ
