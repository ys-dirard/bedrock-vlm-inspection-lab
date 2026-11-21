"""検査エージェントの本体（Runtime にも手元にも依存しない部分）。

- 画像は [良品の見本, 検査対象, 指示] の順で1通の user メッセージにする（11月の layout refs_first と同じ）
- ツールあり（Nova・Claude）は preprocess を渡す。ツールなし（Qwen）は何も渡さない（PL-D3）
- トークン数は全ターンの合計（CI-PL-R3）
"""

from __future__ import annotations

import time
from typing import Any, Callable

from agentlab.images import EncodedImage, encode_image
from agentlab.record import AgentOutcome
from agentlab.runcfg import PlatformRunConfig

from preprocess import Preprocessor, Sandbox


def build_message(cfg: PlatformRunConfig, target: EncodedImage, ref: EncodedImage | None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if ref is not None and cfg.n_refs > 0:
        blocks.append({"image": {"format": ref.format, "source": {"bytes": ref.data}}})
    blocks.append({"image": {"format": target.format, "source": {"bytes": target.data}}})
    blocks.append({"text": cfg.prompt})
    return blocks


def final_text(message: dict[str, Any] | None) -> str | None:
    if not message:
        return None
    texts = [b["text"] for b in message.get("content", []) if isinstance(b, dict) and "text" in b]
    return "".join(texts) if texts else None


def usage_of(metrics: Any) -> dict[str, int]:
    u = dict(getattr(metrics, "accumulated_usage", {}) or {})
    return {
        "input": int(u.get("inputTokens", 0) or 0),
        "output": int(u.get("outputTokens", 0) or 0),
        "cache_read": int(u.get("cacheReadInputTokens", 0) or 0),
        "cache_write": int(u.get("cacheWriteInputTokens", 0) or 0),
    }


def tool_calls_of(metrics: Any) -> int:
    tm = getattr(metrics, "tool_metrics", {}) or {}
    return sum(int(getattr(m, "call_count", 0) or 0) for m in tm.values())


def model_kwargs(cfg: PlatformRunConfig) -> dict[str, Any]:
    kw: dict[str, Any] = dict(model_id=cfg.model_id, region_name=cfg.region, temperature=cfg.temperature,
                              max_tokens=cfg.max_tokens, streaming=False)
    if cfg.prompt_cache:
        # 毎ターンの最後の user メッセージにキャッシュポイントを入れ、画像とそれまでの会話をキャッシュから読む
        from strands.models.model import CacheConfig
        kw["cache_config"] = CacheConfig(strategy="auto")
    return kw


def make_agent(cfg: PlatformRunConfig, tools: list[Any]):
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    return Agent(model=BedrockModel(**model_kwargs(cfg)), tools=tools, callback_handler=None)


def run_judge(cfg: PlatformRunConfig, target_raw: bytes, ref_raw: bytes | None,
              sandbox_factory: Callable[[], Sandbox], agent_factory=make_agent) -> AgentOutcome:
    from strands import tool

    target = encode_image(target_raw, cfg.image_long_edge)
    ref = encode_image(ref_raw, cfg.image_long_edge) if ref_raw is not None else None
    pre = Preprocessor(sandbox_factory, target.data, ref.data if ref else None, cfg.max_tool_calls)

    @tool
    def preprocess(op: str, x: float = 0.0, y: float = 0.0, w: float = 1.0, h: float = 1.0) -> dict:
        """検査対象の画像を加工して、結果の画像を返します。

        Args:
            op: zoom（一部を拡大）、contrast（コントラストを強める）、diff（良品の見本との差分を色で示す）のどれか。
            x: zoom の左端（画像の幅に対する割合、0〜1）。zoom のときだけ使う。
            y: zoom の上端（画像の高さに対する割合、0〜1）。zoom のときだけ使う。
            w: zoom の幅（割合、0.05〜1）。zoom のときだけ使う。
            h: zoom の高さ（割合、0.05〜1）。zoom のときだけ使う。
        """
        return pre.run(op, x, y, w, h)

    agent = agent_factory(cfg, [preprocess] if cfg.tools_enabled else [])
    started = time.perf_counter()
    result, error = None, None
    try:
        result = agent(build_message(cfg, target, ref))
    except Exception as e:  # noqa: BLE001  記録に残して、Step Functions へは正常に返す（CI-PL-R4）
        error = f"{type(e).__name__}: {e}"[:1000]
    finally:
        pre.close()
    latency = time.perf_counter() - started

    metrics = result.metrics if result is not None else getattr(agent, "event_loop_metrics", None)
    u = usage_of(metrics)
    return AgentOutcome(
        raw_text=final_text(result.message) if result is not None else None,
        input_tokens=u["input"], output_tokens=u["output"],
        cache_read_tokens=u["cache_read"], cache_write_tokens=u["cache_write"],
        latency_s=round(latency, 3),
        stop_reason=str(result.stop_reason) if result is not None else None,
        n_cycles=int(getattr(metrics, "cycle_count", 0) or 0),
        n_tool_calls=tool_calls_of(metrics),
        used_code_interpreter=pre.sandbox_used,
        api_error=error,
    )
