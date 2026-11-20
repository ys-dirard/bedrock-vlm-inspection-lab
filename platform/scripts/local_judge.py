"""検査エージェントを手元で1枚だけ動かす（Runtime は使わない。Bedrock と Code Interpreter は本物を呼ぶ）。

  AWS_PROFILE=<profile> uv run python scripts/local_judge.py --image-id pcb1/anomaly/000 \
      --visa-root C:/.../data/visa [--model jp.amazon.nova-2-lite-v1:0] [--no-tools] [--long-edge 768]

結果（記録と、Code Interpreter の生の応答の一部）を runs/local/ に保存する。費用は数円以内の想定。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "agents" / "inspector")]

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter  # noqa: E402

from judge import run_judge  # noqa: E402
from platformlab.record import ImageMeta, build_record, to_line  # noqa: E402
from platformlab.runcfg import new_run_config  # noqa: E402

JST = timezone(timedelta(hours=9))
LAB = ROOT.parent
PRICES = {  # pricing.yaml（2026-11-05）・facts.md F9（2026-11-20）
    "jp.amazon.nova-2-lite-v1:0": (0.396, 3.311, 0.099, 0.0),
    "jp.anthropic.claude-sonnet-4-6": (3.30, 16.50, None, None),
    "qwen.qwen3-vl-235b-a22b": (0.64, 3.22, None, None),
}


class LoggingSandbox(CodeInterpreter):
    """Code Interpreter の応答の形を確かめるため、executeCode の生の応答を残す。"""

    log: list = []

    def execute_code(self, code, language="python", clear_context=False):
        r = super().execute_code(code, language, clear_context)
        events = list(r.get("stream", []))
        LoggingSandbox.log.append(events)
        return {**r, "stream": events}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--visa-root", required=True)
    ap.add_argument("--model", default="jp.amazon.nova-2-lite-v1:0")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--long-edge", type=int)
    ap.add_argument("--prompt", default=None, help="既定はツールありなら v1_agent、なしなら v1_judge")
    a = ap.parse_args()

    test = pd.read_csv(LAB / "data" / "splits" / "visa_platform_100.csv").set_index("image_id")
    refs = pd.read_csv(LAB / "data" / "splits" / "visa_refs.csv")
    row = test.loc[a.image_id]
    ref = refs[(refs.category == row.category) & (refs.ref_rank == 1)].iloc[0]
    version = a.prompt or ("v1_judge" if a.no_tools else "v1_agent")
    prompt = (LAB / "prompts" / f"{version}.txt").read_text(encoding="utf-8")
    pin, pout, pcr, pcw = PRICES[a.model]
    cfg = new_run_config(run_name="local", model_id=a.model, prompt_version=version, prompt=prompt,
                         input_per_mtok=pin, output_per_mtok=pout, cache_read_per_mtok=pcr, cache_write_per_mtok=pcw,
                         tools_enabled=not a.no_tools, image_long_edge=a.long_edge, platform_condition="local",
                         parse={"allow_code_fence": True, "allow_trailing_code_fence": True})
    visa = Path(a.visa_root)
    target_raw, ref_raw = (visa / row.path).read_bytes(), (visa / ref.path).read_bytes()
    out = run_judge(cfg, target_raw, ref_raw, lambda: LoggingSandbox("ap-northeast-1"))
    meta = ImageMeta("local", a.image_id, row.category, row.label, str(row.path))
    rec = build_record(cfg, meta, out, None, datetime.now(JST).isoformat(timespec="seconds"))

    out_dir = ROOT / "runs" / "local"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now(JST):%Y%m%d-%H%M%S}_{a.image_id.replace('/', '__')}"
    (out_dir / f"{stem}.json").write_text(to_line(rec) + "\n", encoding="utf-8")
    (out_dir / f"{stem}_ci_events.json").write_text(
        json.dumps(LoggingSandbox.log, ensure_ascii=False, default=str, indent=1)[:20000], encoding="utf-8")
    keys = ["label", "pred", "confidence", "parse_error", "api_error", "input_tokens", "output_tokens",
            "cost_usd", "latency_s", "n_cycles", "n_tool_calls", "used_code_interpreter"]
    print(json.dumps({k: rec[k] for k in keys}, ensure_ascii=False, indent=1))
    print("raw_text:", rec["raw_text"])
    print("saved:", out_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
