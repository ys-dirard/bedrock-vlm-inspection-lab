"""分析エージェント（Runtime）に質問する（条件 Q で使う）。

  AWS_PROFILE=<profile> uv run python scripts/ask_analyst.py --session q-trial-0001 "trial-qwen-5 の件数は？"

同じ --session（33文字以上になるよう後ろを埋める）で続けて聞くと、Memory の短期記憶で文脈が引き継がれる。
結果は runs/analyst/<session>.jsonl に追記する。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3

from upload_run import REGION, outputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--session", required=True)
    a = ap.parse_args()
    session = a.session.ljust(33, "0")  # runtimeSessionId は 33〜256 文字
    client = boto3.client("bedrock-agentcore", region_name=REGION)
    started = time.perf_counter()
    resp = client.invoke_agent_runtime(agentRuntimeArn=outputs()["AnalystArn"], runtimeSessionId=session,
                                       payload=json.dumps({"prompt": a.question, "session_id": session}).encode("utf-8"),
                                       contentType="application/json", accept="application/json")
    body = json.loads(resp["response"].read())
    rec = {"session": session, "question": a.question, "wall_s": round(time.perf_counter() - started, 3), **body}
    out = ROOT / "runs" / "analyst"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{a.session}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
