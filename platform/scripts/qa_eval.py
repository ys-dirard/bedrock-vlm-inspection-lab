"""条件 Q：分析エージェントへの10問と、Athena の SQL で別に出した正解の照らし合わせ。

- 正解は、分析ツール（sqlbuild）を使わずに、ここに書いた SQL を Athena で直接流して出す
- 10問は同じセッションで順に聞く（10問目は Memory の短期記憶で前の文脈を使う）
- 合否は、正解の数値（丸めたもの）が答えの文に含まれるかで機械的に決め、決められないもの（画像の一覧など）は
  正解の集合と、答えに出てきた画像 ID の集合で比べる。判定の根拠は qa_eval.csv の列に残す

  AWS_PROFILE=<profile> uv run python scripts/qa_eval.py --out runs/qa
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import boto3

from upload_run import REGION, outputs

TABLE = "vlmlab_platform.results"
RUNS = ("p-nova", "p-qwen", "p-claude")

# (質問, 正解の SQL, 正解から合否を決める種類)
QUESTIONS = [
    ("実験 p-nova の判定の件数と、そのうち解析に失敗した件数を教えてください。",
     f"SELECT count(*) AS n, count_if(pred IS NULL) AS fail FROM {TABLE} WHERE run_name = 'p-nova'", "ints"),
    ("実験 p-qwen の再現率と誤検知率を教えてください。",
     f"SELECT CAST(count_if(label='anomaly' AND pred='NG') AS double)/nullif(count_if(label='anomaly' AND pred IS NOT NULL),0) AS recall, "
     f"CAST(count_if(label='normal' AND pred='NG') AS double)/nullif(count_if(label='normal' AND pred IS NOT NULL),0) AS fpr "
     f"FROM {TABLE} WHERE run_name = 'p-qwen'", "rates"),
    ("実験 p-claude の再現率と誤検知率を教えてください。",
     f"SELECT CAST(count_if(label='anomaly' AND pred='NG') AS double)/nullif(count_if(label='anomaly' AND pred IS NOT NULL),0) AS recall, "
     f"CAST(count_if(label='normal' AND pred='NG') AS double)/nullif(count_if(label='normal' AND pred IS NOT NULL),0) AS fpr "
     f"FROM {TABLE} WHERE run_name = 'p-claude'", "rates"),
    ("p-claude で、品目（category）ごとの再現率を教えてください。",
     f"SELECT category, CAST(count_if(label='anomaly' AND pred='NG') AS double)/nullif(count_if(label='anomaly' AND pred IS NOT NULL),0) AS recall "
     f"FROM {TABLE} WHERE run_name = 'p-claude' GROUP BY category ORDER BY category", "rates"),
    ("p-qwen で NG と判定した画像は何枚ですか。",
     f"SELECT count_if(pred='NG') AS n FROM {TABLE} WHERE run_name = 'p-qwen'", "ints"),
    ("p-claude で、不良品なのに OK と判定した画像（見逃し）を、画像 ID ですべて挙げてください。",
     f"SELECT image_id FROM {TABLE} WHERE run_name = 'p-claude' AND label = 'anomaly' AND pred = 'OK' ORDER BY image_id", "ids"),
    ("p-claude で、良品なのに NG と判定した画像は何枚ですか。",
     f"SELECT count_if(label='normal' AND pred='NG') AS n FROM {TABLE} WHERE run_name = 'p-claude'", "ints"),
    ("p-nova、p-qwen、p-claude の、それぞれの費用の合計（USD）を教えてください。",
     f"SELECT run_name, sum(cost_usd) AS cost FROM {TABLE} WHERE run_name IN ('p-nova','p-qwen','p-claude') GROUP BY run_name ORDER BY run_name", "money"),
    ("p-claude の画像1枚あたりの費用の平均（USD）はいくらですか。",
     f"SELECT avg(cost_usd) AS cost FROM {TABLE} WHERE run_name = 'p-claude'", "money"),
    ("さっきの3つの実験のうち、再現率がいちばん高かったのはどれで、その値はいくつですか。",
     f"SELECT run_name, CAST(count_if(label='anomaly' AND pred='NG') AS double)/nullif(count_if(label='anomaly' AND pred IS NOT NULL),0) AS recall "
     f"FROM {TABLE} WHERE run_name IN ('p-nova','p-qwen','p-claude') GROUP BY run_name ORDER BY recall DESC NULLS LAST LIMIT 1", "best"),
]


def athena(sql: str, wg: str) -> list[dict]:
    a = boto3.client("athena", region_name=REGION)
    qid = a.start_query_execution(QueryString=sql, WorkGroup=wg)["QueryExecutionId"]
    while True:
        st = a.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if st in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.5)
    if st != "SUCCEEDED":
        raise RuntimeError(f"{st}: {sql}")
    rows = a.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    head = [c.get("VarCharValue") for c in rows[0]["Data"]]
    return [dict(zip(head, [c.get("VarCharValue") for c in r["Data"]])) for r in rows[1:]]


def numbers_in(text: str) -> list[float]:
    # 英字・/・_・- のすぐ後ろの数字（pcb1、画像 ID の 085 など）は数値として拾わない
    return [float(x.replace(",", "").rstrip(".")) for x in re.findall(r"(?<![A-Za-z/_\-\d.])\d[\d,]*\.?\d*", text)]


def judge(kind: str, truth: list[dict], answer: str) -> tuple[bool, str]:
    nums = numbers_in(answer)

    def has(v: float, tol: float) -> bool:
        """答えの数を1回ずつしか使わない（100 と 100 が要るときに、100 が1つでは足りない）。"""
        for i, n in enumerate(nums):
            if abs(n - v) <= tol:
                del nums[i]
                return True
        return False

    if kind == "ints":
        need = [int(v) for r in truth for v in r.values()]
        return all(has(v, 0) for v in need), f"need {need}"
    if kind == "rates":
        vals = [float(v) for r in truth for k, v in r.items() if k not in ("category",) and v is not None]
        # 0.xxx と xx.x% のどちらで書かれてもよい（小数3桁、百分率は小数1桁の丸めを許す）
        ok = all(has(v, 0.0015) or has(v * 100, 0.15) for v in vals)
        return ok, f"need {vals} (or %)"
    if kind == "money":
        vals = [float(r["cost"]) for r in truth]
        ok = all(has(v, max(0.0005, v * 0.01)) for v in vals)  # 1% か 0.0005 USD まで
        return ok, f"need {vals} (±1%)"
    if kind == "ids":
        need = {r["image_id"] for r in truth}
        got = set(re.findall(r"(?:pcb1|capsules)/(?:anomaly|normal)/\d+", answer))
        return got == need, f"need {sorted(need)} got {sorted(got)}"
    if kind == "best":
        r = truth[0]
        v = float(r["recall"])
        ok = r["run_name"] in answer and (has(v, 0.0015) or has(v * 100, 0.15))
        return ok, f"need {r['run_name']} {v}"
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", default="qa-production-20261120")
    a = ap.parse_args()
    out = outputs()
    client = boto3.client("bedrock-agentcore", region_name=REGION)
    session = a.session.ljust(33, "0")
    rows = []
    for i, (q, sql, kind) in enumerate(QUESTIONS, 1):
        truth = athena(sql, out["AthenaWorkGroup"])
        t0 = time.perf_counter()
        resp = client.invoke_agent_runtime(agentRuntimeArn=out["AnalystArn"], runtimeSessionId=session,
                                           payload=json.dumps({"prompt": q, "session_id": session}).encode("utf-8"),
                                           contentType="application/json", accept="application/json")
        body = json.loads(resp["response"].read())
        wall = time.perf_counter() - t0
        ok, why = judge(kind, truth, body.get("answer", ""))
        rows.append({"no": i, "question": q, "answer": body.get("answer", ""), "truth": json.dumps(truth, ensure_ascii=False),
                     "correct": ok, "basis": why, "n_tool_calls": body.get("n_tool_calls"),
                     "input_tokens": body.get("input_tokens"), "output_tokens": body.get("output_tokens"),
                     "latency_s": body.get("latency_s"), "wall_s": round(wall, 3), "cost_usd": body.get("cost_usd")})
        print(i, ok, why, "|", body.get("answer", "")[:120].replace("\n", " "))
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "qa_eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("correct", sum(r["correct"] for r in rows), "/", len(rows),
          "cost_usd", round(sum(r["cost_usd"] or 0 for r in rows), 5))


if __name__ == "__main__":
    main()
