"""1つの条件（run）を基盤に流す（セッション2で使う。デプロイの後）。

1. 実験の設定を S3 の run-configs/<run_name>.json に置く（単価は pricing.yaml。null なら止める＝CI-C5）
2. 良品の見本（visa_refs.csv の ref_rank=1）を refs/<category>.img に置く
3. 画像を inbox/<run_name>/ に、メタデータ（runname・imageid・category・label・runtime・stopsession）付きで置く
   - --pace 枚ずつ置き、その分が DynamoDB で DONE か FAILED になるまで待ってから次を置く（並列の数を抑える。PL-D1）
   - 置く前に「この run の累計 + これまでで最も高かった1枚」が --budget を超えるなら止める（CI-C4 と同じ考え方）

  AWS_PROFILE=<profile> uv run python scripts/upload_run.py --run-name p-nova-trial5 \
      --split ../data/splits/visa_platform_100.csv --limit 5 --model jp.amazon.nova-2-lite-v1:0 \
      --visa-root C:/.../data/visa --budget 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from platformlab.runcfg import new_run_config  # noqa: E402
from vlmlab.cost import PricingTable  # noqa: E402

REGION = "ap-northeast-1"
STACK = "VlmlabInspectionPlatform"


def outputs() -> dict[str, str]:
    cf = boto3.client("cloudformation", region_name=REGION)
    st = cf.describe_stacks(StackName=STACK)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in st.get("Outputs", [])}


def evenly(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """試運転は、品目×正解の4群から均等に（並べた順に1枚ずつ回す）。"""
    if n is None or n >= len(df):
        return df
    groups = [g.reset_index(drop=True) for _, g in sorted(df.groupby(["category", "label"]))]
    rows, i = [], 0
    while len(rows) < n:
        for g in groups:
            if i < len(g) and len(rows) < n:
                rows.append(g.iloc[i])
        i += 1
    return pd.DataFrame(rows).reset_index(drop=True)


def wait_done(table, keys: list[str], timeout_s: int = 900) -> list[dict]:
    deadline, items = time.monotonic() + timeout_s, {}
    while time.monotonic() < deadline:
        for k in keys:
            if k in items:
                continue
            it = table.get_item(Key={"image_key": k}, ConsistentRead=True).get("Item")
            if it and it.get("status") in ("DONE", "FAILED"):
                items[k] = it
        if len(items) == len(keys):
            return [items[k] for k in keys]
        time.sleep(3)
    raise TimeoutError(f"not finished in {timeout_s}s: {[k for k in keys if k not in items]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--visa-root", required=True)
    ap.add_argument("--budget", type=float, required=True, help="この run の上限（USD）")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--long-edge", type=int)
    ap.add_argument("--prompt", help="既定はツールありなら v1_agent、なしなら v1_judge")
    ap.add_argument("--runtime", choices=["main", "idle60"], default="main")
    ap.add_argument("--stop-session", choices=["0", "1"], default="1")
    ap.add_argument("--condition", default="", help="記録に残す条件の名前（P-nova など）")
    ap.add_argument("--pace", type=int, default=5)
    ap.add_argument("--trailing-fence", action="store_true", help="CI-P7（Claude など）")
    ap.add_argument("--prompt-cache", action="store_true", help="プロンプトキャッシュ（Strands の CacheConfig auto）")
    a = ap.parse_args()

    out = outputs()
    bucket = out["BucketName"]
    s3 = boto3.client("s3", region_name=REGION)
    table = boto3.resource("dynamodb", region_name=REGION).Table(out["StatusTable"])

    price = PricingTable.load(LAB / "pricing.yaml").get(a.model, "standard")
    if not price.runnable:
        sys.exit(f"price is null for {a.model} (CI-C5)")
    version = a.prompt or ("v1_judge" if a.no_tools else "v1_agent")
    cfg = new_run_config(
        run_name=a.run_name, model_id=a.model, prompt_version=version,
        prompt=(LAB / "prompts" / f"{version}.txt").read_text(encoding="utf-8"),
        input_per_mtok=price.input_per_mtok, output_per_mtok=price.output_per_mtok,
        cache_read_per_mtok=price.cache_read_per_mtok, cache_write_per_mtok=price.cache_write_per_mtok,
        price_source=price.source_url, price_checked_on=price.checked_on, tools_enabled=not a.no_tools,
        platform_condition=a.condition, image_long_edge=a.long_edge, prompt_cache=a.prompt_cache,
        parse={"allow_code_fence": True, "allow_trailing_code_fence": a.trailing_fence})
    cfg_key = f"run-configs/{a.run_name}.json"
    try:
        s3.head_object(Bucket=bucket, Key=cfg_key)
        sys.exit(f"{cfg_key} already exists. Use a new --run-name (records of one run share one run_id).")
    except s3.exceptions.ClientError:
        pass
    s3.put_object(Bucket=bucket, Key=cfg_key, Body=cfg.to_json().encode("utf-8"), ContentType="application/json")

    visa = Path(a.visa_root)
    refs = pd.read_csv(LAB / "data" / "splits" / "visa_refs.csv")
    df = evenly(pd.read_csv(a.split), a.limit)
    for cat in sorted(df.category.unique()):
        ref = refs[(refs.category == cat) & (refs.ref_rank == 1)].iloc[0]
        s3.put_object(Bucket=bucket, Key=f"refs/{cat}.img", Body=(visa / ref.path).read_bytes())

    spent, worst, results = 0.0, 0.0, []
    for start in range(0, len(df), a.pace):
        if spent + worst > a.budget:
            print(f"stop: spent {spent:.4f} + worst {worst:.4f} > budget {a.budget}")
            break
        batch, keys = df.iloc[start:start + a.pace], []
        for row in batch.itertuples():
            key = f"inbox/{a.run_name}/{row.image_id.replace('/', '__')}{Path(row.path).suffix.lower()}"
            s3.put_object(Bucket=bucket, Key=key, Body=(visa / row.path).read_bytes(), Metadata={
                "runname": a.run_name, "imageid": row.image_id, "category": row.category, "label": row.label,
                "runtime": a.runtime, "stopsession": a.stop_session})
            keys.append(key)
        for it in wait_done(table, keys):
            summ = json.loads(it["summary"]) if it.get("summary") else {}
            cost = summ.get("cost_usd") or 0.0
            spent, worst = spent + cost, max(worst, cost)
            results.append({"key": it["image_key"], "status": it["status"], **summ})
            print(it["status"], it["image_key"], summ.get("pred"), summ.get("label"), cost, summ.get("api_error"))
    out_dir = ROOT / "runs" / "uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.run_name}.json").write_text(json.dumps({"run_id": cfg.run_id, "spent_usd": spent,
                                                            "items": results}, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(f"run_id={cfg.run_id} images={len(results)} model_cost_usd={spent:.4f}")


if __name__ == "__main__":
    main()
