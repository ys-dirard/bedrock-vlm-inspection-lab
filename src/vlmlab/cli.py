"""コマンドライン。

  vlmlab fetch                         VisA の tar を取得して展開する（認証なし）
  vlmlab splits                        data/splits/ の CSV を作る
  vlmlab run CONFIG [--limit N] [--dry-run] [--suffix TAG]
  vlmlab embed CONFIG [--limit N] [--dry-run]   条件 E（埋め込み＋k近傍法）
  vlmlab report RUN_DIR... --out DIR [--tag NAME]
  vlmlab cascade --nova RUN --claude RUN --embeddings JSONL --out DIR   段階判定（11/26 の記事。API は呼ばない）

  MMAD（11/19 の記事）
  vlmlab mmad prepare                  mmad.json を取得し、data/splits/mmad_1050.csv を作り、使う画像だけ取り出す
  vlmlab mmad subsample --per-task N --out CSV     一覧をさらに減らす（record_id は元のまま）
  vlmlab mmad run CONFIG [--limit N] [--dry-run]     オンデマンドで回す（試運転は --limit 10）
  vlmlab mmad batch-prepare CONFIG                   バッチ推論の入力（JSONL）を作る
  vlmlab mmad batch-collect CONFIG OUT_FILE... [--job-arn ARN]   バッチの出力を記録にする
  vlmlab mmad report RUN_DIR... --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path.cwd()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vlmlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--raw-dir", default="data/raw")
    f.add_argument("--visa-root", default="data/visa")

    s = sub.add_parser("splits")
    s.add_argument("--visa-root", default="data/visa")
    s.add_argument("--out", default="data/splits")

    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--limit", type=int, default=None, help="試運転の枚数（runs/<name>__trial<N>/ に出す）")
    r.add_argument("--dry-run", action="store_true", help="Bedrock を呼ばず、組み立てた内容だけ表示する")
    r.add_argument("--suffix", default=None, help="runs/<name>__<suffix>/ に出す（例: 時間帯ごとに day1・night1）")
    r.add_argument("--visa-root", default="data/visa")
    r.add_argument("--pricing", default="pricing.yaml")
    r.add_argument("--runs-dir", default="runs")

    e = sub.add_parser("embed", help="条件 E：埋め込み＋k近傍法")
    e.add_argument("config")
    e.add_argument("--limit", type=int, default=None, help="試運転：テスト画像 N 枚を埋め込むだけ（判定しない）")
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--visa-root", default="data/visa")
    e.add_argument("--pricing", default="pricing.yaml")
    e.add_argument("--runs-dir", default="runs")

    rp = sub.add_parser("report")
    rp.add_argument("run_dirs", nargs="+")
    rp.add_argument("--out", required=True)
    rp.add_argument("--tag", default="summary")

    cs = sub.add_parser("cascade", help="段階判定：記録から回す条件ごとの指標を計算する（11/26 の記事。API は呼ばない）")
    cs.add_argument("--nova", required=True, help="1段目の Nova の実験（例 runs/n1_B）")
    cs.add_argument("--claude", required=True, help="2段目の Claude の実験（例 runs/n3_claude_refB）")
    cs.add_argument("--embeddings", required=True, help="埋め込みの記録（例 runs/n1_E/embeddings.jsonl）")
    cs.add_argument("--out", required=True)

    m = sub.add_parser("mmad", help="MMAD の選択式の問題（11/19 の記事）")
    msub = m.add_subparsers(dest="mcmd", required=True)
    mp = msub.add_parser("prepare")
    mp.add_argument("--mmad-json", default="data/mmad/mmad.json")
    mp.add_argument("--split", default="data/splits/mmad_1050.csv")
    mp.add_argument("--per-task", type=int, default=150)
    mp.add_argument("--images-root", default="data/mmad/images")
    mp.add_argument("--no-images", action="store_true", help="一覧だけ作り、画像は取らない")
    ms = msub.add_parser("subsample", help="抜き出し済みの一覧から、同じ層の分け方でさらに減らす（record_id は元のまま）")
    ms.add_argument("--src", default="data/splits/mmad_1050.csv")
    ms.add_argument("--per-task", type=int, required=True)
    ms.add_argument("--out", required=True)
    mr = msub.add_parser("run")
    mr.add_argument("config")
    mr.add_argument("--limit", type=int, default=None)
    mr.add_argument("--dry-run", action="store_true")
    mr.add_argument("--pricing", default="pricing.yaml")
    mr.add_argument("--runs-dir", default="runs")
    mb = msub.add_parser("batch-prepare")
    mb.add_argument("config")
    mb.add_argument("--runs-dir", default="runs")
    mc = msub.add_parser("batch-collect")
    mc.add_argument("config")
    mc.add_argument("output_files", nargs="+")
    mc.add_argument("--job-arn", default=None)
    mc.add_argument("--pricing", default="pricing.yaml")
    mc.add_argument("--runs-dir", default="runs")
    mrep = msub.add_parser("report")
    mrep.add_argument("run_dirs", nargs="+")
    mrep.add_argument("--out", required=True)
    mrep.add_argument("--subset", default=None, help="この一覧（CSV）の record_id だけで集計する")

    a = p.parse_args(argv)
    if a.cmd == "mmad":
        _mmad(a)
    elif a.cmd == "fetch":
        from .data import download, extract
        tar = download(ROOT / a.raw_dir)
        print(f"tar: {tar}")
        print(f"visa_root: {extract(tar, ROOT / a.visa_root)}")
    elif a.cmd == "splits":
        from .data import write_splits
        for k, v in write_splits(ROOT / a.visa_root, ROOT / a.out).items():
            print(f"{k}: {v}")
    elif a.cmd == "run":
        from .cost import PricingTable
        from .run import RunConfig, plan_run
        cfg = RunConfig.load(a.config)
        plan = plan_run(cfg, ROOT, ROOT / a.runs_dir, a.limit, a.suffix)
        pricing = PricingTable.load(ROOT / a.pricing)
        price = pricing.get(cfg.model_id, cfg.service_tier)
        if a.dry_run:
            print(json.dumps({
                "run_name": plan.run_name, "n_images": len(plan.items), "repeats": cfg.repeats,
                "calls": len(plan.items) * cfg.repeats, "refs_per_image": cfg.n_refs,
                "price_known": price.runnable, "out_dir": str(plan.out_dir),
                "by_group": plan.items.groupby(["category", "label"]).size().to_dict().__repr__(),
            }, ensure_ascii=False, indent=2))
            return
        from .bedrock import BedrockConverse
        from .run import execute
        execute(plan, BedrockConverse(cfg.region, read_timeout_s=cfg.read_timeout_s), pricing, ROOT / a.visa_root, ROOT)
    elif a.cmd == "embed":
        from .embed_run import EmbedConfig, execute_embed, load_embedding_price, plan_images
        cfg = EmbedConfig.load(a.config)
        price = load_embedding_price(ROOT / a.pricing, cfg.model_id)
        run_name, images, _ = plan_images(cfg, ROOT, a.limit)
        if a.dry_run:
            print(json.dumps({
                "run_name": run_name, "calls": len(images), "per_image_usd": price,
                "by_role": images.groupby(["role", "category", "label"]).size().to_dict().__repr__(),
            }, ensure_ascii=False, indent=2))
            return
        from .embed import NovaEmbedder
        embedder = NovaEmbedder(cfg.model_id, cfg.region, cfg.embedding_purpose, cfg.embedding_dimension)
        execute_embed(cfg, embedder, price, ROOT / a.visa_root, ROOT, ROOT / a.runs_dir, a.limit)
    elif a.cmd == "report":
        from .report import build_report
        for k, v in build_report(a.run_dirs, a.out, a.tag).items():
            print(f"{k}: {v}")
    elif a.cmd == "cascade":
        from .cascade import load_embeddings, run_cascade, write_outputs
        from .report import load_run
        res = run_cascade(load_run(a.nova), load_run(a.claude), load_embeddings(a.embeddings))
        for k, v in write_outputs(res, a.out).items():
            print(f"{k}: {v}")
        print(f"best: {res['best']}")


def _mmad(a: argparse.Namespace) -> None:
    from .cost import PricingTable
    from .mmad import data as md
    from .mmad import run as mrun

    if a.mcmd == "prepare":
        path = md.fetch_mmad_json(ROOT / a.mmad_json)
        table = md.question_table(md.load_mmad(path))
        split = md.sample_questions(table, a.per_task)
        out = ROOT / a.split
        out.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(out, index=False, lineterminator="\n")
        print(f"split: {out} ({len(split)} questions)")
        print(split.groupby(["task", "dataset"]).size().unstack().to_string())
        if not a.no_images:
            from .mmad.fetch import fetch_images
            keys = md.needed_images(split, n_shots=1)
            print(f"images needed: {len(keys)}")
            print(fetch_images(keys, ROOT / a.images_root))
    elif a.mcmd == "subsample":
        sub = md.subsample_split(md.load_split(ROOT / a.src), a.per_task)
        out = ROOT / a.out
        sub.to_csv(out, index=False, lineterminator="\n")
        print(f"split: {out} ({len(sub)} questions)")
        print(sub.groupby(["task", "dataset"]).size().unstack().to_string())
    elif a.mcmd == "run":
        cfg = mrun.MmadConfig.load(a.config)
        pricing = PricingTable.load(ROOT / a.pricing)
        price = pricing.get(cfg.model_id, cfg.pricing_tier)
        if a.dry_run:
            split = mrun.select_trial(md.load_split(ROOT / cfg.split), a.limit)
            print(json.dumps({
                "run_name": mrun.run_dir(cfg, ROOT / a.runs_dir, a.limit)[0], "calls": len(split),
                "model_id": cfg.model_id, "api": cfg.api, "price_known": price.runnable,
                "by_task": split.groupby("task").size().to_dict(),
            }, ensure_ascii=False, indent=2))
            return
        if cfg.api == "converse":
            from .bedrock import BedrockConverse
            model = BedrockConverse(cfg.region)
        else:
            from .mmad.invoke import BedrockInvokeAnthropic
            model = BedrockInvokeAnthropic(cfg.region)
        print(mrun.execute_ondemand(cfg, ROOT, ROOT / a.runs_dir, pricing, model, a.limit))
    elif a.mcmd == "batch-prepare":
        cfg = mrun.MmadConfig.load(a.config)
        print(json.dumps(mrun.prepare_batch(cfg, ROOT, ROOT / a.runs_dir), ensure_ascii=False, indent=2))
    elif a.mcmd == "batch-collect":
        cfg = mrun.MmadConfig.load(a.config)
        pricing = PricingTable.load(ROOT / a.pricing)
        files = [Path(f) for f in a.output_files]
        print(mrun.collect_batch(cfg, ROOT, ROOT / a.runs_dir, files, pricing, a.job_arn))
    elif a.mcmd == "report":
        from .mmad.report import build_report
        subset = set(md.load_split(ROOT / a.subset)["record_id"]) if a.subset else None
        for k, v in build_report(a.run_dirs, a.out, subset).items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
