"""コマンドライン。

  vlmlab fetch                         VisA の tar を取得して展開する（認証なし）
  vlmlab splits                        data/splits/ の CSV を作る
  vlmlab run CONFIG [--limit N] [--dry-run]
  vlmlab embed CONFIG [--limit N] [--dry-run]   条件 E（埋め込み＋k近傍法）
  vlmlab report RUN_DIR... --out DIR [--tag NAME]
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

    a = p.parse_args(argv)
    if a.cmd == "fetch":
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
        plan = plan_run(cfg, ROOT, ROOT / a.runs_dir, a.limit)
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
        execute(plan, BedrockConverse(cfg.region), pricing, ROOT / a.visa_root, ROOT)
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


if __name__ == "__main__":
    main()
