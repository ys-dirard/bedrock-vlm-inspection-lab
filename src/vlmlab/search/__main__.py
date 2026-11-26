"""python -m vlmlab.search embed CONFIG --data-root DIR [--limit N] [--dry-run]

一覧（manifest）の各行を Nova Multimodal Embeddings で埋め込み、runs/<name>/embeddings.jsonl に残す。
試運転は --limit 5〜10。--dry-run は呼ばずに、件数と費用の見込みだけを出す（テキストの費用は文字数を上限のトークン数とみなす）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import EmbedPrices, EmbedRunConfig, execute_embed, read_manifest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m vlmlab.search")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("config")
    e.add_argument("--data-root", required=True, help="画像の path の根（VisA なら data/visa）")
    e.add_argument("--limit", type=int)
    e.add_argument("--dry-run", action="store_true")
    e.add_argument("--pricing", default="pricing.yaml")
    e.add_argument("--runs-dir", default="runs")
    a = p.parse_args(argv)
    base = Path.cwd()
    cfg = EmbedRunConfig.load(a.config)
    prices = EmbedPrices.load(base / a.pricing, cfg.model_id)
    if a.dry_run:
        m = read_manifest(base / cfg.manifest)
        if a.limit is not None:
            m = m.head(a.limit)
        n_std = int(((m.modality == "image") & (m.get("detail") == "STANDARD_IMAGE")).sum())
        n_doc = int(((m.modality == "image") & (m.get("detail") == "DOCUMENT_IMAGE")).sum())
        chars = int(m[m.modality == "text"]["text"].str.len().sum()) if "text" in m.columns else 0
        est = (n_std * (prices.per_image_usd or 0) + n_doc * (prices.per_document_image_usd or 0)
               + chars * (prices.per_mtok_text_usd or 0) / 1e6)
        print(json.dumps({"items": len(m), "standard_images": n_std, "document_images": n_doc,
                          "text_chars": chars, "estimate_usd_upper": round(est, 6)}, ensure_ascii=False))
        return
    from .nova import NovaMultiEmbedder

    embedder = NovaMultiEmbedder(cfg.model_id, cfg.region, cfg.purpose, cfg.dimension)
    execute_embed(cfg, embedder, prices, Path(a.data_root), base, base / a.runs_dir, a.limit)


if __name__ == "__main__":
    main()
