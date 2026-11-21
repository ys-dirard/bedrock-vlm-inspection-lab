"""請求書の正解と画像を作る（CI-IG1〜IG3）。

- 正解: data/splits/invoice_truth.jsonl（コミットする。架空の値だけ）
- 一覧: data/splits/invoice_120.csv（doc_id, invoice_id, variant, path）。きれいな60枚 → 劣化の60枚の順
- 画像: data/invoice/images/（コミットしない。同じシードとフォントで作り直せる）
- 発行元: data/splits/invoice_issuers.csv（登録番号の照合に使う）
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .render import Fonts, degrade, render
from .truth import SEED, make_all

TRUTH_PATH = "data/splits/invoice_truth.jsonl"
SPLIT_PATH = "data/splits/invoice_120.csv"
ISSUERS_PATH = "data/splits/invoice_issuers.csv"
IMAGES_DIR = "data/invoice/images"
VARIANTS = ("clean", "degraded")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def generate(base_dir: Path, font_path: Path, seed: int = SEED) -> dict[str, Any]:
    truths, issuers = make_all(seed)
    fonts = Fonts(font_path)
    img_dir = base_dir / IMAGES_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 1)  # 劣化の乱数は正解と別の系列
    rows: list[dict[str, str]] = []
    degrade_params: dict[str, Any] = {}
    for t in truths:
        img = render(t, fonts)
        clean = img_dir / f"{t['invoice_id']}_clean.png"
        img.save(clean, format="PNG", optimize=True)
        jpg, params = degrade(img, rng)
        deg = img_dir / f"{t['invoice_id']}_degraded.jpg"
        deg.write_bytes(jpg)
        degrade_params[t["invoice_id"]] = params
    for variant in VARIANTS:
        for t in truths:
            name = f"{t['invoice_id']}_{'clean.png' if variant == 'clean' else 'degraded.jpg'}"
            rows.append({"doc_id": f"{t['invoice_id']}_{variant}", "invoice_id": t["invoice_id"],
                         "variant": variant, "path": f"{IMAGES_DIR}/{name}"})

    truth_path = base_dir / TRUTH_PATH
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    with truth_path.open("w", encoding="utf-8", newline="\n") as f:
        for t in truths:
            f.write(json.dumps({**t, "degrade": degrade_params[t["invoice_id"]]}, ensure_ascii=False) + "\n")
    with (base_dir / SPLIT_PATH).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "invoice_id", "variant", "path"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with (base_dir / ISSUERS_PATH).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "address", "registration_number"], lineterminator="\n")
        w.writeheader()
        w.writerows(issuers)
    manifest = {
        "seed": seed, "n_invoices": len(truths), "n_docs": len(rows),
        "font_file": font_path.name, "font_sha256": sha256_file(font_path),
        "pillow_version": Image.__version__,
        "truth_sha256": sha256_file(truth_path),
        "image_sha256": {r["doc_id"]: sha256_file(base_dir / r["path"]) for r in rows},
    }
    (img_dir.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_truths(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            out[t["invoice_id"]] = t
    return out
