"""画像1枚あたりの inputTokens を、Bedrock の usage で確かめる（12/4 の早見表）。

目的: 同じ画像を「画像なし」「原寸」「長辺768px」の3通りで各モデルに1回ずつ送り、
      usage.inputTokens の差から、画像1枚が何トークンとして数えられるかを出す。
成功条件: 各モデルで3回とも応答が返り、usage が記録される。費用は単価表から計算する。
拒否条件: 単価が分からないモデルは呼ばない（費用を計算できないため）。
          合計の見込みが --budget-usd を超えるなら呼ばない。

使い方（検証基盤の venv を使う）:
  uv run --project ../../../../repos/bedrock-vlm-inspection-lab python src/token_check.py \
      --image <VisA の画像> --models jp.amazon.nova-2-lite-v1:0 ... --out results/token_check.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROMPT = "この画像に写っている物を、日本語で一言で答えてください。"
MAX_TOKENS = 50
LONG_EDGE = 768

# 単価（USD / 100万トークン、ap-northeast-1 から呼んだとき、オンデマンドの standard）。
# 出典は results/pricing_used.md（AWS Price List API、2026-11-16 確認）。
PRICES: dict[str, tuple[float, float]] = {
    "jp.amazon.nova-2-lite-v1:0": (0.396, 3.311),
    "jp.anthropic.claude-haiku-4-5-20251001-v1:0": (1.10, 5.50),
    "qwen.qwen3-vl-235b-a22b": (0.64, 3.22),
    # 2026-11-16 追加：このアカウントで契約が AVAILABLE で、単価が分かる残りのモデル（results/cheatsheet.csv の値）
    "amazon.nova-lite-v1:0": (0.072, 0.288),
    "apac.amazon.nova-pro-v1:0": (0.96, 3.84),
    "jp.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.3, 16.5),
    "jp.anthropic.claude-sonnet-4-6": (3.3, 16.5),
    "google.gemma-3-4b-it": (0.05, 0.10),
    "google.gemma-3-12b-it": (0.11, 0.35),
    "google.gemma-3-27b-it": (0.28, 0.46),
    "mistral.magistral-small-2509": (0.61, 1.82),
    "mistral.ministral-3-3b-instruct": (0.12, 0.12),
    "mistral.ministral-3-8b-instruct": (0.18, 0.18),
    "mistral.ministral-3-14b-instruct": (0.24, 0.24),
    "mistral.mistral-large-3-675b-instruct": (0.61, 1.82),
    "moonshotai.kimi-k2.5": (0.72, 3.6),
    "global.moonshotai.kimi-k3": (3.0, 15.0),
    "nvidia.nemotron-nano-12b-v2": (0.24, 0.73),
    "global.xai.grok-4.6": (2.0, 6.0),
    "global.xai.grok-4.7": (2.0, 6.0),
}


NO_TEMPERATURE = {"global.moonshotai.kimi-k3"}


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


def cost_usd(usage: Usage, input_per_mtok: float, output_per_mtok: float) -> float:
    """usage と単価（USD / 100万トークン）から費用を出す。負のトークン数は受け付けない。"""
    if usage.input_tokens < 0 or usage.output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if input_per_mtok < 0 or output_per_mtok < 0:
        raise ValueError("prices must be non-negative")
    return (usage.input_tokens * input_per_mtok + usage.output_tokens * output_per_mtok) / 1_000_000


def resized_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """縦横比を保って、長辺を long_edge にした大きさ。元が小さければそのまま。"""
    if width <= 0 or height <= 0:
        raise ValueError("size must be positive")
    if max(width, height) <= long_edge:
        return width, height
    scale = long_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def claude_visual_tokens(width: int, height: int) -> int:
    """Anthropic の Vision のドキュメントの式 ⌈w/28⌉×⌈h/28⌉（縮小の前の値）。比較用。"""
    return math.ceil(width / 28) * math.ceil(height / 28)


def load_variants(image_path: Path) -> dict[str, tuple[bytes | None, int, int]]:
    from PIL import Image

    raw = image_path.read_bytes()
    with Image.open(image_path) as im:
        w, h = im.size
        rw, rh = resized_size(w, h, LONG_EDGE)
        buf = io.BytesIO()
        im.convert("RGB").resize((rw, rh), Image.LANCZOS).save(buf, format="JPEG", quality=95)
    return {"none": (None, 0, 0), "original": (raw, w, h), f"long{LONG_EDGE}": (buf.getvalue(), rw, rh)}


def call(client, model_id: str, image: bytes | None) -> tuple[Usage | None, str, float, str]:
    content: list[dict] = []
    if image is not None:
        content.append({"image": {"format": "jpeg", "source": {"bytes": image}}})
    content.append({"text": PROMPT})
    t0 = time.perf_counter()
    try:
        r = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
            # temperature を受け付けないモデル（2026-11-16: Kimi K3 が ValidationException）には送らない
            inferenceConfig=({"maxTokens": MAX_TOKENS} if model_id in NO_TEMPERATURE
                             else {"maxTokens": MAX_TOKENS, "temperature": 0}),
        )
    except Exception as e:  # エラーも結果として残す
        return None, "", time.perf_counter() - t0, f"{type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    u = r.get("usage", {})
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    return Usage(int(u["inputTokens"]), int(u["outputTokens"])), text, dt, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--region", default="ap-northeast-1")
    ap.add_argument("--budget-usd", type=float, default=0.05)
    args = ap.parse_args(argv)

    unknown = [m for m in args.models if m not in PRICES]
    if unknown:
        print(f"単価が分からないモデルは呼びません: {unknown}", file=sys.stderr)
        return 2

    variants = load_variants(args.image)
    # 見込み（上限）: 画像あたり最大 5,000 トークン、出力は MAX_TOKENS として計算
    est = sum(cost_usd(Usage(5_000, MAX_TOKENS), *PRICES[m]) * len(variants) for m in args.models)
    print(f"費用の見込み（上限）: {est:.4f} USD")
    if est > args.budget_usd:
        print("見込みが予算を超えるので止めます", file=sys.stderr)
        return 3

    import boto3

    client = boto3.client("bedrock-runtime", region_name=args.region)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["called_at_utc", "region", "model_id", "variant", "image", "width", "height",
                         "input_tokens", "output_tokens", "cost_usd", "latency_s", "output_text", "error"])
        for m in args.models:
            for name, (img, w, h) in variants.items():
                usage, text, dt, err = call(client, m, img)
                c = cost_usd(usage, *PRICES[m]) if usage else ""
                wr.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), args.region, m, name,
                             args.image.as_posix().split("data/visa/")[-1] if img is not None else "",
                             w or "", h or "", usage.input_tokens if usage else "",
                             usage.output_tokens if usage else "", c, f"{dt:.2f}", text.replace("\n", " "), err])
                print(m, name, usage, c, err)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
