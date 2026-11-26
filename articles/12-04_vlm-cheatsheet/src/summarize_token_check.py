"""results/token_check.csv から、画像1枚あたりのトークン（画像あり − 画像なし）と費用の合計を出す。

出力: results/token_check_summary.csv、results/cost_total.md
使い方: python src/summarize_token_check.py（記事フォルダで）
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def claude_resized(width: int, height: int, max_edge: int = 1568, max_tokens: int = 1568) -> tuple[int, int]:
    """Anthropic の Vision のドキュメントの参照実装（vision-coordinates）を写したもの。"""

    def tok(w: int, h: int) -> int:
        return math.ceil(w / 28) * math.ceil(h / 28)

    def fits(w: int, h: int) -> bool:
        return math.ceil(w / 28) * 28 <= max_edge and math.ceil(h / 28) * 28 <= max_edge and tok(w, h) <= max_tokens

    if fits(width, height):
        return width, height
    if height > width:
        h2, w2 = claude_resized(height, width, max_edge, max_tokens)
        return w2, h2
    ar = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / ar), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / ar), 1)


def doc_expected(model_id: str, w: int, h: int) -> tuple[str, str]:
    """公式の文書から計算できる画像のトークン数と、その根拠。計算できなければ ("", 理由)。"""
    base = model_id.split(".", 1)[1] if model_id.split(".")[0] in ("jp", "global", "apac", "us", "eu") else model_id
    if base == "amazon.nova-2-lite-v1:0":
        return "230", "Nova 2 のユーザーガイド：1枚230トークンで課金 [AWS]"
    if base in ("amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"):
        return "", "Nova（Version 1）のユーザーガイドは目安の表だけ（900×900 で約1,300、1400×900 で約1,800 など）[AWS]"
    if base.startswith("anthropic.claude-haiku-4-5") or base.startswith("anthropic.claude-sonnet-4-"):
        rw, rh = claude_resized(w, h)
        t = math.ceil(rw / 28) * math.ceil(rh / 28)
        return str(t), f"Anthropic の Vision（Standard 階層）：{w}x{h} → {rw}x{rh}、⌈{rw}/28⌉×⌈{rh}/28⌉ [提供元]"
    if base.startswith("google.gemma-3"):
        return "256", "Gemma 3 のモデルカード：896×896 に正規化して1枚256トークン [提供元]"
    if base in ("mistral.mistral-large-3-675b-instruct",) or base.startswith("mistral.ministral-3"):
        return f"{w * h / 784:.0f}", f"Mistral の Vision の FAQ：約 (幅×高さ)/784 = {w}×{h}/784（近似）[提供元]"
    if base == "mistral.magistral-small-2509":
        return "", "Mistral の FAQ の表に Magistral Small はない"
    return "", "公式の文書に式なし"


def main() -> None:
    with (RES / "token_check.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if not r["error"]]
    base = {}
    for r in ok:
        if r["variant"] == "none":
            base[r["model_id"]] = int(r["input_tokens"])  # 後の行で上書き（再実行したときは最後の成功）
    out = []
    for r in ok:
        if r["variant"] == "none" or r["model_id"] not in base:
            continue
        w, h = int(r["width"]), int(r["height"])
        it = int(r["input_tokens"])
        d = it - base[r["model_id"]]
        e, basis = doc_expected(r["model_id"], w, h)
        out.append({"model_id": r["model_id"], "variant": r["variant"], "width": w, "height": h,
                    "input_tokens_with_image": it, "input_tokens_text_only": base[r["model_id"]],
                    "image_tokens_measured": d, "image_tokens_by_doc": e,
                    "diff_measured_minus_doc": (d - int(e)) if e else "", "doc_basis": basis})
    failed = {}
    for r in rows:
        if r["error"]:
            failed.setdefault(r["model_id"], set()).add(r["error"].split(":")[0])
    succeeded = {r["model_id"] for r in ok}
    for m, errs in failed.items():
        if m not in succeeded:
            out.append({"model_id": m, "variant": "(失敗)", "width": "", "height": "", "input_tokens_with_image": "",
                        "input_tokens_text_only": "", "image_tokens_measured": "", "image_tokens_by_doc": "",
                        "diff_measured_minus_doc": "", "doc_basis": "エラー: " + " / ".join(sorted(errs))})
    with (RES / "token_check_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)

    total = sum(float(r["cost_usd"]) for r in rows if r["cost_usd"])
    by_model: dict[str, float] = {}
    for r in rows:
        if r["cost_usd"]:
            by_model[r["model_id"]] = by_model.get(r["model_id"], 0.0) + float(r["cost_usd"])
    lines = ["# 実測の費用の合計（12/4、token_check.csv から計算）", "",
             "- 単価は `token_check.py` の PRICES（出典は `pricing_used.md` と `cheatsheet.csv`）× usage の入出力トークン",
             "- 失敗した呼び出し（エラー）は usage が無いので 0 として数えた",
             "- Cost Explorer での照合はまだ【要確認】", "",
             f"呼び出し {len(rows)} 回（成功 {len(ok)}、失敗 {len(rows) - len(ok)}）、合計 **{total:.4f} USD**", "",
             "| モデル | 費用（USD） |", "| --- | --- |"]
    for m, c in sorted(by_model.items(), key=lambda x: -x[1]):
        lines.append(f"| `{m}` | {c:.6f} |")
    (RES / "cost_total.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(out)} rows, total {total:.4f} USD")


if __name__ == "__main__":
    main()
