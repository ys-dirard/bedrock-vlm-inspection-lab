"""早見表（results/cheatsheet.csv / cheatsheet.md）を、保存した API の出力と、公式のページから写した値から作る。

入力（すべて results/ の中）:
  raw/foundation_models_image_apne1.json   list-foundation-models（東京、画像入力のモデルだけ）
  raw/inference_profiles_image_apne1.json  list-inference-profiles（東京、ARN は削除済み）
  raw/model_availability_apne1.json        get-foundation-model-availability（このアカウント固有）
  raw/pricelist_apne1_ondemand.csv         Price List API（東京、OnDemand の行）
  doc_facts.csv                            公式のページから写した値（画像のトークン・上限、URL と確認日）
  token_check.csv                          実際に呼んだ結果（あれば「このアカウントで呼べたか」に使う）

使い方: python src/build_cheatsheet.py（記事フォルダで）
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
CHECKED_ON = "2026-11-16"

# モデル ID → 価格表の行（usageType）。(価格表のモデル名, In-Region/Geo の入力, 同 出力, Global の入力, Global の出力)
# 単位は Price List の Unit に従い、100万トークンあたりに直す。None は「東京の価格表に行が無い」。
FM = "APN1-MP:APN1_"
PRICE_MAP: dict[str, tuple[str, str | None, str | None, str | None, str | None]] = {
    "amazon.nova-2-lite-v1:0": ("Nova 2.0 Lite", "APN1-Nova2.0Lite-input-tokens", "APN1-Nova2.0Lite-output-tokens",
                                "APN1-Nova2.0Lite-input-tokens-cross-region-global", "APN1-Nova2.0Lite-output-tokens-cross-region-global"),
    "amazon.nova-pro-v1:0": ("Nova Pro", "APN1-NovaPro-input-tokens", "APN1-NovaPro-output-tokens", None, None),
    "amazon.nova-lite-v1:0": ("Nova Lite", "APN1-NovaLite-input-tokens", "APN1-NovaLite-output-tokens", None, None),
    "anthropic.claude-haiku-4-5-20251001-v1:0": ("Claude Haiku 4.5", FM + "InputTokenCount-Units", FM + "OutputTokenCount-Units",
                                                 FM + "InputTokenCount_Global-Units", FM + "OutputTokenCount_Global-Units"),
    "anthropic.claude-sonnet-4-5-20250929-v1:0": ("Claude Sonnet 4.5", FM + "InputTokenCount-Units", FM + "OutputTokenCount-Units",
                                                  FM + "InputTokenCount_Global-Units", FM + "OutputTokenCount_Global-Units"),
    "anthropic.claude-sonnet-4-6": ("Claude Sonnet 4.6", FM + "InputTokenCount-Units", FM + "OutputTokenCount-Units",
                                    FM + "InputTokenCount_Global-Units", FM + "OutputTokenCount_Global-Units"),
    "anthropic.claude-opus-4-5-20251101-v1:0": ("(東京の価格表に無い)", None, None, None, None),
    "anthropic.claude-opus-4-6-v1": ("Claude Opus 4.6", None, None,
                                     FM + "InputTokenCount_Global-Units", FM + "OutputTokenCount_Global-Units"),
    "anthropic.claude-sonnet-5-5": ("(東京の価格表に無い)", None, None, None, None),
    "openai.gpt-5.4": ("(東京の価格表に無い)", None, None, None, None),
    "openai.gpt-5.5": ("(東京の価格表に無い)", None, None, None, None),
    "moonshotai.kimi-k3.1": ("(東京の価格表に無い)", None, None, None, None),
    "cohere.embed-v4:0": ("Embed v4", FM + "InputTokenCount-Units", "N/A", None, "N/A"),  # 埋め込みなので出力の単価はない。Global の行は東京の価格表に無い
    "google.gemma-3-4b-it": ("Gemma 3 4B", "APN1-Gemma-3-4B-IT-input-tokens", "APN1-Gemma-3-4B-IT-output-tokens", None, None),
    "google.gemma-3-12b-it": ("Gemma 3 12B", "APN1-Gemma-3-12B-IT-input-tokens", "APN1-Gemma-3-12B-IT-output-tokens", None, None),
    "google.gemma-3-27b-it": ("Gemma 3 27B", "APN1-Gemma-3-27B-IT-input-tokens", "APN1-Gemma-3-27B-IT-output-tokens", None, None),
    "mistral.mistral-large-3-675b-instruct": ("Mistral Large 3", "APN1-Mistral-Large-3-675b-Instruct-input-tokens", "APN1-Mistral-Large-3-675b-Instruct-output-tokens", None, None),
    "mistral.ministral-3-14b-instruct": ("Ministral 14B 3.0", "APN1-Ministral-3-14b-Instruct-input-tokens", "APN1-Ministral-3-14b-Instruct-output-tokens", None, None),
    "mistral.ministral-3-8b-instruct": ("Ministral 8B 3.0", "APN1-Ministral-3-8b-Instruct-input-tokens", "APN1-Ministral-3-8b-Instruct-output-tokens", None, None),
    "mistral.ministral-3-3b-instruct": ("Ministral 3B 3.0", "APN1-Ministral-3-3b-Instruct-input-tokens", "APN1-Ministral-3-3b-Instruct-output-tokens", None, None),
    "mistral.magistral-small-2509": ("Magistral Small 1.2", "APN1-Magistral-Small-2509-input-tokens", "APN1-Magistral-Small-2509-output-tokens", None, None),
    "qwen.qwen3-vl-235b-a22b": ("Qwen3 VL 235B A22B", "APN1-Qwen3-VL-235B-A22B-input-tokens", "APN1-Qwen3-VL-235B-A22B-output-tokens", None, None),
    "moonshotai.kimi-k2.5": ("Kimi K2.5", "APN1-moonshotai.kimi-k2.5-input-tokens", "APN1-moonshotai.kimi-k2.5-output-tokens", None, None),
    "moonshotai.kimi-k3": ("Kimi K3", "APN1-moonshotai.kimi-k3-mantle-input-tokens-standard", "APN1-moonshotai.kimi-k3-mantle-output-tokens-standard",
                           "APN1-moonshotai.kimi-k3-mantle-input-tokens-global-standard", "APN1-moonshotai.kimi-k3-mantle-output-tokens-global-standard"),
    "nvidia.nemotron-nano-12b-v2": ("NVIDIA Nemotron Nano 2 VL", "APN1-Nemotron-Nano-12B-V2-VL-BF16-input-tokens", "APN1-Nemotron-Nano-12B-V2-VL-BF16-output-tokens", None, None),
    "xai.grok-4.6": ("xai.grok-4.6", "APN1-xai.grok-4.6-mantle-input-tokens-standard", "APN1-xai.grok-4.6-mantle-output-tokens-standard",
                     "APN1-xai.grok-4.6-mantle-input-tokens-global-standard", "APN1-xai.grok-4.6-mantle-output-tokens-global-standard"),
    "xai.grok-4.7": ("Grok 4.7", "APN1-xai.grok-4.7-mantle-input-tokens-standard", "APN1-xai.grok-4.7-mantle-output-tokens-standard",
                     "APN1-xai.grok-4.7-mantle-input-tokens-global-standard", "APN1-xai.grok-4.7-mantle-output-tokens-global-standard"),
}
# 新しい課金体系（input_tokens_standard / input_tokens_global_standard）の Marketplace のモデル
for mid, name in [
    ("anthropic.claude-opus-4-7", "Claude Opus 4.7"), ("anthropic.claude-opus-4-8", "Claude Opus 4.8"),
    ("anthropic.claude-opus-5", "Claude Opus 5"), ("anthropic.claude-opus-5-5", "Claude Opus 5.5"),
    ("anthropic.claude-sonnet-5", "Claude Sonnet 5"), ("anthropic.claude-haiku-5-5", "Claude Haiku 5.5"),
    ("anthropic.claude-fable-5", "Claude Fable 5"), ("anthropic.claude-fable-5-1", "Claude Fable 5.1"),
    ("anthropic.claude-fable-5-2", "Claude Fable 5.2"),
    ("openai.gpt-5.6-sol", "OpenAI GPT-5.6 Sol"), ("openai.gpt-5.6-terra", "OpenAI GPT-5.6 Terra"),
    ("openai.gpt-5.6-luna", "OpenAI GPT-5.6 Luna"), ("openai.gpt-6-sol", "OpenAI GPT-6 Sol"),
    ("openai.gpt-6-luna", "OpenAI GPT-6 Luna"), ("openai.gpt-6-astra", "OpenAI GPT-6 Astra"),
    ("openai.gpt-6.1-sol", "OpenAI GPT-6.1 Sol"), ("openai.gpt-6.1-terra", "OpenAI GPT-6.1 Terra"),
]:
    PRICE_MAP[mid] = (f"{name}", FM + "input_tokens_standard-Units", FM + "output_tokens_standard-Units",
                      FM + "input_tokens_global_standard-Units", FM + "output_tokens_global_standard-Units")


def per_mtok(row: dict) -> float:
    p = float(row["pricePerUnit"])
    unit = row["unit"].lower()
    if unit == "1k tokens":
        return round(p * 1000, 6)
    if unit == "1m tokens":
        return round(p, 6)
    raise ValueError(f"unknown unit: {row['unit']}")


def lookup(prices: list[dict], model: str, usage: str | None) -> float | None:
    if usage is None or usage == "N/A":
        return None
    hits = [r for r in prices if r["model"] == model and r["usageType"] == usage]
    if len(hits) != 1:
        return None
    return per_mtok(hits[0])


def main() -> None:
    fm = json.loads((RES / "raw/foundation_models_image_apne1.json").read_text(encoding="utf-8"))["models"]
    ips = json.loads((RES / "raw/inference_profiles_image_apne1.json").read_text(encoding="utf-8"))["profiles"]
    av = {a["modelId"]: a for a in json.loads((RES / "raw/model_availability_apne1.json").read_text(encoding="utf-8"))["results"]}
    with (RES / "raw/pricelist_apne1_ondemand.csv").open(encoding="utf-8") as f:
        prices = list(csv.DictReader(f))
    with (RES / "doc_facts.csv").open(encoding="utf-8-sig") as f:
        facts = {r["family"]: r for r in csv.DictReader(f)}
    called: dict[str, str] = {}
    tc = RES / "token_check.csv"
    if tc.exists():
        with tc.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                base = r["model_id"].split(".", 1)[1] if r["model_id"].split(".")[0] in ("jp", "global", "apac", "us", "eu") else r["model_id"]
                if not r["error"]:
                    called[base] = f"呼べた（{r['model_id']}）"  # 一度でも成功したら「呼べた」
                elif not called.get(base, "").startswith("呼べた"):
                    called[base] = f"エラー: {r['error'].split(':')[0]}（{r['model_id']}）"

    rows = []
    for m in sorted(fm, key=lambda x: (x["providerName"], x["modelId"])):
        mid = m["modelId"]
        profs = [p for p in ips if p["inferenceProfileId"].split(".", 1)[1] == mid]
        geo = [p["inferenceProfileId"] for p in profs if not p["inferenceProfileId"].startswith("global.")]
        glob = [p["inferenceProfileId"] for p in profs if p["inferenceProfileId"].startswith("global.")]
        in_region = "ON_DEMAND" in (m["inferenceTypesSupported"] or [])
        ways = []
        if in_region:
            ways.append("In-Region")
        for g in geo:
            ways.append(f"Geo {g.split('.')[0].upper()}（{g}）")
        for g in glob:
            ways.append(f"Global（{g}）")
        pm = PRICE_MAP.get(mid)
        if pm is None:
            raise SystemExit(f"PRICE_MAP にないモデル: {mid}")
        name, iu, ou, giu, gou = pm
        fam = facts_key(mid)
        fact = facts.get(fam, {})
        rows.append({
            "provider": m["providerName"],
            "model_name": m["modelName"],
            "model_id": mid,
            "output": "/".join(m["outputModalities"]),
            "lifecycle": m["lifecycle"],
            "tokyo_access": " / ".join(ways) if ways else "不明",
            # 東京からその呼び方ができないときは、価格表に行があっても単価を出さない（"-"）
            "price_in_region_or_geo_in": fmt(lookup(prices, name, iu), iu) if (in_region or geo) else "-",
            "price_in_region_or_geo_out": fmt(lookup(prices, name, ou), ou) if (in_region or geo) else "-",
            "price_global_in": fmt(lookup(prices, name, giu), giu) if glob else "-",
            "price_global_out": fmt(lookup(prices, name, gou), gou) if glob else "-",
            "price_source": f"AWS Price List API ap-northeast-1（{name}）",
            "image_token_rule": fact.get("image_token_rule", "【要確認】"),
            "image_limits": fact.get("image_limits", "【要確認】"),
            "apis": fact.get("apis", "【要確認】"),
            "doc_urls": fact.get("doc_urls", ""),
            "doc_checked_on": fact.get("checked_on", ""),
            "account_agreement": av.get(mid, {}).get("agreementAvailability", "不明"),
            "account_called": called.get(mid, "未実施"),
            "access_checked_on": CHECKED_ON,
        })
    with (RES / "cheatsheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_md(rows, facts)
    print(f"{len(rows)} models")


def fmt(v: float | None, usage: str | None = None) -> str:
    if usage == "N/A":
        return "-（埋め込み）"
    return "不明（東京の価格表に行なし）" if v is None else f"{v:g}"


def facts_key(mid: str) -> str:
    if mid in ("anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic.claude-sonnet-4-5-20250929-v1:0",
               "anthropic.claude-sonnet-4-6", "anthropic.claude-opus-4-5-20251101-v1:0", "anthropic.claude-opus-4-6-v1"):
        return "anthropic-standard"
    if mid.startswith("anthropic."):
        return "anthropic-high"
    if mid == "amazon.nova-2-lite-v1:0":
        return "nova2"
    if mid in ("amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0"):
        return "nova1"
    if mid == "moonshotai.kimi-k2.5":
        return "moonshot-k2.5"
    if mid.startswith("moonshotai."):
        return "moonshot-k3"
    # 残りは提供元ごと（mistral / google / openai / xai / nvidia / qwen / cohere）
    return mid.split(".")[0]


def write_md(rows: list[dict], facts: dict) -> None:
    out = [f"# 東京（ap-northeast-1）から呼べる画像入力のモデル（確認日 {CHECKED_ON}）", "",
           "- 「東京から呼べる方法」と単価は、AWS の API（list-foundation-models / list-inference-profiles）と Price List API の値（どのアカウントでも同じ公式の情報）",
           "- 「このアカウント」の2列は、この検証のアカウント固有の状態（Marketplace の契約など）。記事では一般化しない",
           "- 単価は USD / 100万トークン、オンデマンドの Standard。In-Region と Geo（jp. / apac.）は東京の単価で同じ行",
           "- 画像のトークン・上限は doc_facts.csv（公式のページ、URL と確認日つき）。モデルの系列ごとに同じ値", "",
           "| 提供元 | モデル | モデル ID | 東京から呼べる方法 | 入力（In-Region/Geo） | 出力（In-Region/Geo） | 入力（Global） | 出力（Global） | 画像のトークン | このアカウントの契約 | このアカウントで呼んだか |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append("| " + " | ".join([r["provider"], r["model_name"], f"`{r['model_id']}`", r["tokyo_access"],
                                      r["price_in_region_or_geo_in"], r["price_in_region_or_geo_out"],
                                      r["price_global_in"], r["price_global_out"], r["image_token_rule"].split("。")[0],
                                      r["account_agreement"], r["account_called"]]) + " |")
    out += ["", "## 系列ごとの画像の扱い（公式のページから）", "",
            "| 系列 | 画像のトークンの数え方 | 1リクエストの画像の上限 | API | 出典（確認日） |", "| --- | --- | --- | --- | --- |"]
    for k, f in facts.items():
        out.append(f"| {f['label']} | {f['image_token_rule']} | {f['image_limits']} | {f['apis']} | {f['doc_urls']}（{f['checked_on']}） |")
    (RES / "cheatsheet.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
