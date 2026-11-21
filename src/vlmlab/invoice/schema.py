"""抽出の JSON スキーマとプロンプト（CI-IP1〜IP3）。

構造化出力の制約（https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html、2026-11-21 確認）:
すべての object に additionalProperties: false。minimum・minLength などは使えない。pattern は使える。
null を許す項目は registration_number だけ（省略できる項目は 0）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROMPT_VERSION = "inv_v1"

ITEM: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "description": {"type": "string"},
        "quantity": {"type": "integer"},
        "unit_price": {"type": "integer"},
        "amount": {"type": "integer"},
        "reduced_rate": {"type": "boolean"},
    },
    "required": ["description", "quantity", "unit_price", "amount", "reduced_rate"],
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "invoice_number": {"type": "string"},
        "issue_date": {"type": "string", "format": "date"},
        "issuer_name": {"type": "string"},
        "registration_number": {"type": ["string", "null"], "pattern": "^T[0-9]{13}$"},
        "recipient_name": {"type": "string"},
        "subtotal_10": {"type": "integer"},
        "tax_10": {"type": "integer"},
        "subtotal_8": {"type": "integer"},
        "tax_8": {"type": "integer"},
        "total": {"type": "integer"},
        "line_items": {"type": "array", "items": ITEM},
    },
    "required": [
        "invoice_number", "issue_date", "issuer_name", "registration_number", "recipient_name",
        "subtotal_10", "tax_10", "subtotal_8", "tax_8", "total", "line_items",
    ],
}

# 両方の条件で同じ指示。構造化出力なしの条件だけ、末尾にスキーマを付ける
PROMPT = """この請求書の画像から項目を読み取り、JSON で返してください。
- invoice_number: 請求番号
- issue_date: 発行日。和暦は西暦に直し、YYYY-MM-DD で書く
- issuer_name: 請求書の発行元の名称（宛先ではない）
- registration_number: 適格請求書発行事業者の登録番号（T と13桁の数字）
- recipient_name: 宛先の名称（「御中」は含めない）
- subtotal_10, tax_10: 税率10%の対象の小計（税抜）と消費税額
- subtotal_8, tax_8: 税率8%（軽減税率）の対象の小計（税抜）と消費税額。対象が無ければ 0
- total: 合計（税込）
- line_items: 明細の各行。description は品名（行に付いた「※」は含めない）、quantity は数量、unit_price は単価、amount は金額、reduced_rate は軽減税率（8%）の対象なら true
金額は円単位の整数にしてください（「¥」やカンマは付けない）。読めない項目は推測しないでください。登録番号が読めなければ null にしてください。"""

SCHEMA_SUFFIX = "\n次の JSON スキーマに従い、JSON だけを出力してください。\n"

SPLIT_NOTE = "\n画像は1枚の請求書を上下に分けたもので、1枚目が上半分、2枚目が下半分です（境目は少し重なっています）。"


def build_text(structured: bool, split_halves: bool) -> str:
    """送る指示の文（CI-IP2）。"""
    text = PROMPT + (SPLIT_NOTE if split_halves else "")
    if not structured:
        text += SCHEMA_SUFFIX + json.dumps(SCHEMA, ensure_ascii=False)
    return text


def output_config() -> dict[str, Any]:
    """Converse の outputConfig（CI-IP3）。schema は JSON の文字列で渡す。"""
    return {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {
        "schema": json.dumps(SCHEMA), "name": "invoice", "description": "日本の適格請求書の項目"}}}}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
