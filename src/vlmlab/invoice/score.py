"""応答の読み取りと採点（CI-IS1〜IS7）。正解は採点だけに使い、整合チェック（truth.consistent）には渡さない。"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .schema import SCHEMA
from .truth import FIELDS, MONEY_FIELDS, consistent

_WS = re.compile(r"[\s　]+")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FENCE = re.compile(r"^```(?:json|JSON)?[ \t]*\n(.*)\n```$", re.DOTALL)
STR_FIELDS = ["invoice_number", "issue_date", "issuer_name", "registration_number", "recipient_name"]
CHECKABLE = ["registration_number", *MONEY_FIELDS]  # 整合チェックで見られる項目


def parse_invoice(raw: str | None) -> dict[str, Any]:
    """応答の本文を読む（CI-IS1、IS2）。

    - strict: 本文（前後の空白を除く）全体が JSON のオブジェクト
    - fenced: 本文全体が ```json … ``` の囲み1つで、中身が JSON のオブジェクト（strict で読めたものは含めない）
    - それ以外は失敗。文章の中から {…} を探し出すことはしない
    返り値: {"parsed": dict|None, "how": "strict"|"fenced"|None, "error": str|None, "schema_errors": [...]}
    """
    if raw is None:
        return {"parsed": None, "how": None, "error": "no text", "schema_errors": []}
    s = raw.strip()
    for how, body in (("strict", s), ("fenced", None)):
        if how == "fenced":
            m = _FENCE.match(s)
            if not m:
                break
            body = m.group(1).strip()
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            return {"parsed": None, "how": None, "error": "json: not an object", "schema_errors": []}
        return {"parsed": obj, "how": how, "error": None, "schema_errors": schema_errors(obj)}
    return {"parsed": None, "how": None, "error": "json: decode error", "schema_errors": []}


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def schema_errors(obj: dict[str, Any]) -> list[str]:
    """SCHEMA に合わない点の一覧（CI-IS2）。空ならスキーマどおり。"""
    errs = []
    props = SCHEMA["properties"]
    for k in SCHEMA["required"]:
        if k not in obj:
            errs.append(f"missing:{k}")
    for k in obj:
        if k not in props:
            errs.append(f"extra:{k}")
    for k in ["invoice_number", "issuer_name", "recipient_name", "issue_date"]:
        if k in obj and not isinstance(obj[k], str):
            errs.append(f"type:{k}")
    if "issue_date" in obj and isinstance(obj["issue_date"], str) and not _DATE.match(obj["issue_date"]):
        errs.append("format:issue_date")
    rn = obj.get("registration_number", None)
    if "registration_number" in obj and rn is not None and not (isinstance(rn, str) and re.fullmatch(r"T[0-9]{13}", rn)):
        errs.append("pattern:registration_number")
    for k in MONEY_FIELDS:
        if k in obj and not _is_int(obj[k]):
            errs.append(f"type:{k}")
    items = obj.get("line_items")
    if "line_items" in obj:
        if not isinstance(items, list):
            errs.append("type:line_items")
        else:
            for i, it in enumerate(items):
                if not isinstance(it, dict) or set(it) != set(SCHEMA["properties"]["line_items"]["items"]["required"]):
                    errs.append(f"item{i}:keys")
                    continue
                if not isinstance(it["description"], str) or not isinstance(it["reduced_rate"], bool) or not all(
                        _is_int(it[k]) for k in ("quantity", "unit_price", "amount")):
                    errs.append(f"item{i}:type")
    return errs


def norm_str(v: Any) -> Any:
    """文字列は空白（半角・全角・改行）を除く。それ以外はそのまま（CI-IS3）。"""
    return _WS.sub("", v) if isinstance(v, str) else v


def field_correct(field: str, pred: Any, truth: Any) -> bool:
    """1項目の正誤（CI-IS3）。文字列は空白を除いて完全一致、金額は整数（bool を除く）で完全一致。"""
    if field in MONEY_FIELDS:
        return _is_int(pred) and pred == truth
    if not isinstance(pred, str):
        return False
    return norm_str(pred) == norm_str(truth)


def _item_key(it: Any) -> tuple | None:
    if not isinstance(it, dict):
        return None
    try:
        q, u, a, r = it["quantity"], it["unit_price"], it["amount"], it["reduced_rate"]
        d = it["description"]
    except KeyError:
        return None
    if not (isinstance(d, str) and _is_int(q) and _is_int(u) and _is_int(a) and isinstance(r, bool)):
        return None
    return (norm_str(d), q, u, a, r)


def line_item_matches(pred_items: Any, truth_items: list[dict[str, Any]]) -> int:
    """正解の明細行のうち、全列が一致する行が応答にある数（並びは問わず、応答の1行は1回だけ使う。CI-IS4）。"""
    if not isinstance(pred_items, list):
        return 0
    avail = Counter(k for k in (_item_key(it) for it in pred_items) if k is not None)
    n = 0
    for t in truth_items:
        k = _item_key(t)
        if avail[k] > 0:
            avail[k] -= 1
            n += 1
    return n


def score(parsed: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    """1件の採点（CI-IS5）。読めなかった応答は全項目を不正解、明細の一致を 0 とする。

    整合チェック（consistent）は読み取った値だけで計算する（CI-IS6）。
    """
    p = parsed or {}
    row: dict[str, Any] = {f"ok_{f}": field_correct(f, p.get(f), truth[f]) for f in FIELDS}
    row["all_fields"] = all(row[f"ok_{f}"] for f in FIELDS)
    row["checkable_all"] = all(row[f"ok_{f}"] for f in CHECKABLE)
    row["n_items_truth"] = len(truth["line_items"])
    row["n_items_matched"] = line_item_matches(p.get("line_items"), truth["line_items"])
    row["n_items_pred"] = len(p["line_items"]) if isinstance(p.get("line_items"), list) else 0
    c = consistent(parsed)
    row.update({f"cons_{k}": v for k, v in c.items()})
    row["flagged"] = not c["all_ok"]
    return row


def values_for_agreement(parsed: dict[str, Any] | None) -> tuple | None:
    """一致率の比較に使う10項目の値（文字列は空白を除く）。読めなければ None（CI-IS7）。"""
    if parsed is None:
        return None
    return tuple(norm_str(parsed.get(f)) for f in FIELDS)
