"""確認画面の Lambda（関数 URL）。人が OK / NG を押すと、待っている実行に SendTaskSuccess を返す。

- GET  /        確認待ち（status = pending）と、時間切れ（timed_out、要エスカレーション）の一覧
- POST /decide  フォーム（review_id, decision）。CI-HR8〜HR10

タスクトークンはブラウザに渡さない（DynamoDB にだけ置き、review_id で引く）。
関数 URL の認証は AWS_IAM を想定（ブラウザからは、署名をするローカルのプロキシを通す。docs/human_review.md）。

環境変数: TABLE_NAME、PRESIGN_SECONDS（既定 900）
"""

from __future__ import annotations

import base64
import html
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import parse_qs

DECISIONS = ("OK", "NG")
# SendTaskSuccess が「もう待っていない」ことを示すエラー
EXPIRED_ERRORS = ("TaskTimedOut", "TaskDoesNotExist", "InvalidToken")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _resp(status: int, body: str, content_type: str = "text/plain; charset=utf-8",
          headers: dict[str, str] | None = None) -> dict[str, Any]:
    h = {"content-type": content_type, "cache-control": "no-store"}
    h.update(headers or {})
    return {"statusCode": status, "headers": h, "body": body}


def parse_form(event: dict[str, Any]) -> dict[str, str]:
    """CI-HR8（pre）：application/x-www-form-urlencoded の本文を読む。同じキーが2つ以上あれば ValueError。"""
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    q = parse_qs(body, keep_blank_values=True)
    out: dict[str, str] = {}
    for k, v in q.items():
        if len(v) != 1:
            raise ValueError(f"duplicate form field {k!r}")
        out[k] = v[0]
    return out


def _attr(item: dict[str, Any], name: str) -> str | None:
    v = item.get(name)
    return None if v is None else v.get("S")


class ReviewApp:
    def __init__(self, ddb: Any, sfn: Any, s3: Any, table: str, presign_seconds: int = 900,
                 now: Callable[[], str] = _now):
        self.ddb, self.sfn, self.s3 = ddb, sfn, s3
        self.table, self.presign_seconds, self.now = table, presign_seconds, now

    # ---- POST /decide ----
    def decide(self, review_id: str, decision: str) -> tuple[int, str]:
        """CI-HR8〜HR10。返り値は (HTTP の状態, 説明)。"""
        from botocore.exceptions import ClientError

        if decision not in DECISIONS:
            return 400, f"decision must be one of {DECISIONS}"
        if not review_id:
            return 400, "review_id is required"
        key = {"review_id": {"S": review_id}}
        item = self.ddb.get_item(TableName=self.table, Key=key, ConsistentRead=True).get("Item")
        if not item:
            return 404, "no such review"
        if _attr(item, "status") != "pending" or not _attr(item, "task_token"):
            return 409, f"not pending (status={_attr(item, 'status')})"
        submitted_at = self.now()
        try:  # 二重送信を防ぐ（pending のときだけ submitting にできる）
            self.ddb.update_item(
                TableName=self.table, Key=key,
                UpdateExpression="SET #s = :sub, human_decision = :d, submitted_at = :t",
                ConditionExpression="#s = :pending",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":sub": {"S": "submitting"}, ":pending": {"S": "pending"},
                                           ":d": {"S": decision}, ":t": {"S": submitted_at}},
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return 409, "already submitted"
            raise
        output = json.dumps({"decision": decision, "submitted_at": submitted_at})
        try:
            self.sfn.send_task_success(taskToken=_attr(item, "task_token"), output=output)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in EXPIRED_ERRORS:
                raise
            try:  # 時間切れの記録（timed_out）が先に書かれていれば、そちらを残す
                self.ddb.update_item(
                    TableName=self.table, Key=key,
                    UpdateExpression="SET #s = :exp, expired_error = :c",
                    ConditionExpression="#s = :sub",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={":exp": {"S": "expired"}, ":sub": {"S": "submitting"},
                                               ":c": {"S": code}},
                )
            except ClientError as e2:
                if e2.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
                    raise
            return 410, f"the workflow is no longer waiting ({code})"
        return 303, "submitted"

    # ---- GET / ----
    def scan_status(self, status: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self.table, "FilterExpression": "#s = :s",
            "ExpressionAttributeNames": {"#s": "status"}, "ExpressionAttributeValues": {":s": {"S": status}},
        }
        while True:
            page = self.ddb.scan(**kwargs)
            items.extend(page.get("Items", []))
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        return sorted(items, key=lambda i: _attr(i, "requested_at") or "")

    def _img(self, item: dict[str, Any]) -> str:
        url = self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": _attr(item, "s3_bucket"), "Key": _attr(item, "s3_key")},
            ExpiresIn=self.presign_seconds)
        return f'<img src="{html.escape(url)}" alt="" style="max-width:480px">'

    def page(self) -> str:
        e = html.escape
        rows = []
        for it in self.scan_status("pending"):
            rid = _attr(it, "review_id") or ""
            vlm = json.loads(_attr(it, "vlm_result") or "{}")
            buttons = "".join(
                f'<form method="post" action="/decide"><input type="hidden" name="review_id" value="{e(rid)}">'
                f'<button name="decision" value="{d}">{d}</button></form>' for d in DECISIONS)
            rows.append(
                f"<tr><td>{e(_attr(it, 'image_id') or '')}<br><small>{e(rid)}</small></td><td>{self._img(it)}</td>"
                f"<td>VLM: {e(str(vlm.get('judgement')))} / 確信度 {e(str(vlm.get('confidence')))}<br>"
                f"欠陥: {e(str(vlm.get('defect')))}<br>理由: {e(str(vlm.get('route_reason')))}</td>"
                f"<td>{buttons}</td></tr>")
        timed_out = [e(_attr(i, "image_id") or "") for i in self.scan_status("timed_out")]
        return (
            "<!doctype html><html lang=ja><meta charset=utf-8><title>確認待ち</title>"
            f"<h1>確認待ち {len(rows)} 件</h1><table border=1 cellpadding=6>{''.join(rows)}</table>"
            f"<h2>時間切れ（要エスカレーション） {len(timed_out)} 件</h2><p>{'<br>'.join(timed_out)}</p></html>"
        )

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        http = event.get("requestContext", {}).get("http", {})
        method, path = http.get("method", "GET"), event.get("rawPath", "/")
        if method == "GET" and path == "/":
            return _resp(200, self.page(), "text/html; charset=utf-8")
        if method == "POST" and path == "/decide":
            try:
                form = parse_form(event)
            except ValueError as err:
                return _resp(400, str(err))
            status, msg = self.decide(form.get("review_id", ""), form.get("decision", ""))
            if status == 303:
                return _resp(303, msg, headers={"location": "/"})
            return _resp(status, msg)
        return _resp(404, "not found")


_app: ReviewApp | None = None


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _app
    if _app is None:
        import boto3

        _app = ReviewApp(boto3.client("dynamodb"), boto3.client("stepfunctions"), boto3.client("s3"),
                         table=os.environ["TABLE_NAME"],
                         presign_seconds=int(os.environ.get("PRESIGN_SECONDS", "900")))
    return _app(event)
