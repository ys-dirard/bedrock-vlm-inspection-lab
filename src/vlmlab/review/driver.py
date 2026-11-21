"""ローカルから動かす部分（画像のアップロードと実行の開始、記録の書き出し、確認画面の署名プロキシ）。

Lambda には載せない（pandas などを使う）。AWS を呼ぶのは start・export・proxy だけ。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .judge_handler import image_format
from .runlog import COLUMNS, build_rows, summarize

SFN_TRANSITION_USD_APN1 = 0.000025  # AWS Price List API（AmazonStates, APN1-StateTransition, 2026-11-21 確認）


def make_review_split(test_200: pd.DataFrame, n_per_group: int = 5, seed: int = 42) -> pd.DataFrame:
    """CI-HR15：基本のテスト200枚から、品目×正解ごとに n 枚（random_state=seed）。順番は image_id 順。"""
    parts = [g.sample(n=n_per_group, random_state=seed) for _, g in test_200.groupby(["category", "label"], sort=True)]
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def content_type(key: str) -> str:
    """CI-HR17：S3 に上げる画像の Content-Type（確認画面の <img> で画像として扱われるように）。"""
    return "image/" + image_format(key)


def execution_name(tag: str, i: int, image_id: str) -> str:
    """CI-HR16：実行名（80文字以内、英数字と - _ だけ）。同じ tag・番号なら同じ名前（二重に始めない）。"""
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", image_id)
    name = f"{tag}-{i:03d}-{slug}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError(f"invalid execution name {name!r}")
    return name


def plan_start(split: pd.DataFrame, refs: pd.DataFrame, tag: str, n_refs: int = 1) -> list[dict[str, Any]]:
    """CI-HR16：画像ごとの実行の入力。正解（label）は入れない。見本は品目ごとに ref_rank の小さい順に n_refs 枚。"""
    refs = refs.sort_values(["category", "ref_rank"])
    by_cat = refs[refs.ref_rank <= n_refs].groupby("category")["path"].apply(list).to_dict()
    out = []
    for i, row in enumerate(split.itertuples(index=False), start=1):
        ref_paths = by_cat.get(row.category, [])
        if len(ref_paths) < n_refs:
            raise ValueError(f"not enough refs for {row.category}")
        out.append({
            "name": execution_name(tag, i, row.image_id),
            "image_path": row.path, "ref_paths": ref_paths,
            "input": {"image_id": row.image_id, "key": f"images/{row.path}",
                      "ref_keys": [f"refs/{p}" for p in ref_paths]},
        })
    return out


def start(plan: list[dict[str, Any]], visa_root: Path, bucket: str, state_machine_arn: str, region: str) -> list[dict]:
    import boto3

    s3 = boto3.client("s3", region_name=region)
    sfn = boto3.client("stepfunctions", region_name=region)
    started = []
    for p in plan:
        for rel, key in [(p["image_path"], p["input"]["key"]), *zip(p["ref_paths"], p["input"]["ref_keys"])]:
            s3.upload_file(str(visa_root / rel), bucket, key, ExtraArgs={"ContentType": content_type(key)})
        resp = sfn.start_execution(stateMachineArn=state_machine_arn, name=p["name"],
                                   input=json.dumps({**p["input"], "bucket": bucket}))
        started.append({"name": p["name"], "image_id": p["input"]["image_id"], "execution_arn": resp["executionArn"],
                        "start_date": resp["startDate"].isoformat()})
    return started


def scan_all(table: str, region: str) -> list[dict[str, Any]]:
    import boto3

    ddb = boto3.client("dynamodb", region_name=region)
    items, kwargs = [], {"TableName": table, "ConsistentRead": True}
    while True:
        page = ddb.scan(**kwargs)
        items += page.get("Items", [])
        if "LastEvaluatedKey" not in page:
            return items
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def write_run_log(items: list[dict[str, Any]], split: pd.DataFrame, price: Any, out_csv: Path,
                  run_tag: str | None = None) -> dict[str, Any]:
    """CI-HR12・HR13：run_tag があれば、その tag で始まる review_id だけを書く。"""
    if run_tag:
        items = [i for i in items if (i.get("review_id", {}).get("S") or "").startswith(f"{run_tag}-")]
    labels = {r.image_id: {"category": r.category, "label": r.label} for r in split.itertuples(index=False)}
    rows = build_rows(items, labels, price)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, SFN_TRANSITION_USD_APN1)
    out_csv.with_name(out_csv.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def serve_proxy(function_url: str, region: str, port: int = 8765) -> None:
    """確認画面の署名プロキシ。http://127.0.0.1:PORT/ へのリクエストに SigV4 の署名を付けて関数 URL に送る。

    関数 URL を AWS_IAM のままブラウザで使うためのもの。127.0.0.1 だけで待ち受ける。
    """
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    base = function_url.rstrip("/")
    creds = boto3.Session(region_name=region).get_credentials()

    class H(BaseHTTPRequestHandler):
        def _forward(self, method: str) -> None:
            body = self.rfile.read(int(self.headers.get("content-length") or 0)) if method == "POST" else None
            headers = {"content-type": self.headers.get("content-type", "")} if body else {}
            req = AWSRequest(method=method, url=base + self.path, data=body, headers=headers)
            SigV4Auth(creds.get_frozen_credentials(), "lambda", region).add_auth(req)

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *a, **k):
                    return None

            opener = urllib.request.build_opener(NoRedirect)
            try:
                r = opener.open(urllib.request.Request(req.url, data=body, headers=dict(req.headers), method=method))
                status, rh, data = r.status, r.headers, r.read()
            except urllib.error.HTTPError as e:
                status, rh, data = e.code, e.headers, e.read()
            self.send_response(status)
            for k in ("content-type", "location", "cache-control"):
                if rh.get(k):
                    self.send_header(k, rh[k])
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._forward("GET")

        def do_POST(self):  # noqa: N802
            self._forward("POST")

    print(f"http://127.0.0.1:{port}/ -> {base}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
