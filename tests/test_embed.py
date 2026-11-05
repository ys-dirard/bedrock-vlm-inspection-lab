"""条件 E（埋め込み＋k近傍法）の単体テスト。AWS は呼ばない。"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmlab.data import EncodedImage
from vlmlab.embed import EmbedResult, build_embed_body, knn_predict, parse_embed_body, tokens_from_headers
from vlmlab.embed_run import EmbedConfig, embed_cost, execute_embed, load_embedding_price
from vlmlab.report import build_report, load_run

PRICE = 0.00006  # 標準の画像1枚あたり（USD）


def _body(vec, typ="IMAGE"):
    return {"embeddings": [{"embeddingType": typ, "embedding": vec}]}


# ---- 応答の読み取り（CI-E1）----

def test_parse_embed_body_ok():
    """CI-E1"""
    assert parse_embed_body(_body([0.1, -0.2, 0.3]), 3) == [0.1, -0.2, 0.3]


@pytest.mark.parametrize("body", [
    {},                                                    # embeddings が無い
    {"embeddings": []},                                    # 空
    {"embeddings": [{"embeddingType": "IMAGE", "embedding": [1, 2, 3]}] * 2},  # 2つ
    _body([1.0, 2.0, 3.0], typ="TEXT"),                    # 種類が違う
    _body([1.0, 2.0]),                                     # 長さが違う
    _body([1.0, float("nan"), 3.0]),                       # 有限でない
    _body([1.0, True, 3.0]),                               # bool
    _body([0.0, 0.0, 0.0]),                                # すべて 0（コサイン距離が決まらない）
])
def test_parse_embed_body_rejects(body):
    """CI-E1：形が違えば ValueError"""
    with pytest.raises(ValueError):
        parse_embed_body(body, 3)


def test_build_embed_body_validates_and_encodes():
    """単価の前提（STANDARD_IMAGE）を依頼に明示する"""
    body = build_embed_body(EncodedImage(b"abc", "jpeg", 1, 1), "GENERIC_INDEX", 1024)
    p = body["singleEmbeddingParams"]
    assert body["taskType"] == "SINGLE_EMBEDDING" and p["embeddingDimension"] == 1024
    assert p["image"] == {"detailLevel": "STANDARD_IMAGE", "format": "jpeg", "source": {"bytes": "YWJj"}}
    with pytest.raises(ValueError):
        build_embed_body(EncodedImage(b"x", "jpeg", 1, 1), "NOPE", 1024)
    with pytest.raises(ValueError):
        build_embed_body(EncodedImage(b"x", "jpeg", 1, 1), "GENERIC_INDEX", 1000)


# ---- トークン数と費用（CI-E2）----

@pytest.mark.parametrize("headers, expected", [
    ({"x-amzn-bedrock-input-token-count": "123"}, 123),
    ({"X-Amzn-Bedrock-Input-Token-Count": "7"}, 7),       # 大文字・小文字は区別しない
    ({}, None),
    (None, None),
    ({"x-amzn-bedrock-input-token-count": "abc"}, None),
    ({"x-amzn-bedrock-input-token-count": "-1"}, None),
])
def test_tokens_from_headers(headers, expected):
    """記録用。分からなければ None"""
    assert tokens_from_headers(headers) == expected


def test_embed_cost():
    """CI-E2：応答が返った呼び出しは画像1枚分、返らなかった呼び出しは 0"""
    assert embed_cost(True, PRICE) == PRICE
    assert embed_cost(False, PRICE) == 0.0


def test_load_embedding_price(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text(
        "embeddings:\n"
        "  - {model_id: e, per_image_usd: 0.00006, per_mtok_text_usd: 0.135}\n"
        "  - {model_id: n, per_image_usd: null}\n"
        "  - {model_id: m, per_image_usd: -1}\n",
        encoding="utf-8")
    assert load_embedding_price(p, "e") == PRICE
    with pytest.raises(RuntimeError):
        load_embedding_price(p, "n")
    with pytest.raises(ValueError):
        load_embedding_price(p, "m")
    with pytest.raises(KeyError):
        load_embedding_price(p, "x")


# ---- k近傍法（CI-E3）----

def test_knn_majority_and_anomaly_fraction():
    """CI-E3：コサイン距離の近い k 件の多数決。p は近傍のうち不良品の割合"""
    train = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1], [0.1, 0.9]])
    labels = ["normal", "normal", "normal", "anomaly", "anomaly"]
    out = knn_predict(train, labels, np.array([[1, 0.05], [0.05, 1], [0.6, 0.4]]), k=3)
    assert out[0] == ("OK", 0.0)
    assert out[1][0] == "NG" and out[1][1] == pytest.approx(2 / 3)
    assert out[2][0] == "OK" and out[2][1] == pytest.approx(0.0)


def test_knn_scale_invariant():
    """CI-E3：コサイン距離なので、ベクトルの長さは判定に効かない"""
    train = np.array([[1, 0], [0, 1], [1, 0.1]])
    labels = ["normal", "anomaly", "normal"]
    a = knn_predict(train, labels, np.array([[0.2, 1.0]]), k=1)
    b = knn_predict(train * 100, labels, np.array([[2.0, 10.0]]), k=1)
    assert a == b == [("NG", 1.0)]


@pytest.mark.parametrize("k, labels", [(2, ["normal"] * 3), (0, ["normal"] * 3), (5, ["normal"] * 3),
                                       (1, ["normal", "bad", "anomaly"])])
def test_knn_rejects_bad_input(k, labels):
    with pytest.raises(ValueError):
        knn_predict(np.eye(3), labels, np.eye(3), k=k)


# ---- 実行（偽物の埋め込みで）----

class FakeEmbedder:
    """画像の色を2次元のベクトルにする。不良品は緑が強い。

    tokens はヘッダーのトークン数の代わり。fail_first 回目までは API のエラー（応答なし）を返す。
    """

    def __init__(self, tokens=0, fail_first=0):
        self.calls, self.tokens, self.fail_first = 0, tokens, fail_first

    def embed(self, img):
        self.calls += 1
        if self.calls <= self.fail_first:
            return EmbedResult(None, None, None, 6, "ThrottlingException: x")
        with Image.open(io.BytesIO(img.data)) as im:
            r, g, _ = im.convert("RGB").getpixel((0, 0))
        return EmbedResult([r / 255 + 0.1, g / 255 + 0.1], self.tokens, 0.2, 1, None,
                           ["x-amzn-bedrock-input-token-count"])


@pytest.fixture
def elab(tmp_path):
    visa = tmp_path / "visa"
    rows = []
    for cat in ("pcb1", "capsules"):
        for label, folder in (("normal", "Normal"), ("anomaly", "Anomaly")):
            d = visa / cat / folder
            d.mkdir(parents=True)
            for i in range(6):
                p = d / f"{i:03d}.png"
                Image.new("RGB", (8, 8), (200, 230 if label == "anomaly" else 10, 0)).save(p, "PNG")
                rows.append({"image_id": f"{cat}/{label}/{i:03d}", "category": cat, "label": label,
                             "split": "x", "path": f"{cat}/{folder}/{i:03d}.png"})
    df = pd.DataFrame(rows)
    idx = df.groupby(["category", "label"]).cumcount()
    (tmp_path / "data").mkdir()
    df[idx < 3].to_csv(tmp_path / "data" / "train.csv", index=False)
    df[idx >= 3].to_csv(tmp_path / "data" / "test.csv", index=False)
    cfg = EmbedConfig(name="e", train_split="data/train.csv", test_split="data/test.csv",
                      model_id="emb", region="us-east-1", budget_usd=1.0, k=3)
    return tmp_path, visa, cfg


def _run(cfg, fake, base, visa, **kw):
    return execute_embed(cfg, fake, PRICE, visa, base, base / "runs", log=lambda m: None, **kw)


def test_embed_run_classifies_and_reports(elab):
    """CI-E3, CI-E5：埋め込み→品目ごとの k近傍法→calls.jsonl を report で集計できる"""
    base, visa, cfg = elab
    s = _run(cfg, FakeEmbedder(), base, visa)
    assert s["embedded"] == 24 and s["classified"] == 12 and s["stopped"] is None
    assert s["index_cost_usd"] == pytest.approx(12 * PRICE) and s["test_cost_usd"] == pytest.approx(12 * PRICE)
    df = load_run(base / "runs" / "e")
    assert len(df) == 12 and (df.pred == df.label.map({"normal": "OK", "anomaly": "NG"})).all()
    paths = build_report([base / "runs" / "e"], base / "out", "t")
    assert paths and all(Path(p).exists() for p in paths.values())
    bm = pd.read_csv(base / "out" / "t_business_metric.csv")
    assert bm["reason"].isna().all()  # 確信度があるので業務指標が出る


def test_embed_run_resumes_without_recalling(elab):
    """CI-E4：取得済みの画像は呼び直さない。埋め込みの設定が違えば止める"""
    base, visa, cfg = elab
    s1 = _run(cfg, FakeEmbedder(fail_first=2), base, visa)
    assert s1["embedded"] == 22 and "classified" not in s1
    fake2 = FakeEmbedder()
    s2 = _run(cfg, fake2, base, visa)
    assert fake2.calls == 2 and s2["classified"] == 12
    cfg.embedding_dimension = 256
    with pytest.raises(RuntimeError):
        _run(cfg, FakeEmbedder(), base, visa)


@pytest.mark.parametrize("tokens", [None, 0, 1234])
def test_embed_run_cost_does_not_depend_on_header_tokens(elab, tokens):
    """CI-E2：費用は画像1枚あたり。ヘッダーのトークン数（画像では 0）は記録するが費用には使わない"""
    base, visa, cfg = elab
    s = _run(cfg, FakeEmbedder(tokens=tokens), base, visa, limit=4)
    assert s["cost_known"] and s["spent_usd"] == pytest.approx(4 * PRICE)
    recs = [json.loads(l) for l in (base / "runs" / "e__trial4" / "embeddings.jsonl").open(encoding="utf-8")]
    assert all(r["input_tokens"] == tokens and r["cost_usd"] == PRICE for r in recs)


def test_embed_run_failed_calls_cost_nothing(elab):
    """CI-E2：応答が返らなかった呼び出しは 0"""
    base, visa, cfg = elab
    s = _run(cfg, FakeEmbedder(fail_first=2), base, visa, limit=4)
    assert s["spent_usd"] == pytest.approx(2 * PRICE)


def test_embed_run_budget_stops(elab):
    """CI-C4 と同じ見張り"""
    base, visa, cfg = elab
    cfg.budget_usd = PRICE * 3.5
    s = _run(cfg, FakeEmbedder(), base, visa)
    assert s["embedded"] == 3 and "budget" in s["stopped"]


def test_train_test_overlap_refused(elab):
    """CI-E5"""
    base, visa, cfg = elab
    cfg.train_split = cfg.test_split
    with pytest.raises(ValueError):
        _run(cfg, FakeEmbedder(), base, visa)
