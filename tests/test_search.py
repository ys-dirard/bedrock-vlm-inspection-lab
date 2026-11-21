"""埋め込みと手元の検索（vlmlab.search）の単体テスト。AWS は呼ばない。"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmlab.embed import EmbedResult
from vlmlab.search.index import (Hit, chance_hit_at_k, chance_precision, hit_at_k, precision_at_k,
                                 reciprocal_rank, top_k)
from vlmlab.search.nova import EmbedInput, build_body, parse_body
from vlmlab.search.store import EmbedPrices, EmbedRunConfig, embed_cost, execute_embed, load_vectors

PRICES = EmbedPrices(per_image_usd=0.00006, per_document_image_usd=0.0006, per_mtok_text_usd=0.135)


# ---- 依頼の本文（CI-S1）----

def test_build_body_text_and_image():
    """CI-S1"""
    b = build_body(EmbedInput("text", text="左上に細い線キズ"), "IMAGE_RETRIEVAL", 1024)
    p = b["singleEmbeddingParams"]
    assert b["taskType"] == "SINGLE_EMBEDDING"
    assert p["text"] == {"truncationMode": "NONE", "value": "左上に細い線キズ"}
    assert p["embeddingPurpose"] == "IMAGE_RETRIEVAL" and p["embeddingDimension"] == 1024
    b = build_body(EmbedInput("image", data=b"\x89PNG", format="png", detail="DOCUMENT_IMAGE"), "GENERIC_INDEX", 256)
    img = b["singleEmbeddingParams"]["image"]
    assert img["detailLevel"] == "DOCUMENT_IMAGE" and base64.b64decode(img["source"]["bytes"]) == b"\x89PNG"


@pytest.mark.parametrize("inp,purpose,dim", [
    (EmbedInput("text", text="a"), "SEARCH", 1024),                     # purpose が仕様の外
    (EmbedInput("text", text="a"), "GENERIC_INDEX", 512),               # 次元が仕様の外
    (EmbedInput("text", text="  "), "GENERIC_INDEX", 1024),             # 空の文
    (EmbedInput("text", text="あ" * 8193), "GENERIC_INDEX", 1024),      # 8,192 文字を超える
    (EmbedInput("image", data=b"x", format="png", detail=None), "GENERIC_INDEX", 1024),  # detail が無い
    (EmbedInput("image", data=b"x", format="bmp", detail="STANDARD_IMAGE"), "GENERIC_INDEX", 1024),
    (EmbedInput("audio"), "GENERIC_INDEX", 1024),
])
def test_build_body_rejects(inp, purpose, dim):
    """CI-S1"""
    with pytest.raises(ValueError):
        build_body(inp, purpose, dim)


# ---- 応答の読み取り（CI-S2）----

def _body(vec, typ="TEXT", **extra):
    return {"embeddings": [{"embeddingType": typ, "embedding": vec, **extra}]}


def test_parse_body_ok():
    """CI-S2"""
    assert parse_body(_body([0.5, -1, 2]), 3, "TEXT") == [0.5, -1.0, 2.0]


@pytest.mark.parametrize("body", [
    {}, {"embeddings": []}, {"embeddings": [{"embeddingType": "TEXT", "embedding": [1, 2, 3]}] * 2},
    _body([1.0, 2.0, 3.0], typ="IMAGE"),               # 種類が違う
    _body([1.0, 2.0]),                                  # 長さが違う
    _body([1.0, float("inf"), 3.0]),                    # 有限でない
    _body([1.0, False, 3.0]),                           # bool
    _body([0, 0, 0]),                                   # すべて 0
    _body([1.0, 2.0, 3.0], truncatedCharLength=100),    # 文が切られた
])
def test_parse_body_rejects(body):
    """CI-S2"""
    with pytest.raises(ValueError):
        parse_body(body, 3, "TEXT")


# ---- 費用（CI-S3）----

def test_embed_cost():
    """CI-S3"""
    assert embed_cost(PRICES, "image", "STANDARD_IMAGE", True, 0) == 0.00006
    assert embed_cost(PRICES, "image", "DOCUMENT_IMAGE", True, None) == 0.0006
    assert embed_cost(PRICES, "text", None, True, 1000) == pytest.approx(1000 * 0.135 / 1e6)
    assert embed_cost(PRICES, "text", None, True, None) is None        # トークン数が無ければ費用不明
    assert embed_cost(PRICES, "image", "STANDARD_IMAGE", False, None) == 0.0  # 応答が無ければ 0
    assert embed_cost(PRICES, "text", None, False, None) == 0.0


def test_prices_require_refuses_null():
    """CI-S3"""
    p = EmbedPrices(0.00006, None, 0.135)
    p.require({("image", "STANDARD_IMAGE"), ("text", None)})
    with pytest.raises(RuntimeError):
        p.require({("image", "DOCUMENT_IMAGE")})


# ---- 埋め込みの実行・再開（CI-S4、CI-S5）----

class FakeEmbedder:
    def __init__(self, dim=4, fail=(), tokens=7):
        self.dim, self.fail, self.tokens, self.calls = dim, set(fail), tokens, []

    def embed(self, inp):
        key = inp.text if inp.modality == "text" else len(inp.data)
        self.calls.append(key)
        if key in self.fail:
            return EmbedResult(None, None, None, 6, "ThrottlingException: x")
        rng = np.random.default_rng(abs(hash(key)) % 2**32)
        tok = self.tokens if inp.modality == "text" else 0
        return EmbedResult(list(rng.normal(size=self.dim) + 0.1), tok, 0.01, 1, None, ["x-amzn-bedrock-input-token-count"])


def _png(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "imgs"
    _png(root / "a.png", (255, 0, 0))
    _png(root / "b.png", (0, 255, 0))
    m = pd.DataFrame([
        {"item_id": "i1", "modality": "image", "detail": "STANDARD_IMAGE", "path": "a.png", "text": ""},
        {"item_id": "i2", "modality": "image", "detail": "DOCUMENT_IMAGE", "path": "b.png", "text": ""},
        {"item_id": "t1", "modality": "text", "detail": "", "path": "", "text": "細い線キズ"},
        {"item_id": "t2", "modality": "text", "detail": "", "path": "", "text": "欠け"},
    ])
    (tmp_path / "data").mkdir()
    m.to_csv(tmp_path / "data" / "m.csv", index=False)
    cfg = EmbedRunConfig(name="t", manifest="data/m.csv", model_id="m", region="us-east-1",
                         purpose="GENERIC_INDEX", dimension=4, budget_usd=1.0)
    return tmp_path, root, cfg


def test_execute_embed_records_costs_and_resumes(lab):
    """CI-S3, CI-S4"""
    base, root, cfg = lab
    s = execute_embed(cfg, FakeEmbedder(fail={"欠け"}), PRICES, root, base, base / "runs", log=lambda _: None)
    assert s["embedded"] == 3 and s["api_errors"] == 1
    assert s["spent_usd"] == pytest.approx(0.00006 + 0.0006 + 7 * 0.135 / 1e6)
    fake = FakeEmbedder()
    s = execute_embed(cfg, fake, PRICES, root, base, base / "runs", log=lambda _: None)
    assert fake.calls == ["欠け"] and s["embedded"] == 4      # 取れた3件は呼ばない
    ids, vec = load_vectors(base / "runs" / "t" / "embeddings.jsonl", ["t2", "i1"])
    assert ids == ["t2", "i1"] and vec.shape == (2, 4)


def test_execute_embed_trial_uses_head_and_other_name(lab):
    """CI-S4"""
    base, root, cfg = lab
    s = execute_embed(cfg, FakeEmbedder(), PRICES, root, base, base / "runs", limit=2, log=lambda _: None)
    assert s["run_name"] == "t__trial2" and s["planned"] == 2


def test_execute_embed_refuses_changed_config_or_content(lab):
    """CI-S5"""
    base, root, cfg = lab
    execute_embed(cfg, FakeEmbedder(), PRICES, root, base, base / "runs", log=lambda _: None)
    cfg2 = EmbedRunConfig(**{**cfg.__dict__, "purpose": "TEXT_RETRIEVAL"})
    with pytest.raises(RuntimeError, match="config differs"):
        execute_embed(cfg2, FakeEmbedder(), PRICES, root, base, base / "runs", log=lambda _: None)
    _png(root / "a.png", (0, 0, 255))  # 画像の中身が変わった
    with pytest.raises(RuntimeError, match="changed"):
        execute_embed(cfg, FakeEmbedder(), PRICES, root, base, base / "runs", log=lambda _: None)


def test_execute_embed_stops_when_text_cost_unknown(lab):
    """CI-S3"""
    base, root, cfg = lab
    fake = FakeEmbedder(tokens=None)
    s = execute_embed(cfg, fake, PRICES, root, base, base / "runs", log=lambda _: None)
    assert s["stopped"].startswith("cost unknown") and len(fake.calls) == 3  # 最初の文で止まる


def test_execute_embed_budget(lab):
    """CI-S4"""
    base, root, cfg = lab
    cfg.budget_usd = 0.0001
    s = execute_embed(cfg, FakeEmbedder(), PRICES, root, base, base / "runs", log=lambda _: None)
    assert s["stopped"] and s["spent_usd"] <= 0.0001 + 0.0006


# ---- 検索（CI-S6）と指標（CI-S7）----

def test_top_k_order_ties_and_exclude():
    """CI-S6"""
    idx = np.array([[1, 0], [2, 0], [0, 1], [1, 1]], dtype=float)  # a と b は同じ向き（同点）
    hits = top_k(np.array([[1, 0.0]]), idx, ["a", "b", "c", "d"], k=3)[0]
    assert [h.item_id for h in hits] == ["a", "b", "d"]
    assert hits[0].score == pytest.approx(1.0) and hits[2].score == pytest.approx(np.sqrt(0.5))
    hits = top_k(np.array([[1, 0.0]]), idx, ["a", "b", "c", "d"], k=3, exclude=[{"a"}])[0]
    assert [h.item_id for h in hits] == ["b", "d", "c"] and [h.rank for h in hits] == [1, 2, 3]
    assert len(top_k(np.array([[1, 0.0]]), idx, list("abcd"), k=10)[0]) == 4


@pytest.mark.parametrize("q,x,ids", [
    (np.array([[1, 0.0]]), np.array([[0, 0.0]]), ["a"]),           # ゼロのベクトル
    (np.array([[1, 0.0]]), np.array([[1, 0, 0.0]]), ["a"]),        # 次元が違う
    (np.array([[1, 0.0]]), np.array([[1, 0.0], [0, 1.0]]), ["a", "a"]),  # ID が重複
])
def test_top_k_rejects(q, x, ids):
    """CI-S6"""
    with pytest.raises(ValueError):
        top_k(q, x, ids, 1)


def test_metrics():
    """CI-S7"""
    hits = [Hit("x", 1, .9), Hit("r1", 2, .8), Hit("y", 3, .7), Hit("r2", 4, .6)]
    rel = {"r1", "r2", "r3"}
    assert hit_at_k(hits, rel, 1) == 0 and hit_at_k(hits, rel, 2) == 1
    assert precision_at_k(hits, rel, 4) == 0.5 and precision_at_k(hits, rel, 5) == 0.4  # 分母は k
    assert reciprocal_rank(hits, rel) == 0.5 and reciprocal_rank(hits[:1], rel) == 0.0
    assert chance_precision(3, 300) == 0.01
    assert chance_hit_at_k(1, 10, 5) == pytest.approx(0.5)
    assert chance_hit_at_k(0, 10, 5) == 0.0 and chance_hit_at_k(10, 10, 5) == 1.0
