"""12/24 の記事（PDF をテキストで探すか、ページ画像で探すか）の単体テスト。AWS とネットワークは使わない。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmlab.bedrock import ModelResponse
from vlmlab.cost import Price, PricingTable
from vlmlab.search.pdfrag import (AnswerConfig, build_answer_request, execute_answers, normalize_text,
                                  questions_template, read_questions, render_pages, retrieve)

PAGES = ["f_p01", "f_p02", "f_p03", "f_p04"]


def test_normalize_text():
    """CI-S13"""
    assert normalize_text(" 世界の\nICT 市場\t\t 4.77\x00兆ドル \n") == "世界の ICT 市場 4.77兆ドル"


def test_render_pages_image_only_pdf(tmp_path):
    """CI-S13：テキストの無いページは画像の一覧には入り、テキストの一覧には入らない"""
    pdfium = pytest.importorskip("pypdfium2")
    imgs = [Image.new("RGB", (200, 280), c) for c in ("white", "gray")]
    imgs[0].save(tmp_path / "f.pdf", save_all=True, append_images=imgs[1:])
    pages, images, texts = render_pages(tmp_path, tmp_path / "out", files=["f"], dpi=72)
    assert list(pages.page_id) == ["f_p01", "f_p02"] and list(pages.n_chars) == [0, 0]
    assert list(images.detail) == ["DOCUMENT_IMAGE"] * 2 and (tmp_path / "out" / "pages" / "f_p02.png").exists()
    assert len(texts) == 0
    assert pdfium is not None


def _q(rows):
    df = questions_template(len(rows))
    for i, r in enumerate(rows):
        for k, v in r.items():
            df.loc[i, k] = v
    return df


def test_read_questions_validates(tmp_path):
    """CI-S14"""
    p = tmp_path / "q.csv"
    _q([{"question": "市場規模は？", "answer": "4.77兆ドル", "answer_pages": "f_p02; f_p03", "evidence": "figure"},
        {}]).to_csv(p, index=False)
    q = read_questions(p, set(PAGES))
    assert len(q) == 1 and q.iloc[0].answer_page_set == {"f_p02", "f_p03"}   # 空の行は飛ばす
    for bad in [{"answer_pages": "", "evidence": "text"}, {"answer_pages": "x_p09", "evidence": "text"},
                {"answer_pages": "f_p01", "evidence": "chart"}]:
        _q([{"question": "?", "answer": "a", **bad}]).to_csv(p, index=False)
        with pytest.raises(ValueError):
            read_questions(p, set(PAGES))


def _emb(path: Path, vecs):
    path.write_text("\n".join(json.dumps({"item_id": k, "vector": v}) for k, v in vecs.items()), encoding="utf-8")


def test_retrieve_hit_and_recall(tmp_path):
    """CI-S15"""
    q = pd.DataFrame({"question_id": ["q1"], "evidence": ["table"], "answer_page_set": [{"f_p02", "f_p04"}]})
    page_vec = {p: [np.cos(i), np.sin(i)] for i, p in enumerate(PAGES)}
    _emb(tmp_path / "pages.jsonl", page_vec)
    _emb(tmp_path / "texts.jsonl", {k: v for k, v in page_vec.items() if k != "f_p02"})  # テキストが無いページ
    _emb(tmp_path / "qa.jsonl", {"q1": [np.cos(1.9), np.sin(1.9)]})
    _emb(tmp_path / "qb.jsonl", {"q1": [np.cos(1.9), np.sin(1.9)]})
    emb = {"pages": tmp_path / "pages.jsonl", "texts": tmp_path / "texts.jsonl",
           "query_a": tmp_path / "qa.jsonl", "query_b": tmp_path / "qb.jsonl"}
    res, summ = retrieve(q, PAGES, emb, k=2)
    s = summ[summ.evidence == "(all)"].set_index("method")
    assert res.set_index("method").at["B_image", "pages"] == "f_p03;f_p02"
    assert s.at["B_image", "hit_at_1"] == 0 and s.at["B_image", "hit_at_2"] == 1
    assert s.at["B_image", "recall_at_2"] == 0.5 and s.at["A_text", "n_index"] == 3
    # f_p02 は A の候補に無いので、A の上位2件は f_p03・f_p04（f_p04 だけが正解）
    assert res.set_index("method").at["A_text", "pages"] == "f_p03;f_p04"
    assert s.at["A_text", "hit_at_1"] == 0 and s.at["A_text", "recall_at_2"] == 0.5
    assert set(summ.evidence) == {"(all)", "table"}


class FakeModel:
    def __init__(self):
        self.reqs = []

    def invoke(self, req):
        self.reqs.append(req)
        return ModelResponse(raw_text="4.77兆ドル", input_tokens=1000, output_tokens=10, latency_s=0.1,
                             stop_reason="end_turn")


@pytest.fixture
def ans(tmp_path):
    (tmp_path / "data" / "pages").mkdir(parents=True)
    for p in PAGES:
        Image.new("RGB", (40, 60), "white").save(tmp_path / "data" / "pages" / f"{p}.png")
    (tmp_path / "p.txt").write_text("資料：{{context}}\n質問：{{question}}", encoding="utf-8")
    cfg = AnswerConfig(name="a", model_id="m", region="r", prompt="p.txt", budget_usd=1.0)
    pricing = PricingTable([Price("m", "standard", 1.10, 5.50)])
    retrieved = pd.DataFrame({"method": ["A_text", "B_image"], "question_id": ["q1", "q1"],
                              "pages": ["f_p01;f_p02;f_p03", "f_p04;f_p02;f_p01"]})
    qs = pd.DataFrame({"question_id": ["q1"], "question": ["市場規模は？"]})
    return tmp_path, cfg, pricing, retrieved, qs


def test_answer_request_image_and_text(ans):
    """CI-S16"""
    base, cfg, *_ = ans
    req = build_answer_request(cfg, "資料：{{context}} 質問：{{question}}", "Q", ["f_p04", "f_p02"], base / "data", {})
    assert len(req.images) == 2 and "Q" in req.text and req.model_id == "m"
    cfg_t = AnswerConfig(**{**cfg.__dict__, "answer_input": "text"})
    req = build_answer_request(cfg_t, "資料：{{context}} 質問：{{question}}", "Q", ["f_p04", "f_p02"], base / "data",
                               {"f_p04": "四", "f_p02": "二"})
    assert req.images == [] and "[f_p04] 四\n\n[f_p02] 二" in req.text


def test_execute_answers_cost_resume_and_identity(ans):
    """CI-S16"""
    base, cfg, pricing, retrieved, qs = ans
    m = FakeModel()
    s = execute_answers(cfg, m, pricing, retrieved, qs, base / "data", base, base / "runs", log=lambda _: None)
    assert s["called"] == 2 and s["spent_usd"] == pytest.approx(2 * (1000 * 1.10 + 10 * 5.50) / 1e6)
    assert [len(r.images) for r in m.reqs] == [3, 3]
    s = execute_answers(cfg, FakeModel(), pricing, retrieved, qs, base / "data", base, base / "runs",
                        log=lambda _: None)
    assert s["called"] == 0                           # 済んだものは呼ばない
    (base / "p.txt").write_text("別のプロンプト {{context}} {{question}}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="config differs"):
        execute_answers(cfg, FakeModel(), pricing, retrieved, qs, base / "data", base, base / "runs",
                        log=lambda _: None)


def test_execute_answers_budget(ans):
    """CI-S16"""
    base, cfg, pricing, retrieved, qs = ans
    cfg.budget_usd = 0.0015
    s = execute_answers(cfg, FakeModel(), pricing, retrieved, qs, base / "data", base, base / "runs",
                        log=lambda _: None)
    assert s["called"] == 1 and s["stopped"]
