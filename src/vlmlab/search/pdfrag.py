"""12/24 の記事：図表の多い PDF を、テキストで探すか（A）、ページ画像のまま探すか（B）。

  python -m vlmlab.search.pdfrag fetch --pdf-dir DIR
      令和8年版 情報通信白書（総務省、CC BY 4.0）の第2部第1章の PDF（12 ファイル、84 ページ）を取得する
  python -m vlmlab.search.pdfrag pages --pdf-dir DIR [--out data/whitepaper] [--dpi 150]
      ページを PNG にし、テキストを抜き出して、埋め込みの一覧（画像・テキスト）を作る
  python -m vlmlab.search.pdfrag questions-template
      質問の記録の形（30 行、中身は空）を作る。質問と正解のページは書き手が書く
  python -m vlmlab.search.pdfrag queries [--questions data/splits/ac24_questions.csv]
      質問から、テキストの埋め込みの一覧を作る（A は TEXT_RETRIEVAL、B は DOCUMENT_RETRIEVAL で埋め込む）
  python -m vlmlab.search.pdfrag retrieve --pages RUN --texts RUN --query-a RUN --query-b RUN --out DIR [--k 3]
      上位 k ページに正解のページが入るかを数え、回答に渡すページを retrieved.csv に残す
  python -m vlmlab.search.pdfrag answer CONFIG --retrieved CSV --data-root DIR [--limit N]
      検索したページを Claude Haiku 4.5 に渡して答えさせる（Converse、jp. のプロファイル）

PDF と PNG はコミットしない（data/whitepaper/ は .gitignore）。ページの一覧（ac24_pages.csv）にテキストの SHA-256 を残す。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from ..bedrock import ModelRequest, VisionModel
from ..cost import BudgetExceeded, BudgetGuard, PricingTable, call_cost
from ..data import load_image
from ..run import JST, fill_template, load_jsonl
from .index import hit_at_k, top_k
from .store import load_vectors

BASE_URL = "https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r08/pdf/"
# 令和8年版 情報通信白書 第2部 第1章「ICT市場の動向」第1〜12節（2026-11-21 に一覧のページで確認）
FILES = ["n2110000", "n2120000", "n2130000", "n2140000", "n2150000", "n2160000",
         "n2170000", "n2180000", "n2190000", "n21a0000", "n21b0000", "n21c0000"]
LICENSE_URL = BASE_URL + "r08riyou.pdf"


def fetch(pdf_dir: Path, files: list[str] = FILES, log: Callable[[str], None] = print) -> pd.DataFrame:
    """PDF を取得する（既にあれば取得しない）。URL・SHA-256・大きさを返す。"""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        dest = pdf_dir / f"{f}.pdf"
        url = BASE_URL + f"{f}.pdf"
        if not dest.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError(f"{url} did not return a PDF")
            dest.write_bytes(data)
            log(f"fetched {url}")
        b = dest.read_bytes()
        rows.append({"file": f, "url": url, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    return pd.DataFrame(rows)


def normalize_text(t: str) -> str:
    """抜き出したテキストの空白をまとめる（改行・タブ・連続した空白を1つの空白に）。中身は変えない。"""
    return re.sub(r"\s+", " ", t.replace("\x00", "")).strip()


def render_pages(pdf_dir: Path, out_dir: Path, files: list[str] = FILES, dpi: int = 150
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ページを PNG にし、テキストを抜き出す（CI-S13）。(ページの一覧, 画像の一覧, テキストの一覧)。"""
    import pypdfium2 as pdfium

    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    pages = []
    for f in files:
        pdf = pdfium.PdfDocument(str(pdf_dir / f"{f}.pdf"))
        for i in range(len(pdf)):
            page_id = f"{f}_p{i + 1:02d}"
            page = pdf[i]
            text = normalize_text(page.get_textpage().get_text_bounded())
            png = out_dir / "pages" / f"{page_id}.png"
            if not png.exists():
                page.render(scale=dpi / 72).to_pil().convert("RGB").save(png)
            pages.append({"page_id": page_id, "file": f, "page_no": i + 1, "n_chars": len(text),
                          "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                          "png": f"pages/{page_id}.png", "text": text})
    df = pd.DataFrame(pages)
    images = pd.DataFrame({"item_id": df.page_id, "modality": "image", "detail": "DOCUMENT_IMAGE", "path": df.png})
    texts = df[df.n_chars > 0]
    texts = pd.DataFrame({"item_id": texts.page_id, "modality": "text", "text": texts.text})
    return df.drop(columns=["text", "png"]), images, texts


def questions_template(n: int = 30) -> pd.DataFrame:
    """質問の記録の形（CI-S14）。answer_pages は正解のページの page_id を「;」でつなぐ。
    evidence は答えのある場所（figure / table / text）。"""
    return pd.DataFrame({"question_id": [f"q{i + 1:02d}" for i in range(n)], "question": "", "answer": "",
                         "answer_pages": "", "evidence": "", "note": ""})


def read_questions(path: Path, page_ids: set[str]) -> pd.DataFrame:
    q = pd.read_csv(path, keep_default_na=False, dtype=str)
    need = {"question_id", "question", "answer", "answer_pages", "evidence"}
    if not need <= set(q.columns):
        raise ValueError(f"questions lack {sorted(need - set(q.columns))}")
    q = q[q.question.str.strip() != ""].copy()
    if q.question_id.duplicated().any():
        raise ValueError("question_id is not unique")
    q["answer_page_set"] = q.answer_pages.map(lambda s: {p.strip() for p in s.split(";") if p.strip()})
    for r in q.itertuples():
        if not r.answer_page_set:
            raise ValueError(f"{r.question_id} has no answer_pages")
        unknown = r.answer_page_set - page_ids
        if unknown:
            raise ValueError(f"{r.question_id} refers to unknown pages {sorted(unknown)}")
        if r.evidence not in ("figure", "table", "text"):
            raise ValueError(f"{r.question_id}: evidence must be figure/table/text")
    return q


def retrieve(questions: pd.DataFrame, page_ids: list[str], emb: dict[str, Path], k: int = 3
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A（質問 TEXT_RETRIEVAL → ページのテキスト）と B（質問 DOCUMENT_RETRIEVAL → ページ画像）の上位 k（CI-S15）。

    テキストが抜き出せなかったページは A の候補に無いまま数え、候補の数を表に書く。
    """
    rows, per_q = [], []
    for method, (q_key, i_key) in {"A_text": ("query_a", "texts"), "B_image": ("query_b", "pages")}.items():
        _, qv = load_vectors(emb[q_key], list(questions.question_id))
        have = {r["item_id"] for r in load_jsonl(emb[i_key]) if r.get("vector") is not None}
        ids = [p for p in page_ids if p in have]
        _, iv = load_vectors(emb[i_key], ids)
        for q, hits in zip(questions.itertuples(index=False), top_k(qv, iv, ids, k)):
            rel = q.answer_page_set
            rows.append({"method": method, "question_id": q.question_id,
                         "pages": ";".join(h.item_id for h in hits), "scores": ";".join(f"{h.score:.4f}" for h in hits)})
            per_q.append({"method": method, "question_id": q.question_id, "evidence": q.evidence,
                          "n_index": len(ids), f"hit_at_{k}": hit_at_k(hits, rel, k), "hit_at_1": hit_at_k(hits, rel, 1),
                          f"recall_at_{k}": len(rel & {h.item_id for h in hits[:k]}) / len(rel)})
    pq = pd.DataFrame(per_q)
    metrics = ["hit_at_1", f"hit_at_{k}", f"recall_at_{k}"]
    overall = pq.groupby("method", sort=False)[metrics].mean().reset_index().assign(evidence="(all)")
    by_ev = pq.groupby(["method", "evidence"], sort=False)[metrics].mean().reset_index()
    n = pd.concat([pq.groupby("method", sort=False).size().reset_index(name="n").assign(evidence="(all)"),
                   pq.groupby(["method", "evidence"], sort=False).size().reset_index(name="n")])
    summ = pd.concat([overall, by_ev], ignore_index=True).merge(n, on=["method", "evidence"])
    summ["n_index"] = summ.method.map(pq.groupby("method").n_index.first())
    return pd.DataFrame(rows), summ


@dataclass
class AnswerConfig:
    name: str
    model_id: str
    region: str
    prompt: str
    budget_usd: float
    answer_input: str = "image"  # image（ページ画像を渡す）/ text（ページのテキストを渡す）
    image_long_edge: int | None = 1568
    inference: dict[str, Any] = field(default_factory=lambda: {"temperature": 0, "maxTokens": 400})

    @classmethod
    def load(cls, path: str | Path) -> "AnswerConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.answer_input not in ("image", "text"):
            raise ValueError("answer_input must be image or text")
        return cfg


def build_answer_request(cfg: AnswerConfig, prompt_text: str, question: str, pages: list[str], data_root: Path,
                         texts: dict[str, str]) -> ModelRequest:
    """回答の依頼（CI-S16）。ページは検索の順に渡す。text のときは「[page_id] 本文」を順に並べる。"""
    if cfg.answer_input == "image":
        images = [load_image(data_root / "pages" / f"{p}.png", cfg.image_long_edge) for p in pages]
        body = fill_template(prompt_text, {"question": question, "context": "（ページの画像を参照）"})
    else:
        images = []
        ctx = "\n\n".join(f"[{p}] {texts.get(p, '')}" for p in pages)
        body = fill_template(prompt_text, {"question": question, "context": ctx})
    return ModelRequest(model_id=cfg.model_id, images=images, text=body,
                        temperature=cfg.inference.get("temperature", 0),
                        max_tokens=cfg.inference.get("maxTokens", 400))


def execute_answers(cfg: AnswerConfig, model: VisionModel, pricing: PricingTable, retrieved: pd.DataFrame,
                    questions: pd.DataFrame, data_root: Path, base_dir: Path, runs_dir: Path,
                    texts: dict[str, str] | None = None, limit: int | None = None,
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    """検索した上位のページで答えさせ、1回1行で answers.jsonl に残す（CI-S16）。再開・予算は CI-C4・C6 と同じ。"""
    price = pricing.get(cfg.model_id, "standard")
    if not price.runnable:
        raise RuntimeError(f"price for {cfg.model_id} is null in pricing.yaml")
    prompt_text = (base_dir / cfg.prompt).read_text(encoding="utf-8")
    run_name = cfg.name if limit is None else f"{cfg.name}__trial{limit}"
    out = runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers.jsonl"
    prev = load_jsonl(path)
    ident = {"model_id": cfg.model_id, "answer_input": cfg.answer_input, "image_long_edge": cfg.image_long_edge,
             "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]}
    for r in prev:
        if any(r.get(k) != v for k, v in ident.items()):
            raise RuntimeError(f"config differs from existing answers in {out}")
    done = {(r["question_id"], r["method"]) for r in prev if r.get("api_error") is None}
    guard = BudgetGuard.resume(cfg.budget_usd, [r.get("cost_usd") for r in prev])
    qtext = questions.set_index("question_id").question
    todo = retrieved if limit is None else retrieved.head(limit)
    run_id = datetime.now(JST).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    stopped, n_called = None, 0
    with path.open("a", encoding="utf-8") as f:
        for r in todo.itertuples(index=False):
            if (r.question_id, r.method) in done:
                continue
            try:
                guard.check_before_call()
            except BudgetExceeded as e:
                stopped = str(e)
                break
            pages = [p for p in r.pages.split(";") if p]
            req = build_answer_request(cfg, prompt_text, qtext[r.question_id], pages, data_root, texts or {})
            called_at = datetime.now(JST).isoformat(timespec="seconds")
            resp = model.invoke(req)
            cost = call_cost(price, resp.input_tokens, resp.output_tokens, resp.cache_read_tokens,
                             resp.cache_write_tokens)
            rec = {"run_name": run_name, "run_id": run_id, "question_id": r.question_id, "method": r.method,
                   "pages": r.pages, **ident, "raw_text": resp.raw_text, "input_tokens": resp.input_tokens,
                   "output_tokens": resp.output_tokens, "latency_s": resp.latency_s,
                   "stop_reason": resp.stop_reason, "cost_usd": cost, "attempts": resp.attempts,
                   "api_error": resp.api_error, "timestamp": called_at}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            guard.add(cost)
            n_called += 1
            if resp.api_error:
                log(f"[api_error] {r.question_id} {r.method}: {resp.api_error}")
    recs = load_jsonl(path)
    summary = {"run_name": run_name, "run_id": run_id, "called": n_called, "stopped": stopped,
               "spent_usd": guard.spent_usd, "api_errors": sum(1 for x in recs if x.get("api_error")),
               "input_tokens_sum": sum(x.get("input_tokens") or 0 for x in recs),
               "output_tokens_sum": sum(x.get("output_tokens") or 0 for x in recs)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m vlmlab.search.pdfrag")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--pdf-dir", required=True)
    f.add_argument("--out", default="data/splits/ac24_sources.csv")
    g = sub.add_parser("pages")
    g.add_argument("--pdf-dir", required=True)
    g.add_argument("--out", default="data/whitepaper")
    g.add_argument("--dpi", type=int, default=150)
    sub.add_parser("questions-template").add_argument("--out", default="data/splits/ac24_questions.csv")
    q = sub.add_parser("queries")
    q.add_argument("--questions", default="data/splits/ac24_questions.csv")
    q.add_argument("--out", default="data/whitepaper/ac24_query_texts.csv")
    r = sub.add_parser("retrieve")
    for k_ in ("pages", "texts", "query-a", "query-b"):
        r.add_argument(f"--{k_}", required=True)
    r.add_argument("--questions", default="data/splits/ac24_questions.csv")
    r.add_argument("--page-list", default="data/splits/ac24_pages.csv")
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=3)
    a_ = sub.add_parser("answer")
    a_.add_argument("config")
    a_.add_argument("--retrieved", required=True)
    a_.add_argument("--questions", default="data/splits/ac24_questions.csv")
    a_.add_argument("--data-root", default="data/whitepaper")
    a_.add_argument("--limit", type=int)
    a_.add_argument("--pricing", default="pricing.yaml")
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        src = fetch(Path(a.pdf_dir))
        src.to_csv(a.out, index=False, lineterminator="\n")
        print(src.to_string(index=False))
    elif a.cmd == "pages":
        out = Path(a.out)
        pages, images, texts = render_pages(Path(a.pdf_dir), out, dpi=a.dpi)
        pages.to_csv("data/splits/ac24_pages.csv", index=False, lineterminator="\n")
        images.to_csv(out / "ac24_image_manifest.csv", index=False, lineterminator="\n")
        texts.to_csv(out / "ac24_text_manifest.csv", index=False, lineterminator="\n")
        print(json.dumps({"pages": len(pages), "with_text": len(texts),
                          "chars_mean": round(pages.n_chars.mean(), 1), "chars_max": int(pages.n_chars.max())}))
    elif a.cmd == "questions-template":
        questions_template().to_csv(a.out, index=False, lineterminator="\n")
    elif a.cmd == "queries":
        page_ids = set(pd.read_csv("data/splits/ac24_pages.csv").page_id)
        qs = read_questions(Path(a.questions), page_ids)
        pd.DataFrame({"item_id": qs.question_id, "modality": "text", "text": qs.question.str.strip()}).to_csv(
            a.out, index=False, lineterminator="\n")
    elif a.cmd == "retrieve":
        page_ids = list(pd.read_csv(a.page_list).page_id)
        qs = read_questions(Path(a.questions), set(page_ids))
        emb = {"pages": a.pages, "texts": a.texts, "query_a": a.query_a, "query_b": a.query_b}
        res, summ = retrieve(qs, page_ids, {k_: Path(v) / "embeddings.jsonl" for k_, v in emb.items()}, a.k)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "retrieved.csv", index=False, lineterminator="\n")
        summ.to_csv(out / "retrieval_summary.csv", index=False, lineterminator="\n", float_format="%.4f")
        print(summ.to_string(index=False))
    else:
        from ..bedrock import BedrockConverse

        cfg = AnswerConfig.load(a.config)
        page_ids = set(pd.read_csv("data/splits/ac24_pages.csv").page_id)
        qs = read_questions(Path(a.questions), page_ids)
        texts = None
        if cfg.answer_input == "text":
            tm = pd.read_csv(Path(a.data_root) / "ac24_text_manifest.csv", keep_default_na=False)
            texts = dict(zip(tm.item_id, tm.text))
        execute_answers(cfg, BedrockConverse(cfg.region), PricingTable.load(a.pricing),
                        pd.read_csv(a.retrieved, keep_default_na=False), qs, Path(a.data_root), Path.cwd(),
                        Path.cwd() / "runs", texts, a.limit)


if __name__ == "__main__":
    main()
