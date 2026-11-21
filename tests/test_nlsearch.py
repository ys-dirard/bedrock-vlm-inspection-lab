"""12/21 の記事（言葉で不良画像を探す）の単体テスト。AWS は呼ばない。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlmlab.search.nlsearch import (CELLS, captions_manifest, cell_counts, cell_of, evaluate,
                                    make_query_template, queries_manifest)


def test_cell_counts_boundaries():
    """CI-S9"""
    m = np.zeros((9, 9), dtype=np.uint8)
    m[0, 0] = 1      # 左上
    m[3, 3] = 1      # 中央（3*3//9 = 1）
    m[8, 8] = 1      # 右下
    m[2, 3] = 1      # 上（行 0、列 1）
    assert list(cell_counts(m)) == [1, 1, 0, 0, 1, 0, 0, 0, 1]


def test_cell_of_majority_and_ties():
    """CI-S9"""
    m = np.zeros((6, 6), dtype=np.uint8)
    m[0:2, 4:6] = 1   # 右上 4 画素
    m[4, 0] = 1       # 左下 1 画素
    assert cell_of(m) == ("右上", 0.8, 2)
    t = np.zeros((6, 6), dtype=np.uint8)
    t[5, 5] = 1
    t[0, 0] = 1       # 同数なら CELLS の順で先（左上）
    assert cell_of(t)[0] == "左上"
    with pytest.raises(ValueError):
        cell_of(np.zeros((6, 6), dtype=np.uint8))
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    rgb[3, 3, 0] = 255
    assert cell_of(rgb)[0] == "中央"


def _pos():
    rows = []
    for i, (cat, typ, cell) in enumerate([("c", "s", "左"), ("c", "s", "左"), ("c", "s", "左"), ("c", "s", "右"),
                                          ("c", "m", "中央"), ("c", "m", "中央"), ("d", "s", "左"),
                                          ("d", "s", "左"), ("d", "x", "上")]):
        rows.append({"item_id": f"i{i}", "category": cat, "defect_type": typ, "cell": cell})
    return pd.DataFrame(rows)


def test_query_template_rules():
    """CI-S11"""
    t = make_query_template(_pos(), per_category=1, n_total=3)
    assert list(t.query_id) == ["q01", "q02", "q03"]
    first = t[["category", "defect_type", "cell", "n_relevant_strict", "n_relevant_loose"]].values.tolist()
    # c は正解の多い (s, 左)=3、d は (s, 左)=2。足りない1つは残りから多い順 (c, m, 中央)=2
    assert first == [["c", "s", "左", 3, 4], ["c", "m", "中央", 2, 2], ["d", "s", "左", 2, 2]]
    assert (t.query_text == "").all()
    assert not ((t.defect_type == "x") | (t.cell == "右")).any()  # 正解が1枚の組は使わない


def test_captions_manifest_filters(tmp_path):
    """CI-S12"""
    recs = [
        {"image_id": "a", "raw_text": " 中央に点状の汚れ ", "stop_reason": "end_turn", "api_error": None},
        {"image_id": "b", "raw_text": "途中で", "stop_reason": "max_tokens", "api_error": None},
        {"image_id": "c", "raw_text": None, "stop_reason": None, "api_error": "ThrottlingException"},
        {"image_id": "c", "raw_text": "右下に欠け", "stop_reason": "end_turn", "api_error": None},  # 再開で取れた
        {"image_id": "d", "raw_text": "", "stop_reason": "end_turn", "api_error": None},
    ]
    p = tmp_path / "calls.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    m, bad = captions_manifest(p)
    assert m.to_dict("records") == [{"item_id": "a", "modality": "text", "text": "中央に点状の汚れ"},
                                    {"item_id": "c", "modality": "text", "text": "右下に欠け"}]
    assert sorted(bad.item_id) == ["b", "d"]


def test_queries_manifest_requires_text():
    """CI-S11"""
    t = pd.DataFrame({"query_id": ["q01", "q02"], "query_text": ["左の線キズ", ""]})
    with pytest.raises(ValueError):
        queries_manifest(t)
    t.loc[1, "query_text"] = " 中央の溶け "
    assert list(queries_manifest(t).text) == ["左の線キズ", "中央の溶け"]


def _emb(path: Path, vecs):
    path.write_text("\n".join(json.dumps({"item_id": k, "vector": v}) for k, v in vecs.items()), encoding="utf-8")


def test_evaluate_strict_and_loose(tmp_path):
    """CI-S10"""
    pos = _pos()
    t = pd.DataFrame([{"query_id": "q01", "category": "c", "defect_type": "s", "cell": "右", "query_text": "右の s"}])
    vec = {f"i{i}": [np.cos(i / 3), np.sin(i / 3)] for i in range(9)}
    _emb(tmp_path / "img.jsonl", vec)
    cap = {k: v for k, v in vec.items() if k != "i3"}       # i3（厳しい正解）の説明文は作れなかった
    _emb(tmp_path / "cap.jsonl", cap)
    _emb(tmp_path / "qa.jsonl", {"q01": [np.cos(3 / 3), np.sin(3 / 3)]})
    _emb(tmp_path / "qb.jsonl", {"q01": [np.cos(3 / 3), np.sin(3 / 3)]})
    emb = {"captions": tmp_path / "cap.jsonl", "images": tmp_path / "img.jsonl",
           "query_a": tmp_path / "qa.jsonl", "query_b": tmp_path / "qb.jsonl"}
    res, summ = evaluate(pos, t, emb, k=2)
    s = summ[summ.query_id.isna()].set_index("method")
    assert s.at["B_image", "hit_at_1_strict"] == 1 and s.at["B_image", "n_index"] == 9
    assert s.at["A_caption", "hit_at_2_strict"] == 0 and s.at["A_caption", "n_index"] == 8
    assert s.at["A_caption", "hit_at_2_loose"] == 1   # 隣の i2・i4 のうち i2 は c/s
    assert s.at["B_image", "chance_precision_at_2_strict"] == pytest.approx(1 / 9)
    assert set(res.columns) >= {"strict", "loose", "hit_cell"}
    assert CELLS[0] == "左上" and len(CELLS) == 9
