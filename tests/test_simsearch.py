"""12/20 の記事（似た不良画像を探す）の分割と評価の単体テスト。AWS は呼ばない。"""

import json
from pathlib import Path

import pandas as pd
import pytest

from vlmlab.search.simsearch import evaluate, make_split


# ---- 12/20 の分割（CI-D20）と評価（CI-S8）----

def _fake_visa(root: Path, cats, n_anom_per_type, n_normal=12):
    rows = []
    for cat in cats:
        anno = []
        k = 0
        for typ, n in n_anom_per_type.items():
            for _ in range(n):
                p = f"{cat}/Data/Images/Anomaly/{k:03d}.JPG"
                anno.append({"image": p, "label": typ, "mask": p.replace("Images", "Masks").replace("JPG", "png")})
                rows.append({"object": cat, "split": "test", "label": "anomaly", "image": p, "mask": ""})
                k += 1
        # 複数のラベル・other は使われない
        for lab_ in ["a,b", "other"]:
            p = f"{cat}/Data/Images/Anomaly/{k:03d}.JPG"
            anno.append({"image": p, "label": lab_, "mask": ""})
            rows.append({"object": cat, "split": "test", "label": "anomaly", "image": p, "mask": ""})
            k += 1
        for i in range(n_normal):
            p = f"{cat}/Data/Images/Normal/{i:04d}.JPG"
            anno.append({"image": p, "label": "normal", "mask": ""})
            rows.append({"object": cat, "split": "test", "label": "normal", "image": p, "mask": ""})
        (root / cat).mkdir(parents=True)
        pd.DataFrame(anno).to_csv(root / cat / "image_anno.csv", index=False)
    (root / "split_csv").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "split_csv" / "1cls.csv", index=False)


def test_make_split_rules(tmp_path):
    """CI-D20"""
    _fake_visa(tmp_path, ["c1", "c2"], {"scratch": 6, "melt": 4, "bent": 2})
    d = make_split(tmp_path, ["c1", "c2"], n_per_cat=12, n_query=3, n_normal=5)
    assert d.role.value_counts().to_dict() == {"index": 18, "query": 6, "normal_index": 10}
    assert not (d.defect_type.str.contains(",") | (d.defect_type == "other")).any()
    for (cat, typ), g in d[d.role == "query"].groupby(["category", "defect_type"]):
        left = d[(d.role == "index") & (d.category == cat) & (d.defect_type == typ)]
        assert len(left) >= 2
    assert set(d[d.role == "query"].defect_type) <= {"scratch", "melt"}  # 2枚しかない bent はクエリにしない
    d2 = make_split(tmp_path, ["c1", "c2"], n_per_cat=12, n_query=3, n_normal=5)
    pd.testing.assert_frame_equal(d, d2)  # 乱数を固定


def test_make_split_refuses_small_pool(tmp_path):
    """CI-D20"""
    _fake_visa(tmp_path, ["c1"], {"scratch": 3})
    with pytest.raises(ValueError):
        make_split(tmp_path, ["c1"], n_per_cat=12, n_query=1, n_normal=5)


def _write_emb(path: Path, vecs: dict[str, list[float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in vecs.items():
            f.write(json.dumps({"item_id": k, "vector": v}) + "\n")


def test_evaluate_counts_relevant_by_category_and_type(tmp_path):
    """CI-S8"""
    split = pd.DataFrame([
        {"item_id": "q1", "category": "c", "label": "anomaly", "defect_type": "s", "role": "query"},
        {"item_id": "a1", "category": "c", "label": "anomaly", "defect_type": "s", "role": "index"},
        {"item_id": "a2", "category": "c", "label": "anomaly", "defect_type": "m", "role": "index"},
        {"item_id": "a3", "category": "d", "label": "anomaly", "defect_type": "s", "role": "index"},  # 別品目の同名
        {"item_id": "n1", "category": "c", "label": "normal", "defect_type": "", "role": "normal_index"},
    ])
    _write_emb(tmp_path / "i.jsonl", {"a1": [0.8, 0.6], "a2": [0.6, 0.8], "a3": [1, 0.01], "n1": [1, 0]})
    _write_emb(tmp_path / "q.jsonl", {"q1": [1, 0]})
    res, summ = evaluate(split, tmp_path / "i.jsonl", tmp_path / "q.jsonl", k=2)
    allrow = summ[summ.category == "(all)"].set_index("variant")
    assert allrow.at["anomaly_only", "hit_at_1"] == 0          # 1位は別品目の a3
    assert allrow.at["anomaly_only", "hit_at_2"] == 1 and allrow.at["anomaly_only", "precision_at_2"] == 0.5
    assert allrow.at["with_normals", "normals_in_top2"] == 1   # 良品 n1 が1位
    assert allrow.at["with_normals", "chance_precision_at_2"] == 0.25
    assert allrow.at["with_normals", "chance_precision_at_2_in_category"] == pytest.approx(1 / 3)  # 品目 c は3件
    assert allrow.at["anomaly_only", "chance_precision_at_2_in_category"] == 0.5
    assert set(res.variant) == {"anomaly_only", "with_normals"}
