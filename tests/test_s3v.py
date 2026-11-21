"""S3 Vectors への読み込みと比較（vlmlab.search.s3v）の単体テスト。AWS は呼ばない。"""

import numpy as np
import pandas as pd
import pytest

from vlmlab.search.s3v import S3Vectors, batches, build_records, compare_hits, put_bytes, select_rows


def _meta():
    return pd.DataFrame({"item_id": ["a", "b", "c"], "category": ["x", "x", "y"], "role": ["index", "query", "index"],
                         "text": ["本文A", "本文B", "本文C"]})


def test_select_rows():
    """CI-S17"""
    assert list(select_rows(_meta(), {"role": ["index"]}).item_id) == ["a", "c"]
    assert len(select_rows(_meta(), {})) == 3
    with pytest.raises(ValueError):
        select_rows(_meta(), {"nope": ["1"]})


def test_build_records():
    """CI-S17"""
    v = np.array([[0.1, 0.2], [0.3, 0.4]])
    r = build_records(["a", "c"], v, _meta(), ["category"], text_col="text")
    assert r[1] == {"key": "c", "data": {"float32": [pytest.approx(0.3), pytest.approx(0.4)]},
                    "metadata": {"category": "y", "text": "本文C"}}
    assert set(build_records(["a"], v[:1], _meta(), [])[0]["metadata"]) == set()
    for ids, vec in [(["a", "a"], v), (["a"], v), (["z", "a"], v), (["a", "c"], np.array([[np.nan, 0], [1, 1]]))]:
        with pytest.raises(ValueError):
            build_records(ids, vec, _meta(), [])


def test_batches_and_bytes():
    """CI-S17"""
    assert [len(b) for b in batches(list(range(1201)))] == [500, 500, 201]
    with pytest.raises(ValueError):
        batches([1], 501)
    r = build_records(["a"], np.array([[1.0, 2.0, 3.0]]), _meta(), ["category"])
    assert put_bytes(r) == 4 * 3 + 1 + len("category") + 1


def test_compare_hits():
    """CI-S18"""
    local = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    assert compare_hits(local, [("a", 0.1), ("b", 0.2), ("c", 0.3)]) == {
        "same_order": True, "same_set": True, "overlap": 3, "max_abs_score_diff": pytest.approx(0.0)}
    c = compare_hits(local, [("b", 0.2), ("a", 0.1000005), ("d", 0.25)])
    assert not c["same_order"] and not c["same_set"] and c["overlap"] == 2
    assert c["max_abs_score_diff"] == pytest.approx(5e-7)
    assert compare_hits(local, [])["max_abs_score_diff"] is None


class FakeClient:
    def __init__(self, existing=None):
        self.existing = existing or {}
        self.created, self.puts = [], []

    def list_indexes(self, vectorBucketName):
        return {"indexes": [{"indexName": n} for n in self.existing]}

    def get_index(self, vectorBucketName, indexName):
        return {"index": self.existing[indexName]}

    def create_index(self, **kw):
        self.created.append(kw)

    def put_vectors(self, vectorBucketName, indexName, vectors):
        self.puts.append(len(vectors))


def test_ensure_index_and_put():
    """CI-S17"""
    c = FakeClient({"old": {"dimension": 1024, "distanceMetric": "cosine"},
                    "bad": {"dimension": 256, "distanceMetric": "cosine"}})
    s = S3Vectors("b", client=c)
    assert s.ensure_index("old", 1024, []) is False and c.created == []
    with pytest.raises(RuntimeError):
        s.ensure_index("bad", 1024, [])
    assert s.ensure_index("new", 1024, ["text"]) is True
    assert c.created[0]["metadataConfiguration"] == {"nonFilterableMetadataKeys": ["text"]}
    assert c.created[0]["distanceMetric"] == "cosine" and c.created[0]["dataType"] == "float32"
    assert s.put("new", [{"key": str(i)} for i in range(501)], log=lambda _: None) == 2 and c.puts == [500, 1]
