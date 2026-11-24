"""anomalydino.py の計算（torch を使わない部分）。"""

import json

import numpy as np
import pandas as pd
import pytest

from vlmlab.anomalydino import AdConfig, image_score, patch_distances, select_refs


def _post_eval_mean_top1p(distances):
    """公式 src/post_eval.py の mean_top1p（commit dd35b0f）をそのまま写したもの。突き合わせ用。"""
    if int(len(distances) * 0.01) == 0:
        return np.max(distances)
    return np.mean(sorted(distances.flatten(), reverse=True)[:int(len(distances) * 0.01)])


def test_patch_distance_is_one_minus_max_cosine():
    """CI-A2：最も近い見本のパッチとの 1 − コサイン類似度。大きさ（ノルム）には依らない。"""
    bank = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    test = np.array([[3.0, 0.0], [1.0, 1.0], [-2.0, 0.0]], dtype=np.float32)
    d = patch_distances(test, np.array([True, True, True]), bank)
    assert d == pytest.approx([0.0, 1 - 1 / np.sqrt(2), 1.0], abs=1e-6)


def test_patch_distance_masked_patches_are_zero():
    """CI-A2：背景として除いたパッチは 0。"""
    bank = np.array([[1.0, 0.0]], dtype=np.float32)
    test = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    d = patch_distances(test, np.array([True, False]), bank)
    assert d.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)


def test_patch_distance_matches_faiss_style_normalized_l2():
    """CI-A2：公式の L2_normalized（正規化した特徴の2乗 L2 距離 / 2）と同じ値。"""
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(50, 16)).astype(np.float32)
    test = rng.normal(size=(20, 16)).astype(np.float32)
    bn = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    tn = test / np.linalg.norm(test, axis=1, keepdims=True)
    sq = ((tn[:, None, :] - bn[None, :, :]) ** 2).sum(-1).min(axis=1) / 2
    d = patch_distances(test, np.ones(20, dtype=bool), bank)
    assert d == pytest.approx(sq, abs=1e-5)


def test_patch_distance_rejects_bad_shapes():
    with pytest.raises(ValueError):
        patch_distances(np.zeros((3, 4)), np.ones(3, bool), np.ones((2, 5)))
    with pytest.raises(ValueError):
        patch_distances(np.ones((3, 4)), np.ones(2, bool), np.ones((2, 4)))


def test_patch_distance_rejects_zero_vector():
    with pytest.raises(ValueError):
        patch_distances(np.zeros((1, 2)), np.ones(1, bool), np.ones((1, 2)))


def test_image_score_top1_percent_mean():
    """CI-A3：N=250 なら上位 2 個の平均。"""
    d = np.zeros(250)
    d[[3, 7, 9]] = [0.9, 0.5, 0.7]
    assert image_score(d) == pytest.approx(0.8)


def test_image_score_max_when_fewer_than_100():
    """CI-A3：floor(0.01 N) が 0 なら最大値。"""
    assert image_score(np.array([0.1, 0.4, 0.2])) == pytest.approx(0.4)


@pytest.mark.parametrize("n", [1, 99, 100, 101, 199, 200, 1024, 1536])
def test_image_score_matches_official_mean_top1p(n):
    """CI-A3：公式の mean_top1p と同じ値。"""
    d = np.random.default_rng(n).random(n)
    assert image_score(d) == pytest.approx(float(_post_eval_mean_top1p(d)), abs=1e-9)


def test_image_score_rejects_nan_and_empty():
    with pytest.raises(ValueError):
        image_score(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        image_score(np.array([]))


def _refs():
    return pd.DataFrame({
        "image_id": ["a/normal/r1", "a/normal/r2", "b/normal/r1"],
        "ref_rank": [1, 2, 1], "category": ["a", "a", "b"], "label": ["normal"] * 3,
        "split": ["train"] * 3, "path": ["x", "y", "z"],
    })


def _test():
    return pd.DataFrame({"image_id": ["a/normal/t1", "b/anomaly/t2"], "category": ["a", "b"],
                         "label": ["normal", "anomaly"]})


def test_select_refs_rank1_per_category():
    """CI-A1：品目ごとに ref_rank 1 の1枚。"""
    out = select_refs(_test(), _refs(), 1)
    assert {c: [r["image_id"] for r in rs] for c, rs in out.items()} == {"a": ["a/normal/r1"], "b": ["b/normal/r1"]}


def test_select_refs_stops_when_not_enough():
    """CI-A1：品目 b に2枚目が無いので止める。"""
    with pytest.raises(ValueError, match="b"):
        select_refs(_test(), _refs(), 2)


def test_select_refs_stops_on_overlap():
    """CI-A1：テストと見本が重なれば止める。"""
    t = _test()
    t.loc[0, "image_id"] = "a/normal/r1"
    with pytest.raises(ValueError, match="overlap"):
        select_refs(t, _refs(), 1)


def test_identity_excludes_device_and_threads(tmp_path):
    """CI-A5：device・num_threads は異常度の中身を変えないので identity に入れない。"""
    base = dict(name="x", split="s", refs="r", anomalydino_commit="a", dinov2_commit="d")
    a = AdConfig(**base, device="cpu", num_threads=4).identity("h")
    b = AdConfig(**base, device="cuda", num_threads=None).identity("h")
    assert a == b
    assert AdConfig(**base, resolution=672).identity("h") != a
    p = tmp_path / "c.yaml"
    p.write_text(json.dumps({**base, "bogus": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        AdConfig.load(p)
