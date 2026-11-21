import numpy as np
import pandas as pd
import pytest

from vlmlab import noise as N
from vlmlab.anomalydino import image_score, patch_distances
from vlmlab.noise_dino import sim_rows


def table():
    rows = []
    for cat in ("capsules", "pcb1", "candle"):
        for split, label, n in (("train", "normal", 260), ("test", "normal", 60), ("test", "anomaly", 80)):
            for i in range(n):
                iid = f"{cat}/{label}/{split}{i}"
                rows.append({"image_id": iid, "category": cat, "label": label, "split": split, "path": f"{iid}.JPG"})
    return pd.DataFrame(rows)


def test_make_noisy_split_counts_ids_and_truth():
    t = table()
    exclude = {f"pcb1/anomaly/test{i}" for i in range(60)} | {"capsules/normal/train0"}
    run, truth = N.make_noisy_split(t, exclude)
    run2, truth2 = N.make_noisy_split(t, exclude)
    assert run.equals(run2) and truth.equals(truth2)
    assert len(run) == 420 and list(run.columns) == N.RUN_COLUMNS
    assert set(run["label"]) == {"unknown"} and set(run["category"]) == {"capsules", "pcb1"}
    assert run["image_id"].str.fullmatch(r"(capsules|pcb1)/n\d{3}").all()
    assert truth.groupby("category")["injected"].sum().to_dict() == {"capsules": 10, "pcb1": 10}
    assert not set(truth["source_image_id"]) & exclude
    assert (truth.loc[truth.injected, "source_image_id"].str.contains("/anomaly/test")).all()
    assert (truth.loc[~truth.injected, "source_image_id"].str.contains("/normal/train")).all()
    # 混入は並びの後ろにまとまっていない（混ぜてある）
    assert truth.loc[truth.category == "pcb1", "injected"].iloc[-10:].sum() < 10
    with pytest.raises(ValueError):
        N.make_noisy_split(t, {f"pcb1/anomaly/test{i}" for i in range(75)})


def _rand(rng, n, d=8):
    return rng.normal(size=(n, d)).astype(np.float32)


def test_sim_table_score_matches_direct_computation():
    rng = np.random.default_rng(0)
    bank_feats = [_rand(rng, 50), _rand(rng, 40), _rand(rng, 30)]
    x = _rand(rng, 300)
    mask = rng.random(300) > 0.3
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    bn = [b / np.linalg.norm(b, axis=1, keepdims=True) for b in bank_feats]
    t = N.SimTable(["x"], ["a", "b", "c"], [sim_rows(xn[mask], bn)], [300])
    for use in ([1, 1, 1], [1, 0, 1], [0, 1, 0]):
        use = np.array(use, dtype=bool)
        bank = np.concatenate([b for b, u in zip(bank_feats, use) if u])
        expect = image_score(patch_distances(x, mask, bank))
        assert t.score("x", use) == pytest.approx(expect, abs=1e-6)
    with pytest.raises(ValueError):
        t.score("x", np.zeros(3, dtype=bool))


def test_leave_one_out_excludes_self_and_bank_subset():
    # 自分の列は 1.0（距離 0）。自分を除けば、ほかの列の値で決まる
    sims = {
        "a": np.array([[1.0, 0.2, 0.5]] * 100, dtype=np.float32),
        "b": np.array([[0.9, 1.0, 0.1]] * 100, dtype=np.float32),
        "c": np.array([[0.3, 0.4, 1.0]] * 100, dtype=np.float32),
    }
    t = N.SimTable(["a", "b", "c"], ["a", "b", "c"], [sims[k] for k in "abc"], [100, 100, 100])
    loo = t.leave_one_out()
    assert loo.to_dict() == pytest.approx({"a": 0.5, "b": 0.1, "c": 0.6})
    assert t.scores_with_bank({"c"}).to_dict() == pytest.approx({"a": 0.5, "b": 0.9, "c": 0.0})
    with pytest.raises(ValueError):
        N.SimTable(["a"], ["b"], [sims["a"]], [100])


def test_background_patches_count_as_zero():
    # 前景 1 パッチ（距離 0.8）・総数 200：k = 2 なので (0.8 + 0) / 2
    t = N.SimTable(["x"], ["a"], [np.array([[0.2]], dtype=np.float32)], [200])
    assert t.score("x", np.array([True])) == pytest.approx(0.4)


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    t = N.SimTable(["x", "y"], ["a", "b"], [rng.random((3, 2), dtype=np.float32), rng.random((0, 2), dtype=np.float32)],
                   [5, 4])
    N.save_sim_table(t, tmp_path / "t.npz")
    u = N.load_sim_table(tmp_path / "t.npz")
    assert u.rows == t.rows and u.bank == t.bank and u.n_patches == t.n_patches
    assert all(np.array_equal(a, b) for a, b in zip(u.sims, t.sims))


def test_top_fraction_ceil_and_ties():
    s = pd.DataFrame({"image_id": ["p/1", "p/2", "p/3", "q/1", "q/2"], "category": ["p", "p", "p", "q", "q"],
                      "score": [0.5, 0.9, 0.9, 0.1, 0.2]})
    sus = N.top_fraction(s, 10)  # ceil(0.3) = 1 枚ずつ
    assert sus.to_dict() == {"p/1": False, "p/2": True, "p/3": False, "q/1": False, "q/2": True}
    assert N.top_fraction(s, 50).sum() == 2 + 1


def test_detection_summary():
    truth = pd.DataFrame({"image_id": ["p/1", "p/2", "p/3", "p/4"], "category": ["p"] * 4,
                          "injected": [True, True, False, False]})
    sus = pd.Series({"p/1": True, "p/2": False, "p/3": True, "p/4": False})
    unj = pd.Series({"p/1": False, "p/2": True, "p/3": False, "p/4": False})
    rows = N.detection_summary(sus, truth, "m", unj)
    assert [r["category"] for r in rows] == ["p", "all"]
    r = rows[0]
    assert (r["n_found"], r["detection_rate"], r["n_normal_suspected"], r["normal_suspect_rate"], r["n_unjudged"]) == \
        (1, 0.5, 1, 0.5, 1)
    with pytest.raises(ValueError):
        N.detection_summary(sus.iloc[:3], truth, "m")


def test_threshold_and_test_metrics():
    test = pd.DataFrame({"image_id": list("abcdef"), "label": ["normal"] * 3 + ["anomaly"] * 3})
    scores = pd.Series({"a": 0.1, "b": 0.2, "c": 0.5, "d": 0.3, "e": 0.6, "f": 0.7})
    m = N.eval_test_metrics(scores, test, 0.0)
    assert m["threshold"] == 0.3 and m["miss_rate_at_t"] == 0.0
    assert m["false_positive_rate_at_t"] == pytest.approx(1 / 3)
    assert m["auroc"] == pytest.approx(8 / 9)
    assert N.threshold_at_miss(np.array([0.3, 0.6, 0.7]), np.array([True] * 3), 0.34) == 0.6
    with pytest.raises(ValueError):
        N.eval_test_metrics(scores.drop("a"), test)


@pytest.mark.parametrize("sizes", [[7, 7, 7, 7, 7], [5, 9, 3, 8, 6]])
def test_sim_rows_torch_matches_numpy(sizes):
    torch = pytest.importorskip("torch")
    from vlmlab.noise_dino import sim_rows_torch
    rng = np.random.default_rng(2)
    bank = [_rand(rng, n) for n in sizes]
    bank = [b / np.linalg.norm(b, axis=1, keepdims=True) for b in bank]
    x = _rand(rng, 11)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    bank_t = torch.from_numpy(np.concatenate(bank, axis=0))
    got = sim_rows_torch(x, bank_t, sizes, chunk=2)  # 区切りをまたぐ場合も確かめる
    assert np.allclose(got, sim_rows(x, bank), atol=1e-6)
