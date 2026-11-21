"""MVTec AD / AD 2 / LOCO AD の一覧と抜き出し（docs/contracts.md の CI-V1〜V6）。データは合成のディレクトリで作る。"""

import pandas as pd
import pytest

from vlmlab import mvtec


def touch(d, n, start=0):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        (d / f"{i:03d}.png").write_bytes(b"x")


def make_ad(root, cat="hazelnut", train=10, good=30, defects=(("crack", 20), ("hole", 15))):
    touch(root / cat / "train" / "good", train)
    touch(root / cat / "test" / "good", good)
    for name, n in defects:
        touch(root / cat / "test" / name, n)
    touch(root / cat / "ground_truth" / defects[0][0], 3)  # マスクは一覧に入らない


def make_ad2(root, cat="walnuts", train=10, val=4, good=30, bad=30):
    touch(root / cat / "train" / "good", train)
    touch(root / cat / "validation" / "good", val)
    touch(root / cat / "test_public" / "good", good)
    touch(root / cat / "test_public" / "bad", bad)
    touch(root / cat / "test_public" / "ground_truth" / "bad", 3)
    touch(root / cat / "test_private", 5)
    touch(root / cat / "test_private_mixed", 5)


def make_loco(root, cat="breakfast_box", train=10, val=4, good=60, structural=30, logical=30):
    touch(root / cat / "train" / "good", train)
    touch(root / cat / "validation" / "good", val)
    touch(root / cat / "test" / "good", good)
    touch(root / cat / "test" / "structural_anomalies", structural)
    touch(root / cat / "test" / "logical_anomalies", logical)


def test_ad_listing_labels_and_paths(tmp_path, monkeypatch):
    """CI-V1：good は normal、それ以外の下位ディレクトリは anomaly。path は image_root からの相対。ground_truth は入らない。"""
    root = tmp_path / "mvtec_ad"
    make_ad(root)
    monkeypatch.setitem(mvtec.EXPECTED_AD, "hazelnut", (10, 30, 35))
    lst = mvtec.list_ad(root, tmp_path, ["hazelnut"])
    t = lst.table
    assert lst.warnings == []
    assert t.groupby(["split", "label"]).size().to_dict() == {
        ("test", "anomaly"): 35, ("test", "normal"): 30, ("train", "normal"): 10}
    assert set(t[t.label == "anomaly"].subtype) == {"crack", "hole"}
    assert t.path.iloc[0].startswith("mvtec_ad/hazelnut/") and "\\" not in "".join(t.path)
    assert not t.path.str.contains("ground_truth").any()
    assert t.image_id.is_unique and "mvtec_ad/hazelnut/test/crack/000" in set(t.image_id)


def test_count_mismatch_stops_unless_allowed(tmp_path):
    """CI-V2：枚数が論文の表と違えば止める。allow なら警告にして続ける。"""
    root = tmp_path / "mvtec_ad"
    make_ad(root)
    with pytest.raises(mvtec.LayoutError, match="differ from the paper"):
        mvtec.list_ad(root, tmp_path, ["hazelnut"])
    lst = mvtec.list_ad(root, tmp_path, ["hazelnut"], allow_count_mismatch=True)
    assert len(lst.warnings) == 1 and len(lst.table) == 75


def test_unknown_category_and_missing_dir(tmp_path):
    """CI-V2：表に無い品目は確かめられないので止める。ディレクトリが無ければ止める。"""
    root = tmp_path / "mvtec_ad"
    make_ad(root, cat="mystery")
    with pytest.raises(mvtec.LayoutError, match="not in the paper"):
        mvtec.list_ad(root, tmp_path, ["mystery"])
    with pytest.raises(mvtec.LayoutError, match="directory not found"):
        mvtec.list_ad(root, tmp_path, ["hazelnut"], allow_count_mismatch=True)


def test_name_normalization_matches_paper_table(tmp_path, monkeypatch):
    """CI-V2：wall_plugs と wallplugs のような表記の違いは同じ品目として照合する。"""
    root = tmp_path / "mvtec_ad_2"
    make_ad2(root, cat="wall_plugs")
    monkeypatch.setitem(mvtec.EXPECTED_AD2, "wallplugs", (10, 4, 30, 30))
    assert mvtec.list_ad2(root, tmp_path, ["wall_plugs"]).warnings == []


def test_ad2_uses_only_test_public(tmp_path, monkeypatch):
    """CI-V1：AD 2 は test_public の good・bad だけ。test_private・validation・ground_truth は一覧に入らない。"""
    root = tmp_path / "mvtec_ad_2"
    make_ad2(root)
    monkeypatch.setitem(mvtec.EXPECTED_AD2, "walnuts", (10, 4, 30, 30))
    t = mvtec.list_ad2(root, tmp_path, ["walnuts"]).table
    assert set(t.split) == {"train", "test_public"}
    assert t[t.split == "test_public"].groupby(["label", "subtype"]).size().to_dict() == {
        ("anomaly", "bad"): 30, ("normal", "good"): 30}
    assert not t.path.str.contains("private|validation|ground_truth").any()


def test_loco_requires_exact_subdirectories(tmp_path, monkeypatch):
    """CI-V1：LOCO の test は good・logical_anomalies・structural_anomalies の3つでなければ止める。"""
    root = tmp_path / "mvtec_loco_ad"
    make_loco(root)
    monkeypatch.setitem(mvtec.EXPECTED_LOCO, "breakfastbox", (10, 4, 60, 30, 30))
    t = mvtec.list_loco(root, tmp_path, ["breakfast_box"]).table
    assert t[t.split == "test"].groupby("subtype").size().to_dict() == {
        "good": 60, "logical_anomalies": 30, "structural_anomalies": 30}
    touch(root / "breakfast_box" / "test" / "other", 2)
    with pytest.raises(mvtec.LayoutError, match="expected subdirectories"):
        mvtec.list_loco(root, tmp_path, ["breakfast_box"])


def loco_table(tmp_path):
    root = tmp_path / "mvtec_loco_ad"
    for cat in ("breakfast_box", "screw_bag"):
        make_loco(root, cat=cat)
    return mvtec.list_loco(root, tmp_path, ["breakfast_box", "screw_bag"], allow_count_mismatch=True).table


def test_sample_counts_and_determinism(tmp_path):
    """CI-V3・V4：品目×種類ごとに指定の枚数。入力の並びを変えても同じ一覧（random_state=42）。"""
    t = loco_table(tmp_path)
    counts = mvtec.DEFAULT_COUNTS["mvtec_loco_ad"]
    a = mvtec.sample_test(t, counts)
    b = mvtec.sample_test(t.sample(frac=1, random_state=7), counts)
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))
    assert len(a) == 200
    assert (a.groupby(["category", "subtype"]).size() == pd.Series({
        ("breakfast_box", "good"): 50, ("breakfast_box", "logical_anomalies"): 25,
        ("breakfast_box", "structural_anomalies"): 25, ("screw_bag", "good"): 50,
        ("screw_bag", "logical_anomalies"): 25, ("screw_bag", "structural_anomalies"): 25})).all()
    assert (a.split == "test").all() and list(a.columns) == mvtec.SPLIT_COLUMNS


def test_sample_pools_defect_types_for_ad(tmp_path):
    """CI-V3：MVTec AD の不良品は欠陥の種類をまとめた中から選ぶ（"anomaly" のキー）。"""
    root = tmp_path / "mvtec_ad"
    make_ad(root)
    t = mvtec.list_ad(root, tmp_path, ["hazelnut"], allow_count_mismatch=True).table
    s = mvtec.sample_test(t, mvtec.DEFAULT_COUNTS["mvtec_ad"])
    assert s.groupby("label").size().to_dict() == {"anomaly": 25, "normal": 25}
    assert set(s[s.label == "anomaly"].subtype) <= {"crack", "hole"}


def test_sample_fails_when_short(tmp_path):
    """CI-V3：足りなければ止める（他の種類や品目で埋めない）。"""
    root = tmp_path / "mvtec_ad_2"
    make_ad2(root, good=20)
    t = mvtec.list_ad2(root, tmp_path, ["walnuts"], allow_count_mismatch=True).table
    with pytest.raises(ValueError, match="need 25 images but only 20"):
        mvtec.sample_test(t, mvtec.DEFAULT_COUNTS["mvtec_ad_2"])


def test_refs_from_train_and_no_overlap(tmp_path):
    """CI-V5：見本は train の良品から品目ごとに3枚、ref_rank 1〜3。テストと重ならない。"""
    t = loco_table(tmp_path)
    refs = mvtec.sample_refs(t)
    test = mvtec.sample_test(t, mvtec.DEFAULT_COUNTS["mvtec_loco_ad"])
    assert len(refs) == 6 and (refs.split == "train").all() and (refs.label == "normal").all()
    assert refs.groupby("category").ref_rank.apply(list).to_dict() == {
        "breakfast_box": [1, 2, 3], "screw_bag": [1, 2, 3]}
    assert not set(refs.image_id) & set(test.image_id)
    mvtec.write_split(test, refs, tmp_path / "s.csv", tmp_path / "s_refs.csv")
    assert (tmp_path / "s.csv").read_bytes().count(b"\r") == 0
    with pytest.raises(ValueError, match="overlap"):
        mvtec.write_split(test, test.assign(ref_rank=1), tmp_path / "x.csv", tmp_path / "x_refs.csv")


def test_split_csv_is_readable_by_run(tmp_path):
    """CI-V6：作った一覧は vlmlab run の plan_run でそのまま読める（見本は品目ごとに引ける）。"""
    from vlmlab.run import RunConfig, plan_run

    t = loco_table(tmp_path)
    split_dir = tmp_path / "data" / "splits"
    mvtec.write_split(mvtec.sample_test(t, mvtec.DEFAULT_COUNTS["mvtec_loco_ad"]), mvtec.sample_refs(t),
                      split_dir / "loco.csv", split_dir / "loco_refs.csv")
    (tmp_path / "p.txt").write_text("judge", encoding="utf-8")
    cfg = RunConfig(name="x", split="data/splits/loco.csv", refs="data/splits/loco_refs.csv", n_refs=1,
                    model_id="m", region="r", prompt="p.txt", budget_usd=0.1)
    plan = plan_run(cfg, tmp_path, tmp_path / "runs", limit=10)
    assert len(plan.items) == 10
    assert set(plan.refs_by_cat) == {"breakfast_box", "screw_bag"}
    assert all((tmp_path / p).exists() for paths in plan.refs_by_cat.values() for p in paths)
