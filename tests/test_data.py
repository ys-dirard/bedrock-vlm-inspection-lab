import pandas as pd
import pytest
from PIL import Image

from vlmlab.data import CATEGORIES, load_image, make_refs, make_test_split, make_wide_split


def synthetic_table():
    rows = []
    for cat in CATEGORIES:
        for split, label, n in (("train", "normal", 20), ("test", "normal", 60), ("test", "anomaly", 60)):
            for i in range(n):
                rows.append({"image_id": f"{cat}/{label}/{split}{i}", "category": cat, "label": label,
                             "split": split, "path": f"{cat}/{split}/{label}/{i}.JPG"})
    return pd.DataFrame(rows)


def test_splits_are_deterministic_and_balanced():
    t = synthetic_table()
    a, b = make_test_split(t), make_test_split(t)
    assert a.equals(b)
    assert len(a) == 200
    assert a.groupby(["category", "label"]).size().to_dict() == {
        ("capsules", "anomaly"): 50, ("capsules", "normal"): 50, ("pcb1", "anomaly"): 50, ("pcb1", "normal"): 50}
    assert (a.split == "test").all()


def test_refs_come_from_train_normal_and_do_not_overlap():
    t = synthetic_table()
    refs = make_refs(t)
    assert len(refs) == 36 and (refs.split == "train").all() and (refs.label == "normal").all()
    assert set(refs.ref_rank) == {1, 2, 3}
    assert not set(refs.image_id) & set(make_test_split(t).image_id)


def test_wide_split():
    w = make_wide_split(synthetic_table())
    assert len(w) == 120 and (w.groupby(["category", "label"]).size() == 5).all()


def test_too_few_rows_raises():
    t = synthetic_table()
    t = t[~((t.category == "pcb1") & (t.label == "anomaly"))]
    with pytest.raises(ValueError):
        make_test_split(t)


def test_load_image_original_and_resized(tmp_path):
    p = tmp_path / "a.JPG"
    Image.new("RGB", (1500, 1000), (10, 20, 30)).save(p, "JPEG")
    orig = load_image(p)
    assert orig.data == p.read_bytes() and (orig.width, orig.height) == (1500, 1000) and orig.format == "jpeg"
    small = load_image(p, 768)
    assert (small.width, small.height) == (768, 512) and small.format == "jpeg"
    assert load_image(p, 2000).data == p.read_bytes()  # 拡大はしない
