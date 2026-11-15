"""12/5 の記事：検査対象の 2×2 分割（CI-T*）、欠陥の大きさの区分（CI-S*）、分割した画像を渡す実行（CI-R4・R5）。"""

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmlab.data import assign_size_bins, grid_boxes, load_tiles, make_defect_size_table, mask_area
from vlmlab.run import RunConfig, execute, max_embedded_images, plan_run

from test_run_and_report import FakeModel, lab  # noqa: F401  (fixture)


# ---- 分割 ----

def test_grid_boxes_even():
    """CI-T1: 行優先（左上・右上・左下・右下）で、重なりなく全体を覆う。"""
    assert grid_boxes(100, 60) == [(0, 0, 50, 30), (50, 0, 100, 30), (0, 30, 50, 60), (50, 30, 100, 60)]


@pytest.mark.parametrize("w,h", [(1404, 1070), (1500, 1000), (101, 61), (3, 2), (2, 2)])
def test_grid_boxes_cover_without_overlap(w, h):
    """CI-T2: 各画素がちょうど1つの枠に入る。端数は右の列と下の行に入る。"""
    boxes = grid_boxes(w, h)
    cover = np.zeros((h, w), dtype=int)
    for l, t, r, b in boxes:
        assert r > l and b > t
        cover[t:b, l:r] += 1
    assert (cover == 1).all()
    assert boxes[0][2] - boxes[0][0] == w // 2 and boxes[1][2] - boxes[1][0] == w - w // 2
    assert boxes[0][3] - boxes[0][1] == h // 2 and boxes[2][3] - boxes[2][1] == h - h // 2


def test_grid_boxes_rejects_too_small_and_bad_n():
    """CI-T3: n×n に分けられない大きさ、n < 1 は ValueError。"""
    with pytest.raises(ValueError):
        grid_boxes(1, 5)
    with pytest.raises(ValueError):
        grid_boxes(10, 10, 0)


def test_load_tiles_content_order_and_size(tmp_path):
    """CI-T4: 原寸から切り出した4枚を、左上・右上・左下・右下の順に JPEG で返す。long_edge は各タイルに当てる。"""
    p = tmp_path / "a.JPG"
    im = Image.new("RGB", (101, 61))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for (l, t, r, b), c in zip(grid_boxes(101, 61), colors):
        im.paste(c, (l, t, r, b))
    im.save(p, "JPEG", quality=100)
    tiles = load_tiles(p)
    assert [(x.width, x.height) for x in tiles] == [(50, 30), (51, 30), (50, 31), (51, 31)]
    assert all(x.format == "jpeg" for x in tiles)
    for x, c in zip(tiles, colors):
        with Image.open(io.BytesIO(x.data)) as t:
            px = t.convert("RGB").getpixel((t.width // 2, t.height // 2))
        assert all(abs(a - b) < 30 for a, b in zip(px, c))
    small = load_tiles(p, long_edge=20)
    assert max(small[0].width, small[0].height) == 20
    assert load_tiles(p, long_edge=1000)[0].width == 50  # 拡大はしない


# ---- 欠陥の大きさ ----

def test_mask_area_counts_nonzero(tmp_path):
    """CI-S1: 0 より大きい画素（値が 1 でも 255 でも）を欠陥として数える。"""
    a = np.zeros((10, 20), dtype=np.uint8)
    a[0, :5] = 1
    a[5, :3] = 255
    Image.fromarray(a).save(tmp_path / "m.png")
    assert mask_area(tmp_path / "m.png") == (8, 20, 10, 200)


def test_assign_size_bins_equal_counts_and_ties():
    """CI-S2: 小さい順に件数で3等分（100枚なら 34・33・33）。同じ値は image_id の順。入力の順に依らない。"""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"image_id": [f"x{i:03d}" for i in range(100)], "defect_area_ratio": rng.random(100)})
    bins = assign_size_bins(df)
    assert bins.value_counts().to_dict() == {"small": 34, "medium": 33, "large": 33}
    mx = df.groupby(bins)["defect_area_ratio"].max()
    mn = df.groupby(bins)["defect_area_ratio"].min()
    assert mx["small"] <= mn["medium"] and mx["medium"] <= mn["large"]
    shuffled = df.sample(frac=1, random_state=1)
    assert assign_size_bins(shuffled).sort_index().equals(bins.sort_index())
    ties = pd.DataFrame({"image_id": ["c", "a", "b"], "defect_area_ratio": [0.1, 0.1, 0.1]})
    assert assign_size_bins(ties).tolist() == ["large", "small", "medium"]


def test_assign_size_bins_rejects_empty_and_missing():
    """CI-S2: 空の表、割合の欠けは ValueError。"""
    with pytest.raises(ValueError):
        assign_size_bins(pd.DataFrame({"image_id": [], "defect_area_ratio": []}))
    with pytest.raises(ValueError):
        assign_size_bins(pd.DataFrame({"image_id": ["a"], "defect_area_ratio": [None]}))


def _visa_with_masks(tmp_path, sizes, mask_size=None, empty=None):
    visa = tmp_path / "visa"
    rows, cls = [], []
    for i, n_px in enumerate(sizes):
        img_rel = f"pcb1/Data/Images/Anomaly/{i:03d}.JPG"
        mask_rel = f"pcb1/Data/Masks/Anomaly/{i:03d}.png"
        for rel in (img_rel, mask_rel):
            (visa / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30)).save(visa / img_rel, "JPEG")
        m = np.zeros((30, 40) if mask_size is None else mask_size, dtype=np.uint8)
        if empty != i:
            m.flat[:n_px] = 1
        Image.fromarray(m).save(visa / mask_rel)
        rows.append({"image_id": f"pcb1/anomaly/{i:03d}", "category": "pcb1", "label": "anomaly",
                     "split": "test", "path": img_rel})
        cls.append({"object": "pcb1", "split": "test", "label": "anomaly", "image": img_rel, "mask": mask_rel})
    rows.append({"image_id": "pcb1/normal/000", "category": "pcb1", "label": "normal", "split": "test",
                 "path": "pcb1/Data/Images/Normal/000.JPG"})
    (visa / "split_csv").mkdir(parents=True)
    pd.DataFrame(cls).to_csv(visa / "split_csv" / "1cls.csv", index=False)
    return visa, pd.DataFrame(rows)


def test_make_defect_size_table(tmp_path):
    """CI-S3: 不良品だけを対象にし、割合 = 欠陥の画素 / 画像の画素（小数9桁）。区分は CI-S2。"""
    visa, split = _visa_with_masks(tmp_path, [600, 12, 120])
    t = make_defect_size_table(visa, split)
    assert len(t) == 3 and set(t.columns) >= {"image_id", "defect_px", "image_px", "defect_area_ratio", "size_bin"}
    assert t.set_index("image_id")["defect_area_ratio"].to_dict() == {
        "pcb1/anomaly/000": 0.5, "pcb1/anomaly/001": 0.01, "pcb1/anomaly/002": 0.1}
    assert t.set_index("image_id")["size_bin"].to_dict() == {
        "pcb1/anomaly/000": "large", "pcb1/anomaly/001": "small", "pcb1/anomaly/002": "medium"}


def test_make_defect_size_table_rejects_bad_masks(tmp_path):
    """CI-S4: マスクと画像の大きさが違う、欠陥の画素が 0 のときは ValueError（黙って区分しない）。"""
    visa, split = _visa_with_masks(tmp_path / "a", [5, 6], mask_size=(31, 40))
    with pytest.raises(ValueError, match="size"):
        make_defect_size_table(visa, split)
    visa, split = _visa_with_masks(tmp_path / "b", [5, 6], empty=1)
    with pytest.raises(ValueError, match="no defect"):
        make_defect_size_table(visa, split)


# ---- 実行 ----

def _tile_cfg(base, **kw):
    cfg = json.loads((base / "c.yaml").read_text(encoding="utf-8"))
    cfg.update({"repeats": 1, "tile_grid": 2, **kw})
    (base / "tile.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    return RunConfig.load(base / "tile.yaml")


def test_run_with_tiles_sends_ref_then_four_tiles(lab):  # noqa: F811
    """CI-R5: 見本1枚（分けない）＋検査対象の4タイルの順に渡し、記録に tile_grid を残す。"""
    base, visa, pricing = lab
    cfg = _tile_cfg(base)
    assert cfg.images_per_call == 5
    plan = plan_run(cfg, base, base / "runs")
    model = FakeModel()
    execute(plan, model, pricing, visa, base, log=lambda _: None)
    req = model.requests[0]
    assert len(req.images) == 5
    assert (req.images[0].width, req.images[0].height) == (64, 32)          # 見本は原寸のまま
    assert [(i.width, i.height) for i in req.images[1:]] == [(32, 16)] * 4  # 64x32 の 2×2
    recs = [json.loads(l) for l in (plan.out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["tile_grid"] for r in recs} == {2}
    # 分けないで回す設定では、同じ記録の続きとして再開できない（CI-R1）
    cfg2 = _tile_cfg(base, tile_grid=None)
    with pytest.raises(RuntimeError, match="config differs"):
        execute(plan_run(cfg2, base, base / "runs"), FakeModel(), pricing, visa, base, log=lambda _: None)


def test_image_count_limit_is_checked_before_calls(lab):  # noqa: F811
    """CI-R4: Nova 2 Lite で1回の画像が5枚を超える設定は、呼ぶ前に止める。5枚ちょうどは通す。"""
    base, _, _ = lab
    assert max_embedded_images("jp.amazon.nova-2-lite-v1:0") == 5
    assert max_embedded_images("jp.anthropic.claude-haiku-4-5-20251001-v1:0") is None
    plan_run(_tile_cfg(base, model_id="jp.amazon.nova-2-lite-v1:0"), base, base / "runs")
    with pytest.raises(ValueError, match="exceeds the limit"):
        plan_run(_tile_cfg(base, model_id="jp.amazon.nova-2-lite-v1:0", n_refs=2), base, base / "runs")


def test_tile_grid_config_validation(lab):  # noqa: F811
    """tile_grid は null か 2 以上で、layout は refs_first のときだけ。"""
    base, _, _ = lab
    with pytest.raises(ValueError):
        _tile_cfg(base, tile_grid=1)
    with pytest.raises(ValueError):
        _tile_cfg(base, layout="target_first")


# ---- 区分ごとの再現率 ----

def _rec(image_id, cat, label, pred, repeat=1):
    return {"image_id": image_id, "category": cat, "label": label, "pred": pred, "repeat": repeat}


def test_recall_by_bin_counts_and_parse_fail():
    """CI-M11: 区分ごと・区分×品目ごとの TP/FN/再現率。解析失敗は分母から除き、記録の無い画像は n_missing。良品は数えない。"""
    from vlmlab.metrics import recall_by_bin
    bins = pd.DataFrame({"image_id": ["a", "b", "c", "d", "e"], "category": ["p", "p", "q", "q", "q"],
                         "size_bin": ["small", "small", "small", "large", "large"]})
    df = pd.DataFrame([_rec("a", "p", "anomaly", "NG"), _rec("b", "p", "anomaly", "OK"),
                       _rec("c", "q", "anomaly", None), _rec("d", "q", "anomaly", "NG"),
                       _rec("n", "q", "normal", "NG")])  # e は記録なし
    t = recall_by_bin(df, bins, ("small", "medium", "large")).set_index(["category", "size_bin"])
    assert t.loc[("all", "small")][["n_anomaly", "n_valid", "tp", "fn", "n_parse_fail", "n_missing"]].tolist() == [3, 2, 1, 1, 1, 0]
    assert t.loc[("all", "small"), "recall"] == 0.5
    assert t.loc[("all", "large")][["n_anomaly", "tp", "n_missing"]].tolist() == [2, 1, 1]
    assert t.loc[("all", "large"), "recall"] == 1.0
    assert t.loc[("all", "medium"), "n_anomaly"] == 0 and pd.isna(t.loc[("all", "medium"), "recall"])
    assert t.loc[("p", "small"), "recall"] == 0.5 and pd.isna(t.loc[("q", "small"), "recall"])


def test_recall_by_bin_rejects_bad_input():
    """CI-M11: 区分の無い不良品、知らない区分、2回以上の繰り返しは ValueError。"""
    from vlmlab.metrics import recall_by_bin
    bins = pd.DataFrame({"image_id": ["a"], "category": ["p"], "size_bin": ["small"]})
    with pytest.raises(ValueError, match="without a bin"):
        recall_by_bin(pd.DataFrame([_rec("z", "p", "anomaly", "NG")]), bins, ("small",))
    with pytest.raises(ValueError, match="unknown bins"):
        recall_by_bin(pd.DataFrame([_rec("a", "p", "anomaly", "NG")]), bins, ("large",))
    with pytest.raises(ValueError, match="single repeat"):
        recall_by_bin(pd.DataFrame([_rec("a", "p", "anomaly", "NG", 1), _rec("a", "p", "anomaly", "NG", 2)]),
                      bins, ("small",))
