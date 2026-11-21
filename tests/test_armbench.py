import json

import pandas as pd
import pytest
import yaml

from vlmlab import armbench as ab
from vlmlab.report import load_run


def _pick(i: int) -> str:
    return f"{i:021d}"


def make_tree(root, spec):
    """spec: [(split_dir, defect_type, pick_index, view)]"""
    for split_dir, dtype, i, view in spec:
        d = root / "imagenet" / split_dir / dtype
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{_pick(i)}_{view}.jpg").write_bytes(b"x")


def test_read_pick_ids_keeps_zeros_and_strips_crlf(tmp_path):
    p = tmp_path / "train.csv"
    p.write_bytes(b"002100500165279085665\r\n000000000000000000001\r\n")
    assert ab.read_pick_ids(p) == {"002100500165279085665", "000000000000000000001"}
    p.write_bytes(b"123\r\n")
    with pytest.raises(ValueError):
        ab.read_pick_ids(p)


def test_scan_imagenet_labels_and_split(tmp_path):
    make_tree(tmp_path, [("test", "nominal", 1, 2), ("test", "multi_pick", 1, 1), ("test", "package_defect", 2, 3)])
    t = ab.scan_imagenet(tmp_path, train_ids={_pick(2)}, test_ids={_pick(1)})
    assert list(t.columns) == ab.COLUMNS
    r = t.set_index("image_id")
    assert r.loc[f"nominal/{_pick(1)}_2", "label"] == "normal"
    assert r.loc[f"multi_pick/{_pick(1)}_1", "label"] == "anomaly"
    assert r.loc[f"package_defect/{_pick(2)}_3", "split"] == "train"
    assert r.loc[f"nominal/{_pick(1)}_2", "path"] == f"imagenet/test/nominal/{_pick(1)}_2.jpg"
    with pytest.raises(ValueError):
        ab.scan_imagenet(tmp_path, train_ids={_pick(1)}, test_ids={_pick(1)})


def test_scan_rejects_unknown_folder(tmp_path):
    make_tree(tmp_path, [("test", "scratch", 1, 1)])
    with pytest.raises(ValueError):
        ab.scan_imagenet(tmp_path)


def big_table():
    rows = []
    # 欠陥のピック: multi_pick 0..29（各2視点）、package_defect 100..109（各3視点）。正常のピック 200..259
    # ピック 0..4 は nominal の視点も持つ（欠陥のあるピックの正常の視点）
    for i in range(30):
        rows += [("multi_pick", i, v) for v in (1, 2)]
    for i in range(5):
        rows.append(("nominal", i, 3))
    for i in range(100, 110):
        rows += [("package_defect", i, v) for v in (1, 2, 3)]
    for i in range(200, 260):
        rows += [("nominal", i, v) for v in (1, 2)]
    df = pd.DataFrame([{
        "image_id": f"{d}/{_pick(i)}_{v}", "category": ab.CATEGORY, "label": "normal" if d == "nominal" else "anomaly",
        "split": "test", "path": f"imagenet/test/{d}/{_pick(i)}_{v}.jpg", "defect_type": d, "pick_id": _pick(i), "view": v,
    } for d, i, v in rows], columns=ab.COLUMNS)
    return df


def test_quotas_are_even_and_spill_over():
    assert ab._quotas(10, {"multi_pick": 30, "package_defect": 30}) == {"multi_pick": 5, "package_defect": 5}
    assert ab._quotas(11, {"multi_pick": 30, "package_defect": 30}) == {"multi_pick": 6, "package_defect": 5}
    assert ab._quotas(30, {"multi_pick": 30, "package_defect": 10}) == {"multi_pick": 20, "package_defect": 10}
    with pytest.raises(ValueError):
        ab._quotas(41, {"multi_pick": 30, "package_defect": 10})


def test_sample_one_per_pick_balanced_deterministic():
    t = big_table()
    a = ab.sample(t, n_normal=20, n_defect=30)
    assert a.equals(ab.sample(t, n_normal=20, n_defect=30))
    assert not a["pick_id"].duplicated().any()
    assert a.groupby("defect_type").size().to_dict() == {"multi_pick": 20, "nominal": 20, "package_defect": 10}
    # 正常は欠陥のあるピックから選ばない
    defect_picks = set(t.loc[t.defect_type != "nominal", "pick_id"])
    assert not set(a.loc[a.defect_type == "nominal", "pick_id"]) & defect_picks
    assert not ab.sample(t, 20, 30, seed=1).equals(a)


def test_sample_raises_when_short():
    t = big_table()
    with pytest.raises(ValueError):
        ab.sample(t, n_normal=61, n_defect=10)
    with pytest.raises(ValueError):
        ab.sample(t, n_normal=10, n_defect=41)
    with pytest.raises(ValueError):
        ab.sample(t, n_normal=1, n_defect=1, splits=("train",))


def test_pred_type():
    assert ab.pred_type("multi_pick") == "multi_pick"
    assert ab.pred_type("package_open") == "package_defect"
    assert ab.pred_type("deconstruction") == "package_defect"
    assert ab.pred_type("scratch") == "other"
    assert ab.pred_type("") is None and ab.pred_type(None) is None and ab.pred_type(float("nan")) is None


def _records(split, preds):
    recs = []
    for (_, r), (pred, defect) in zip(split.iterrows(), preds):
        raw = None if pred is None else json.dumps({"judgement": pred, "defect": defect, "confidence": 0.9})
        recs.append({"run_name": "x", "repeat": 1, "image_id": r.image_id, "category": r.category, "label": r.label,
                     "path": r.path, "model_id": "m", "raw_text": raw if raw else "not json", "api_error": None,
                     "cost_usd": 0.001, "input_tokens": 10, "output_tokens": 5, "latency_s": 1.0})
    return recs


def test_by_defect_type_counts(tmp_path):
    split = pd.DataFrame([
        {"image_id": f"{d}/{_pick(i)}_1", "category": ab.CATEGORY, "label": "normal" if d == "nominal" else "anomaly",
         "split": "test", "path": "p", "defect_type": d, "pick_id": _pick(i), "view": 1}
        for d, i in [("nominal", 1), ("nominal", 2), ("multi_pick", 3), ("multi_pick", 4), ("package_defect", 5),
                     ("package_defect", 6)]
    ])
    preds = [("NG", "multi_pick"), ("OK", ""), ("NG", "multi_pick"), ("NG", "package_open"),
             ("NG", "deconstruction"), (None, None)]
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in _records(split, preds)) + "\n", encoding="utf-8")
    (rd / "config.yaml").write_text(yaml.safe_dump({"parse": {}}), encoding="utf-8")
    df = load_run(rd)
    t = ab.by_defect_type(df, split).set_index("defect_type")
    assert t.loc["nominal", "ng_rate"] == 0.5 and t.loc["nominal", "metric"] == "false_positive_rate"
    assert t.loc["multi_pick", "ng_rate"] == 1.0
    assert t.loc["multi_pick", "type_match_rate"] == 0.5
    assert t.loc["package_defect", "n_valid"] == 1 and t.loc["package_defect", "n_parse_fail"] == 1
    assert t.loc["package_defect", "type_match_rate"] == 1.0
    row = ab.summary_row(df)
    assert row["n_parse_fail"] == 1 and row["recall"] == 1.0 and row["false_positive_rate"] == 0.5
    with pytest.raises(ValueError):
        ab.by_defect_type(df, split.iloc[:-1])


JPEG = b"\xff\xd8\xff\xe0" + b"0" * 20
PNG = b"\x89PNG\r\n\x1a\n" + b"1" * 20


def _write_parquet(path, items):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rgb = [{"bytes": b, "path": None} for _, _, b in items]
    meta = [{"id": i, "label": lab, "sublabel": "none", "polygon": [[0, 0]]} for i, lab, _ in items]
    pq.write_table(pa.table({"rgb": rgb, "meta": meta}), path)


def test_export_mirror_writes_imagenet_layout(tmp_path):
    _write_parquet(tmp_path / "a.parquet", [(f"{_pick(1)}_1.jpg", "nominal", JPEG), (f"{_pick(1)}_2.jpg", "multi_pick", JPEG)])
    _write_parquet(tmp_path / "b.parquet", [(f"{_pick(2)}_1", "package_defect", PNG)])  # .jpg の無い id
    meta = ab.export_mirror([tmp_path / "a.parquet", tmp_path / "b.parquet"], tmp_path / "out")
    assert meta[["label", "shard", "format", "bytes"]].values.tolist() == [
        ["nominal", "a.parquet", "jpg", len(JPEG)], ["multi_pick", "a.parquet", "jpg", len(JPEG)],
        ["package_defect", "b.parquet", "png", len(PNG)]]
    assert (tmp_path / "out/imagenet/mirror/multi_pick" / f"{_pick(1)}_2.jpg").read_bytes() == JPEG
    # id は .jpg でも中身が PNG なら、拡張子を .png にして書く（バイト列は変えない）
    assert (tmp_path / "out/imagenet/mirror/package_defect" / f"{_pick(2)}_1.png").read_bytes() == PNG
    t = ab.scan_imagenet(tmp_path / "out", test_ids={_pick(2)})
    assert t.set_index("image_id").loc[f"package_defect/{_pick(2)}_1", "split"] == "test"
    assert len(t) == 3


@pytest.mark.parametrize("items", [
    [(f"{_pick(1)}_1.jpg", "scratch", JPEG)],
    [("abc_1.jpg", "nominal", JPEG)],
    [(f"{_pick(1)}_1.jpg", "nominal", b"GIF89a....")],
    [(f"{_pick(1)}_1.jpg", "nominal", b"")],
    [(f"{_pick(1)}_1.jpg", "nominal", JPEG), (f"{_pick(1)}_1.jpg", "nominal", JPEG)],
])
def test_export_mirror_rejects_bad_rows(tmp_path, items):
    _write_parquet(tmp_path / "a.parquet", items)
    with pytest.raises(ValueError):
        ab.export_mirror([tmp_path / "a.parquet"], tmp_path / "out")
