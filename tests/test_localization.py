"""位置の評価（localization.py・locreport.py）と、2×2 分割の実行（run.py の tiles）。docs/contracts.md の CI-L1〜L13。"""

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmlab import localization as L
from vlmlab.bedrock import ModelResponse
from vlmlab.cost import Price, PricingTable
from vlmlab.data import load_tile
from vlmlab.locreport import build_loc_report, repeat_metrics
from vlmlab.parse import ParseSpec, parse_response
from vlmlab.run import RunConfig, execute, plan_run


# ---------------------------------------------------------------- 読み取り（CI-L1〜L3）

def test_parse_boxes_reads_guide_format():
    """CI-L1：ユーザーガイドの例の形（カンマ区切り、末尾の「.」）を出た順に読む"""
    bp = L.parse_boxes("[356, 770, 393, 872],\n[626, 770, 659, 878].")
    assert bp.error is None
    assert bp.boxes == [(356, 770, 393, 872), (626, 770, 659, 878)]


def test_parse_boxes_accepts_decimals_and_bounds():
    """CI-L1：小数を読み、0 と 1000 ちょうどは範囲内"""
    assert L.parse_boxes("[0, 0.5, 1000, 999.5]").boxes == [(0, 0.5, 1000, 999.5)]


@pytest.mark.parametrize("text, reason", [
    ("[1, 2, 3]", "not 4 numbers"),
    ("[1, 2, 3, 4, 5]", "not 4 numbers"),
    ("[x1, y1, x2, y2]", "not 4 numbers"),
    ("[-1, 2, 30, 40]", "not 4 numbers"),  # 符号は認めない
    ("[10, 20, 1001, 40]", "out of 0..1000"),
    ("[50, 20, 40, 60]", "x1<x2"),  # 左右が逆
    ("[10, 60, 40, 60]", "x1<x2"),  # 高さ 0
    ("[10, 20, 30, 40] and [1, 2]", "group 1"),  # 1つでも壊れていれば全体を失敗にする
])
def test_parse_boxes_rejects_bad_groups(text, reason):
    """CI-L1：救済しない"""
    bp = L.parse_boxes(text)
    assert bp.boxes is None and reason in bp.error


def test_parse_boxes_none_and_empty():
    """CI-L2：NONE だけなら枠なし。組も NONE も無い、両方ある、空は失敗"""
    assert L.parse_boxes("NONE").boxes == []
    assert L.parse_boxes("There is no defect. NONE.").boxes == []
    assert L.parse_boxes("none").boxes is None  # 大文字だけ
    assert L.parse_boxes("NONEXISTENT").boxes is None  # 単語境界
    assert "both" in L.parse_boxes("[1, 2, 3, 4] NONE").error
    assert L.parse_boxes("No defects found.").boxes is None
    assert L.parse_boxes("").error.startswith("empty")
    assert L.parse_boxes(None).error.startswith("empty")


def test_parse_boxes_truncated_is_failure():
    """CI-L2b：stopReason が max_tokens なら、閉じた [...] があっても解析失敗"""
    text = "[10, 20, 30, 40]\n1. Scratches: [50, 60, 70, 80]\n2. Cracks: [90,"
    assert L.parse_boxes(text).boxes == [(10, 20, 30, 40), (50, 60, 70, 80)]  # stop_reason なしでは読める
    bp = L.parse_boxes(text, "max_tokens")
    assert bp.boxes is None and bp.error == L.TRUNCATED
    assert L.parse_boxes("NONE", "max_tokens").boxes is None
    assert L.parse_boxes("[10, 20, 30, 40]", "end_turn").boxes == [(10, 20, 30, 40)]
    r = parse_response(text, ParseSpec.from_dict({"mode": "boxes"}), "max_tokens")
    assert r.pred is None and r.parse_error == L.TRUNCATED


def test_parse_boxes_first_line():
    """CI-L15：参考の読み方は最初の空でない行だけを読む。切れた応答は失敗のまま"""
    f = L.parse_boxes_first_line
    assert f("[0, 0, 1000, 1000]\n\n NONE").boxes == [(0, 0, 1000, 1000)]  # 主の規則では失敗
    assert L.parse_boxes("[0, 0, 1000, 1000]\n\n NONE").boxes is None
    assert f("\n  NONE\nThe image shows [1, 2, 3, 4]").boxes == []
    assert f("[10, 20, 30, 40], [50, 60, 70, 80].\nExplanation: [x1, y1, x2, y2]").boxes == [
        (10, 20, 30, 40), (50, 60, 70, 80)]
    assert f("[0, 0, 0, 0]\nNONE").boxes is None  # 幅 0 は最初の行でも失敗
    assert f("The image shows no defect.\nNONE").boxes is None  # 最初の行に枠も NONE も無い
    assert f("[10, 20, 30, 40]", "max_tokens").error == L.TRUNCATED
    assert f(None).boxes is None and f("   ").boxes is None


def test_error_kind():
    assert L.error_kind(None) is None
    assert L.error_kind(L.parse_boxes("x", "max_tokens").error) == "truncated"
    assert L.error_kind(L.parse_boxes("[1, 2, 3, 4] NONE").error) == "both_none_and_boxes"
    assert L.error_kind(L.parse_boxes("").error) == "empty"
    assert L.error_kind(L.parse_boxes("[1, 2]").error) == "other"


def test_area_fraction():
    """CI-L14：0〜1000 の枠が領域に占める面積の割合"""
    assert L.area_fraction((0, 0, 1000, 1000)) == 1.0
    assert L.area_fraction((0, 0, 500, 1000)) == 0.5
    assert L.area_fraction((100, 100, 200, 300)) == pytest.approx(0.02)


def test_parse_response_boxes_mode():
    """CI-L3：parsed に枠、pred は枠があれば NG"""
    spec = ParseSpec.from_dict({"mode": "boxes"})
    r = parse_response("[10, 20, 30, 40]", spec)
    assert r.parsed == {"boxes": [[10, 20, 30, 40]]} and r.pred == "NG" and r.confidence is None
    r = parse_response("NONE", spec)
    assert r.parsed == {"boxes": []} and r.pred == "OK"
    r = parse_response("oops", spec)
    assert r.pred is None and r.parse_error


# ---------------------------------------------------------------- 座標（CI-L4、L5）

def test_norm_to_px_whole_image():
    """CI-L4：0〜1000 → 画素（幅と高さで別に掛ける）"""
    assert L.norm_to_px((0, 0, 1000, 1000), (0, 0, 1404, 1070)) == (0, 0, 1404, 1070)
    assert L.norm_to_px((500, 250, 750, 1000), (0, 0, 1000, 400)) == (500, 100, 750, 400)


def test_tile_regions_cover_odd_sizes_exactly_once():
    """CI-L5：奇数の幅・高さでも、4枚ですべての画素をちょうど1回覆う。番号は左上・右上・左下・右下"""
    regs = L.tile_regions(1405, 1071)
    assert regs == [(0, 0, 702, 535), (702, 0, 1405, 535), (0, 535, 702, 1071), (702, 535, 1405, 1071)]
    cover = np.zeros((1071, 1405), dtype=int)
    for x0, y0, x1, y1 in regs:
        cover[y0:y1, x0:x1] += 1
    assert (cover == 1).all()


def test_tile_box_maps_back_to_original():
    """CI-L4・L5：タイル内の 0〜1000 の枠を、元の画像の画素の座標に戻す"""
    regs = L.tile_regions(1404, 1070)  # 各タイル 702×535
    # 右下のタイルの全体 → 右下の4分の1
    assert L.norm_to_px((0, 0, 1000, 1000), regs[3]) == (702, 535, 1404, 1070)
    # 右上のタイルの中央 → 元の画像の (702 + 351, 267.5)
    x1, y1, x2, y2 = L.norm_to_px((400, 400, 600, 600), regs[1])
    assert (x1, y1, x2, y2) == pytest.approx((702 + 280.8, 214.0, 702 + 421.2, 321.0))
    # 左下のタイル
    assert L.norm_to_px((0, 0, 500, 1000), regs[2]) == pytest.approx((0, 535, 351, 1070))


def test_tile_box_matches_object_drawn_on_image(tmp_path):
    """CI-L5：実際に切り出したタイルの中の物体の位置を 0〜1000 で表し、戻すと元の位置になる"""
    W, H = 101, 61
    im = Image.new("RGB", (W, H), (0, 0, 0))
    im.paste((255, 255, 255), (70, 40, 90, 55))  # 右下のタイルにある白い四角
    p = tmp_path / "a.png"
    im.save(p)
    reg = L.tile_regions(W, H)[3]
    tile = load_tile(p, reg)
    with Image.open(io.BytesIO(tile.data)) as t:
        assert t.size == (reg[2] - reg[0], reg[3] - reg[1])
        a = np.asarray(t.convert("L")) > 128
    ys, xs = np.where(a)
    tw, th = t.size
    norm = (xs.min() / tw * 1000, ys.min() / th * 1000, (xs.max() + 1) / tw * 1000, (ys.max() + 1) / th * 1000)
    assert L.norm_to_px(norm, reg) == pytest.approx((70, 40, 90, 55), abs=1.0)  # JPEG の誤差で ±1 画素


def test_tile_errors(tmp_path):
    with pytest.raises(ValueError):
        L.tile_regions(1, 10)
    p = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(p)
    with pytest.raises(ValueError):
        load_tile(p, (5, 5, 11, 10))


# ---------------------------------------------------------------- 正解枠・IoU・当たり（CI-L6〜L9）

def test_gt_boxes_8_connectivity_and_min_area():
    """CI-L6：斜めにつながる画素は1つの成分。外接矩形の面積が 20 未満は除く。上から順"""
    m = np.zeros((40, 40), dtype=bool)
    m[2, 2] = m[3, 3] = m[4, 4] = True  # 斜め → 1成分、外接 3×3 = 9 < 20 → 除く
    m[20:25, 10:14] = True  # 4×5 = 20 → 残る
    m[10:12, 30:40] = True  # 10×2 = 20 → 残る
    boxes = L.gt_boxes(m)
    assert boxes == [(30.0, 10.0, 40.0, 12.0), (10.0, 20.0, 14.0, 25.0)]
    assert len(L.gt_boxes(m, min_box_area=1)) == 3


def test_iou():
    """CI-L7"""
    assert L.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert L.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert L.iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0  # 辺が接するだけ
    assert L.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0  # 和集合 0


def test_hit_and_center():
    """CI-L8：枠が覆う画素（⌊x1⌋〜⌈x2⌉）に異常があれば当たり。中心の画素で中心の当たり"""
    m = np.zeros((10, 10), dtype=bool)
    m[5, 5] = True
    assert L.box_hits_mask((5.9, 5.9, 6.0, 6.0), m)  # 画素 (5,5) にかかる
    assert not L.box_hits_mask((6.0, 6.0, 9.0, 9.0), m)  # x=6 から右は画素 6 から
    assert L.box_hits_mask((0, 0, 5.1, 5.1), m)
    assert not L.box_hits_mask((0, 0, 5.0, 5.0), m)  # 画素 5 は [5, 6) なので 5.0 では覆わない
    assert L.center_in_mask((5, 5, 6, 6), m)  # 中心 (5.5, 5.5)
    assert L.center_in_mask((0, 0, 10, 10), m)  # 中心 (5, 5) → 画素 (5, 5)
    assert not L.center_in_mask((0, 0, 4, 4), m)
    edge = np.zeros((10, 10), dtype=bool)
    edge[9, 9] = True
    assert L.center_in_mask((9, 9, 10, 10), edge)  # 範囲の端は幅−1 に切り詰める
    assert L.box_hits_mask((8, 8, 12, 12), edge)  # 画像の外は切り詰める


def test_score_anomaly():
    """CI-L9：画像単位の当たり・中心、正解枠ごとの最良 IoU（枠なしなら 0）"""
    m = np.zeros((100, 100), dtype=bool)
    m[10:20, 10:20] = True
    m[60:80, 60:80] = True
    gts = L.gt_boxes(m)
    s = L.score_anomaly([(10, 10, 20, 20), (40, 40, 50, 50)], m, gts)
    assert s.hit and s.center_hit and s.n_pred == 2 and s.n_pred_hit == 1 and s.n_pred_center_hit == 1
    assert s.gt_best_iou == [1.0, 0.0]
    s0 = L.score_anomaly([], m, gts)
    assert not s0.hit and not s0.center_hit and s0.gt_best_iou == [0.0, 0.0]
    # 大きな枠は当たるが中心は外れる
    s2 = L.score_anomaly([(0, 0, 100, 100)], m, gts)
    assert s2.hit and not s2.center_hit


def test_split_gt_count():
    """CI-L12：2つ以上の異なるタイルの枠と重なる正解枠を数える"""
    gts = [(40, 40, 60, 60), (0, 0, 10, 10)]
    preds = [(0, (40, 40, 50, 50)), (1, (50, 40, 60, 50)), (0, (0, 0, 5, 5)), (0, (5, 5, 10, 10))]
    assert L.split_gt_count(gts, preds) == 1


# ---------------------------------------------------------------- ぶれ（CI-L10）

def test_set_iou_and_jitter():
    """CI-L10：対称な最良一致。両方枠なしは平均に入れず数える。片方だけ枠なしは 0。読めない回は使わない"""
    a = [(0, 0, 10, 10)]
    b = [(0, 0, 10, 10), (50, 50, 60, 60)]
    assert L.set_iou(a, b) == pytest.approx((1.0 + (1.0 + 0.0) / 2) / 2)
    assert L.set_iou([], []) is None
    assert L.set_iou(a, []) == 0
    j = L.jitter([a, a, None])
    assert j.n_pairs == 1 and j.pair_iou_mean == 1.0
    j = L.jitter([[], [], a])
    assert j.n_pairs == 3 and j.n_pairs_both_empty == 1 and j.pair_iou_mean == 0.0
    assert L.jitter([[], [], []]).pair_iou_mean is None


# ---------------------------------------------------------------- 集計（CI-L11）

def test_repeat_metrics_denominators():
    """CI-L11：解析失敗・欠けは分母から除く。分母 0 は None"""
    none = {"n_pred": None, "hit": None, "center_hit": None, "n_pred_hit": None, "n_pred_center_hit": None,
            "gt_best_iou": None, "n_split_gt": None, "hit_excl_large": None, "box_area_fractions": None,
            "n_calls_truncated": 0, "n_calls_both_none_and_boxes": 0, "n_calls_other_fail": 0}
    img = pd.DataFrame([
        {**none, "label": "anomaly", "status": "ok", "n_pred": 1, "hit": True, "center_hit": False, "n_pred_hit": 1,
         "n_pred_center_hit": 0, "gt_best_iou": json.dumps([0.6, 0.05]), "hit_excl_large": False,
         "box_area_fractions": json.dumps([0.8])},
        {**none, "label": "anomaly", "status": "ok", "n_pred": 0, "hit": False, "center_hit": False, "n_pred_hit": 0,
         "n_pred_center_hit": 0, "gt_best_iou": json.dumps([0.0]), "hit_excl_large": False,
         "box_area_fractions": json.dumps([])},
        {**none, "label": "anomaly", "status": "parse_fail", "n_calls_truncated": 1},
        {**none, "label": "anomaly", "status": "incomplete"},
    ])
    m = repeat_metrics(img)
    assert m["n_anomaly_valid"] == 2 and m["n_parse_fail"] == 1 and m["n_incomplete"] == 1
    assert m["parse_fail_rate"] == pytest.approx(1 / 3)  # 欠けは分母に入れない
    assert m["hit_rate"] == 0.5 and m["center_hit_rate"] == 0.0 and m["anomaly_no_box_rate"] == 0.5
    assert m["n_gt"] == 3 and m["iou_mean"] == pytest.approx(0.65 / 3)
    assert m["iou50_rate"] == pytest.approx(1 / 3) and m["gt_recall_iou10"] == pytest.approx(1 / 3)
    assert m["box_hit_precision"] == 1.0
    assert m["false_positive_rate"] is None  # 良品なし
    assert m["n_split_gt"] is None
    # CI-L14：大きな枠（面積の割合 0.8）だけで当たった画像は、大きな枠を除くと外れ
    assert m["hit_rate_excl_large"] == 0.0
    assert m["box_area_fraction"] == pytest.approx(0.8) and m["large_box_share"] == 1.0
    assert m["n_calls_truncated"] == 1


# ---------------------------------------------------------------- 通し（run.py の tiles と locreport）

class BoxModel:
    """検査対象の画像（最後の画像）の白い画素の外接矩形を 0〜1000 で返す。白が無ければ NONE。"""

    def __init__(self, broken_calls=(), truncated_calls=(), full_box=False):
        self.requests = []
        self.broken = set(broken_calls)
        self.truncated = set(truncated_calls)
        self.full_box = full_box

    def invoke(self, req):
        self.requests.append(req)
        if len(self.requests) in self.broken:
            return ModelResponse(raw_text="I cannot tell.", input_tokens=500, output_tokens=5, latency_s=0.5)
        if len(self.requests) in self.truncated:
            return ModelResponse(raw_text="[0, 0, 10, 10] and then", input_tokens=500, output_tokens=600,
                                 latency_s=0.5, stop_reason="max_tokens")
        if self.full_box:  # 位置を答えず、画像全体の枠だけを返す
            return ModelResponse(raw_text="[0, 0, 1000, 1000]", input_tokens=500, output_tokens=20, latency_s=1.0)
        with Image.open(io.BytesIO(req.images[-1].data)) as im:
            a = np.asarray(im.convert("L")) > 128
            w, h = im.size
        if not a.any():
            text = "NONE"
        else:
            ys, xs = np.where(a)
            b = [xs.min() / w * 1000, ys.min() / h * 1000, (xs.max() + 1) / w * 1000, (ys.max() + 1) / h * 1000]
            text = "[" + ", ".join(f"{v:.0f}" for v in b) + "]"
        return ModelResponse(raw_text=text, input_tokens=500, output_tokens=20, latency_s=1.0)


@pytest.fixture
def loclab(tmp_path):
    visa = tmp_path / "visa"
    rows = []
    for label, folder in (("anomaly", "Anomaly"), ("normal", "Normal")):
        for i in range(2):
            img_dir = visa / "pcb1" / "Data" / "Images" / folder
            img_dir.mkdir(parents=True, exist_ok=True)
            im = Image.new("RGB", (201, 101), (0, 0, 0))
            mask_rel = None
            if label == "anomaly":
                # 0: 右下のタイルの中だけ、1: 縦の境目（x=100）をまたぐ
                rect = (150, 60, 180, 90) if i == 0 else (80, 10, 120, 40)
                im.paste((255, 255, 255), rect)
                m = Image.new("L", (201, 101), 0)
                m.paste(1, rect)
                mdir = visa / "pcb1" / "Data" / "Masks" / folder
                mdir.mkdir(parents=True, exist_ok=True)
                m.save(mdir / f"{i:03d}.png")
                mask_rel = f"pcb1/Data/Masks/{folder}/{i:03d}.png"
            im.save(img_dir / f"{i:03d}.png")
            rows.append({"image_id": f"pcb1/{label}/{i:03d}", "category": "pcb1", "label": label, "split": "test",
                         "path": f"pcb1/Data/Images/{folder}/{i:03d}.png", "mask": mask_rel})
    (tmp_path / "data").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "loc.csv", index=False)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "loc.txt").write_text("find defects", encoding="utf-8")
    base_cfg = {"split": "data/loc.csv", "model_id": "m", "region": "ap-northeast-1", "prompt": "prompts/loc.txt",
                "repeats": 2, "budget_usd": 1.0, "parse": {"mode": "boxes"}}
    for name, extra in (("A", {}), ("C", {"tiles": 2}), ("Aanom", {"labels": ["anomaly"]})):
        (tmp_path / f"{name}.yaml").write_text(json.dumps({"name": name, **base_cfg, **extra}), encoding="utf-8")
    pricing = PricingTable([Price("m", "standard", 0.4, 3.3)])
    return tmp_path, visa, pricing


def test_labels_filter(loclab):
    base, visa, pricing = loclab
    plan = plan_run(RunConfig.load(base / "Aanom.yaml"), base, base / "runs")
    assert set(plan.items.label) == {"anomaly"}


def test_tiles_config_rejects_refs(loclab, tmp_path):
    base, _, _ = loclab
    d = json.loads((base / "C.yaml").read_text(encoding="utf-8"))
    d.update(refs="data/loc.csv", n_refs=1)
    (base / "bad.yaml").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="tiles"):
        RunConfig.load(base / "bad.yaml")


def test_tiles_run_records_and_resume(loclab):
    """CI-L5・L12：画像ごとに4回呼び、タイルの番号と範囲を記録する。再開ではタイル単位で飛ばす"""
    base, visa, pricing = loclab
    plan = plan_run(RunConfig.load(base / "C.yaml"), base, base / "runs")
    model = BoxModel()
    s = execute(plan, model, pricing, visa, base, log=lambda _: None)
    assert s["called"] == 4 * 4 * 2
    recs = [json.loads(l) for l in (plan.out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["tile"] for r in recs[:4]] == [0, 1, 2, 3]
    assert recs[3]["tile_region"] == [100, 50, 201, 101]
    assert recs[0]["tiles"] == 2
    assert all(len(r.images) == 1 for r in model.requests)
    s2 = execute(plan, BoxModel(), pricing, visa, base, log=lambda _: None)
    assert s2["called"] == 0 and s2["skipped"] == 32


def test_locreport_end_to_end(loclab):
    """CI-L9〜L12：A と C を通しで集計。白い四角を正しく返すモデルなら当たり率 1、IoU ほぼ 1、誤検知 0。

    C では境目をまたぐ欠陥が2つのタイルの枠に割れ、IoU は分割前の正解枠と比べるので下がる。
    """
    base, visa, pricing = loclab
    for name in ("A", "C"):
        plan = plan_run(RunConfig.load(base / f"{name}.yaml"), base, base / "runs")
        # C の2回目の最初の画像の、2枚目のタイルだけ読めない応答にする
        broken = {4 * 4 + 2} if name == "C" else set()
        execute(plan, BoxModel(broken), pricing, visa, base, log=lambda _: None)
    paths = build_loc_report([base / "runs" / "A", base / "runs" / "C"], base / "out", visa, base)
    s = pd.read_csv(paths["summary"]).set_index("run_name")
    assert s.loc["A", "hit_rate_mean"] == 1.0 and s.loc["A", "center_hit_rate_mean"] == 1.0
    assert s.loc["A", "iou_mean_mean"] == pytest.approx(1.0, abs=0.02)
    assert s.loc["A", "false_positive_rate_mean"] == 0.0
    assert s.loc["A", "parse_fail_rate_mean"] == 0.0
    assert s.loc["A", "cost_per_image_usd"] == pytest.approx((500 * 0.4 + 20 * 3.3) / 1e6)
    # C：1回目は全部有効、2回目は1枚が解析失敗
    assert s.loc["C", "n_parse_fail_sum"] == 1
    assert s.loc["C", "n_anomaly_valid_sum"] == 3
    assert s.loc["C", "hit_rate_mean"] == 1.0
    # 費用は4回の合計を1枚の値にする（読めなかったタイルも課金されたので含める）。4枚×2回 = 8 枚・回
    assert s.loc["C", "cost_per_image_usd"] == pytest.approx((31 * (500 * 0.4 + 20 * 3.3) + (500 * 0.4 + 5 * 3.3)) / 8 / 1e6)
    assert s.loc["C", "input_tokens_per_image_mean"] == 2000
    per = pd.read_csv(paths["per_image"])
    c = per[(per.run_name == "C") & (per.status == "ok") & (per.label == "anomaly")].set_index(["image_id", "repeat"])
    # 右下のタイルの中だけの欠陥：元の位置に戻る
    assert np.allclose(json.loads(c.loc[("pcb1/anomaly/000", 1), "pred_boxes_px"]), [[150, 60, 180, 90]], atol=1)
    assert json.loads(c.loc[("pcb1/anomaly/000", 1), "gt_best_iou"])[0] == pytest.approx(1.0, abs=0.05)
    # 境目をまたぐ欠陥：左上と右上のタイルで2つの枠に割れる
    assert c.loc[("pcb1/anomaly/001", 1), "n_split_gt"] == 1
    assert np.allclose(json.loads(c.loc[("pcb1/anomaly/001", 1), "pred_boxes_px"]),
                       [[80, 10, 100, 40], [100, 10, 120, 40]], atol=1)
    assert json.loads(c.loc[("pcb1/anomaly/001", 1), "gt_best_iou"])[0] == pytest.approx(0.5, abs=0.05)
    j = pd.read_csv(paths["jitter"]).set_index(["run_name", "label"])
    assert j.loc[("A", "anomaly"), "pair_iou_mean"] == pytest.approx(1.0)
    assert j.loc[("A", "normal"), "n_pairs_both_empty"] == 2
    assert j.loc[("A", "anomaly"), "identical_text_rate"] == 1.0


def test_locreport_large_boxes_and_truncation(loclab):
    """CI-L14・L2b：画像全体の枠だけなら当たり率は 1 だが、大きな枠を除くと 0。切れた応答は解析失敗として数える"""
    base, visa, pricing = loclab
    plan = plan_run(RunConfig.load(base / "A.yaml"), base, base / "runs")
    model = BoxModel(full_box=True)
    model.truncated = {1}
    execute(plan, model, pricing, visa, base, log=lambda _: None)
    recs = [json.loads(l) for l in (plan.out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs[0]["parse_error"] == L.TRUNCATED  # 記録の時点でも解析失敗
    paths = build_loc_report([base / "runs" / "A"], base / "out", visa, base)
    s = pd.read_csv(paths["summary"]).iloc[0]
    assert s["hit_rate_mean"] == 1.0 and s["hit_rate_excl_large_mean"] == 0.0
    assert s["box_area_fraction_mean"] == 1.0 and s["large_box_share_mean"] == 1.0
    assert s["n_calls_truncated_sum"] == 1 and s["n_parse_fail_sum"] == 1
    assert s["false_positive_rate_mean"] == 1.0


def test_locreport_firstline_reference_table(loclab):
    """CI-L15：参考の読み方の表は別のファイルに出る。主の規則で失敗した応答も、最初の行が読めれば有効になる"""
    base, visa, pricing = loclab

    class FullThenNone(BoxModel):
        def invoke(self, req):
            r = super().invoke(req)
            if r.raw_text == "NONE":  # 良品：画像全体の枠を書いてから NONE と答える
                r.raw_text = "[0, 0, 1000, 1000]\n\nNONE"
            return r

    plan = plan_run(RunConfig.load(base / "A.yaml"), base, base / "runs")
    execute(plan, FullThenNone(), pricing, visa, base, log=lambda _: None)
    paths = build_loc_report([base / "runs" / "A"], base / "out", visa, base)
    strict = pd.read_csv(paths["summary"]).iloc[0]
    fl = pd.read_csv(paths["summary_firstline"]).iloc[0]
    assert strict["reader"] == "strict" and fl["reader"] == "first_line"
    assert strict["n_normal_valid_sum"] == 0 and strict["n_calls_both_none_and_boxes_sum"] == 4
    assert fl["n_normal_valid_sum"] == 4 and fl["false_positive_rate_mean"] == 1.0
    assert fl["large_box_share_mean"] == pytest.approx(0.5)  # 良品の全体の枠4つと、不良の小さな枠4つ
    assert fl["hit_rate_mean"] == 1.0 and fl["hit_rate_excl_large_mean"] == 1.0
    assert "L-D9" in paths["summary_firstline_md"].read_text(encoding="utf-8")
    p2 = build_loc_report([base / "runs" / "A"], base / "out2", visa, base, reference_first_line=False)
    assert "summary_firstline" not in p2


def test_locreport_rejects_mask_size_mismatch(loclab):
    """CI-L13：マスクと画像の大きさが違えば止める"""
    base, visa, pricing = loclab
    Image.new("L", (10, 10), 0).save(visa / "pcb1" / "Data" / "Masks" / "Anomaly" / "000.png")
    plan = plan_run(RunConfig.load(base / "A.yaml"), base, base / "runs")
    execute(plan, BoxModel(), pricing, visa, base, log=lambda _: None)
    with pytest.raises(ValueError, match="mask size"):
        build_loc_report([base / "runs" / "A"], base / "out", visa, base)
