"""位置の評価の集計（12/2 の記事。docs/contracts.md の CI-L9〜L13）。API は呼ばない。

記録（calls.jsonl）の raw_text から枠を読み直し、マスクと比べる。出すもの:
  <tag>_summary.csv / .md     条件ごとの指標（回ごとに計算して平均・最小・最大）
  <tag>_by_category.csv       品目ごとの指標
  <tag>_per_image.csv         画像×回ごとの採点（例の画像を選ぶのにも使う）
  jitter.csv                  座標のぶれ（条件×正解ごと）
  jitter_per_image.csv        画像ごとのぶれ
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import localization as L
from .data import image_size
from .metrics import cost_stats, latency_stats
from .run import load_jsonl


def load_calls(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """記録を読み、(image_id, repeat, tile) ごとに最後の成功した呼び出しを残す（CI-R3）。"""
    run_dir = Path(run_dir)
    records = load_jsonl(run_dir / "calls.jsonl")
    if not records:
        raise ValueError(f"no records in {run_dir}")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    df = pd.DataFrame(records)
    if "tile" not in df:
        df["tile"] = None
    df["tile_key"] = df["tile"].map(lambda t: -1 if t is None or (isinstance(t, float) and np.isnan(t)) else int(t))
    ok = df[df["api_error"].isna()].drop_duplicates(["image_id", "repeat", "tile_key"], keep="last").copy()
    ok.attrs["n_calls"] = len(df)
    ok.attrs["n_api_error"] = int(df["api_error"].notna().sum())
    ok.attrs["cost_all_calls_usd"] = None if df["cost_usd"].isna().any() else float(df["cost_usd"].sum())
    # すべての呼び出しが API エラーだった画像×回も「欠け」として数えるため、記録にある組をすべて残す
    ok.attrs["all_keys"] = df.drop_duplicates(["image_id", "repeat"])[
        ["run_name", "image_id", "category", "label", "path", "repeat"]].to_dict("records")
    return ok.reset_index(drop=True), cfg


class _Truth:
    """画像の大きさ・マスク・正解枠をまとめて持つ（同じ画像を何度も読まない）。"""

    def __init__(self, visa_root: Path, masks: dict[str, str], min_box_area: int):
        self.visa_root = visa_root
        self.masks = masks
        self.min_box_area = min_box_area
        self._cache: dict[str, tuple[tuple[int, int], np.ndarray | None, list[L.Box]]] = {}

    def get(self, image_id: str, path: str, label: str):
        if image_id not in self._cache:
            w, h = image_size(self.visa_root / path)
            mask, gts = None, []
            if label == "anomaly":
                mp = self.masks.get(image_id)
                if not isinstance(mp, str) or not mp:
                    raise ValueError(f"no mask path for anomaly image {image_id}")  # CI-L13
                mask = L.load_mask(self.visa_root / mp)
                if mask.shape != (h, w):
                    raise ValueError(f"mask size {mask.shape[::-1]} != image size {(w, h)} for {image_id}")
                gts = L.gt_boxes(mask, self.min_box_area)
            self._cache[image_id] = ((w, h), mask, gts)
        return self._cache[image_id]


def score_images(calls: pd.DataFrame, cfg: dict[str, Any], truth: _Truth) -> pd.DataFrame:
    """画像×回ごとに予測をまとめて採点する（CI-L9、L12）。"""
    tiles = cfg.get("tiles")
    expected = (tiles * tiles) if tiles else 1
    rows = []
    for (image_id, repeat), g in calls.groupby(["image_id", "repeat"], sort=True):
        g = g.sort_values("tile_key")
        first = g.iloc[0]
        (w, h), mask, gts = truth.get(image_id, first["path"], first["label"])
        parses = [L.parse_boxes(t) for t in g["raw_text"]]
        row: dict[str, Any] = {
            "run_name": first["run_name"], "image_id": image_id, "category": first["category"],
            "label": first["label"], "repeat": int(repeat), "n_calls": len(g),
            "latency_s": float(g["latency_s"].sum()),
            "input_tokens": int(g["input_tokens"].sum()), "output_tokens": int(g["output_tokens"].sum()),
            "cost_usd": None if g["cost_usd"].isna().any() else float(g["cost_usd"].sum()),
            "raw_text": json.dumps(list(g["raw_text"]), ensure_ascii=False) if tiles else first["raw_text"],
            "n_gt": len(gts),
        }
        errors = [p.error for p in parses if p.boxes is None]
        if len(g) < expected:
            row.update(status="incomplete", parse_error=None)
        elif errors:
            row.update(status="parse_fail", parse_error="; ".join(errors))
        else:
            row.update(status="ok", parse_error=None)
        if row["status"] != "ok":
            rows.append(row)
            continue
        pred: list[tuple[int, L.Box]] = []
        for (_, c), p in zip(g.iterrows(), parses):
            region = tuple(c["tile_region"]) if tiles else (0, 0, w, h)
            pred += [(int(c["tile_key"]), L.norm_to_px(b, region)) for b in p.boxes]  # type: ignore[union-attr]
        boxes = [b for _, b in pred]
        row["n_pred"] = len(boxes)
        row["pred_boxes_px"] = json.dumps([[round(v, 2) for v in b] for b in boxes])
        if first["label"] == "anomaly":
            s = L.score_anomaly(boxes, mask, gts)  # type: ignore[arg-type]
            row.update(hit=s.hit, center_hit=s.center_hit, n_pred_hit=s.n_pred_hit,
                       n_pred_center_hit=s.n_pred_center_hit,
                       gt_best_iou=json.dumps([round(v, 6) for v in s.gt_best_iou]),
                       n_split_gt=L.split_gt_count(gts, pred) if tiles else None)
        rows.append(row)
    seen = {(r["image_id"], r["repeat"]) for r in rows}
    for k in calls.attrs.get("all_keys", []):
        if (k["image_id"], int(k["repeat"])) not in seen:
            rows.append({**{x: k[x] for x in ("run_name", "image_id", "category", "label")},
                         "repeat": int(k["repeat"]), "n_calls": 0, "status": "incomplete", "parse_error": None})
    out = pd.DataFrame(rows)
    for col in SCORE_COLUMNS:
        if col not in out:
            out[col] = None
    return out


SCORE_COLUMNS = ["n_pred", "pred_boxes_px", "hit", "center_hit", "n_pred_hit", "n_pred_center_hit",
                 "gt_best_iou", "n_split_gt", "latency_s", "input_tokens", "output_tokens", "cost_usd",
                 "raw_text", "n_gt"]


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def repeat_metrics(img: pd.DataFrame) -> dict[str, Any]:
    """1回分の画像の採点から指標を出す（CI-L11）。"""
    an = img[img["label"] == "anomaly"]
    no = img[img["label"] == "normal"]
    an_ok = an[an["status"] == "ok"]
    no_ok = no[no["status"] == "ok"]
    ious = [v for s in an_ok.get("gt_best_iou", pd.Series(dtype=object)) for v in json.loads(s)]
    n_pred_an = int(an_ok["n_pred"].sum()) if len(an_ok) else 0
    attempted = img[img["status"] != "incomplete"]
    return {
        "n_anomaly_valid": len(an_ok),
        "n_normal_valid": len(no_ok),
        "n_parse_fail": int((img["status"] == "parse_fail").sum()),
        "n_incomplete": int((img["status"] == "incomplete").sum()),
        "parse_fail_rate": _div(int((attempted["status"] == "parse_fail").sum()), len(attempted)),
        "hit_rate": _div(int(an_ok["hit"].sum()), len(an_ok)) if len(an_ok) else None,
        "center_hit_rate": _div(int(an_ok["center_hit"].sum()), len(an_ok)) if len(an_ok) else None,
        "anomaly_no_box_rate": _div(int((an_ok["n_pred"] == 0).sum()), len(an_ok)),
        "box_hit_precision": _div(int(an_ok["n_pred_hit"].sum()), n_pred_an) if len(an_ok) else None,
        "box_center_precision": _div(int(an_ok["n_pred_center_hit"].sum()), n_pred_an) if len(an_ok) else None,
        "n_gt": len(ious),
        "iou_mean": float(np.mean(ious)) if ious else None,
        "iou50_rate": _div(sum(v >= 0.5 for v in ious), len(ious)),
        "gt_recall_iou10": _div(sum(v >= 0.1 for v in ious), len(ious)),
        "false_positive_rate": _div(int((no_ok["n_pred"] > 0).sum()), len(no_ok)),
        "boxes_per_anomaly": _div(n_pred_an, len(an_ok)),
        "boxes_per_normal": _div(int(no_ok["n_pred"].sum()) if len(no_ok) else 0, len(no_ok)),
        "n_split_gt": (int(an_ok["n_split_gt"].sum()) if "n_split_gt" in an_ok and an_ok["n_split_gt"].notna().any()
                       else None),
    }


MINMAX = ("hit_rate", "center_hit_rate", "iou_mean", "false_positive_rate")


def aggregate(img: pd.DataFrame) -> dict[str, Any]:
    """回ごとに計算して平均（主な指標は最小・最大も）。None の回は平均に入れない（D2）。"""
    per = [repeat_metrics(g) for _, g in img.groupby("repeat", sort=True)]
    out: dict[str, Any] = {"n_repeats": len(per)}
    for k in per[0]:
        vals = [p[k] for p in per if p[k] is not None]
        if k.startswith("n_"):
            out[f"{k}_sum"] = sum(vals) if vals else None
            continue
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        if k in MINMAX:
            out[f"{k}_min"] = min(vals) if vals else None
            out[f"{k}_max"] = max(vals) if vals else None
    called = img[img["status"] != "incomplete"]
    out.update(latency_stats(called))
    out.update(cost_stats(called))
    out["cost_per_image_usd"] = out.pop("cost_per_call_usd")
    for k in ("input_tokens", "output_tokens"):
        out[f"{k}_per_image_mean"] = float(called[k].mean()) if len(called) else None
    return out


def jitter_table(img: pd.DataFrame) -> pd.DataFrame:
    """画像ごとの座標のぶれ（CI-L10）と、全回の本文が同じだったか。"""
    rows = []
    for image_id, g in img.groupby("image_id", sort=True):
        g = g.sort_values("repeat")
        reps = [[tuple(b) for b in json.loads(r["pred_boxes_px"])] if r["status"] == "ok" else None
                for _, r in g.iterrows()]
        j = L.jitter(reps)
        all_ok = all(r is not None for r in reps) and len(reps) >= 2
        rows.append({
            "run_name": g["run_name"].iloc[0], "image_id": image_id, "category": g["category"].iloc[0],
            "label": g["label"].iloc[0], "n_repeats": len(g), "n_pairs": j.n_pairs,
            "n_pairs_both_empty": j.n_pairs_both_empty, "pair_iou_mean": j.pair_iou_mean,
            "identical_text": (g["raw_text"].nunique() == 1) if all_ok else None,
            "n_pred_by_repeat": ";".join("" if r is None else str(len(r)) for r in reps),
        })
    return pd.DataFrame(rows)


def jitter_summary(per_image: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (run, label), g in per_image.groupby(["run_name", "label"], sort=False):
        vals = g["pair_iou_mean"].dropna()
        ident = g["identical_text"].dropna()
        rows.append({
            "run_name": run, "label": label, "n_images": len(g),
            "n_images_with_boxes": len(vals),
            "pair_iou_mean": float(vals.mean()) if len(vals) else None,
            "pair_iou_median": float(vals.median()) if len(vals) else None,
            "n_pairs": int(g["n_pairs"].sum()), "n_pairs_both_empty": int(g["n_pairs_both_empty"].sum()),
            "identical_text_rate": float(ident.astype(bool).mean()) if len(ident) else None,
            "n_images_all_repeats_ok": len(ident),
        })
    return rows


def build_loc_report(run_dirs: list[str | Path], out_dir: str | Path, visa_root: str | Path,
                     base_dir: str | Path = ".", tag: str = "localization",
                     min_box_area: int = 20) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, by_cat, per_img_all, jit_all = [], [], [], []
    for rd in run_dirs:
        calls, cfg = load_calls(rd)
        split = pd.read_csv(Path(base_dir) / cfg["split"])
        masks = dict(zip(split["image_id"], split["mask"])) if "mask" in split else {}
        img = score_images(calls, cfg, _Truth(Path(visa_root), masks, min_box_area))
        name = calls["run_name"].iloc[0]
        summary.append({"run_name": name, "tiles": cfg.get("tiles"), "n_refs": cfg.get("n_refs", 0),
                        "n_images": img["image_id"].nunique(), "n_calls": calls.attrs["n_calls"],
                        "n_api_error": calls.attrs["n_api_error"],
                        "cost_all_calls_usd": calls.attrs["cost_all_calls_usd"], **aggregate(img)})
        for cat, g in img.groupby("category", sort=True):
            by_cat.append({"run_name": name, "category": cat, **aggregate(g)})
        per_img_all.append(img)
        jit_all.append(jitter_table(img))
    paths = {
        "summary": out / f"{tag}_summary.csv",
        "summary_md": out / f"{tag}_summary.md",
        "by_category": out / f"{tag}_by_category.csv",
        "per_image": out / f"{tag}_per_image.csv",
        "jitter": out / "jitter.csv",
        "jitter_per_image": out / "jitter_per_image.csv",
    }
    s = pd.DataFrame(summary)
    s.to_csv(paths["summary"], index=False, lineterminator="\n")
    pd.DataFrame(by_cat).to_csv(paths["by_category"], index=False, lineterminator="\n")
    pd.concat(per_img_all).to_csv(paths["per_image"], index=False, lineterminator="\n")
    jpi = pd.concat(jit_all)
    jpi.to_csv(paths["jitter_per_image"], index=False, lineterminator="\n")
    pd.DataFrame(jitter_summary(jpi)).to_csv(paths["jitter"], index=False, lineterminator="\n")
    paths["summary_md"].write_text(to_markdown(s), encoding="utf-8")
    return paths


def _fmt(v: Any, d: int | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if d is not None and isinstance(v, (int, float)):
        return f"{v:.{d}f}"
    return str(v)


def to_markdown(s: pd.DataFrame) -> str:
    cols = [
        ("run_name", "条件", None),
        ("n_anomaly_valid_sum", "有効な不良（全回）", None),
        ("n_normal_valid_sum", "有効な良品（全回）", None),
        ("parse_fail_rate_mean", "解析失敗率", 3),
        ("hit_rate_mean", "当たり率", 3),
        ("center_hit_rate_mean", "中心の当たり率", 3),
        ("iou_mean_mean", "IoU 平均", 3),
        ("iou50_rate_mean", "IoU≥0.5", 3),
        ("gt_recall_iou10_mean", "再現率(IoU≥0.1)", 3),
        ("false_positive_rate_mean", "誤検知率", 3),
        ("box_hit_precision_mean", "枠単位の当たり", 3),
        ("latency_p50_s", "p50 (秒)", 2),
        ("latency_p95_s", "p95 (秒)", 2),
        ("input_tokens_per_image_mean", "入力トークン/枚", 0),
        ("output_tokens_per_image_mean", "出力トークン/枚", 0),
        ("cost_per_1000_usd", "1,000枚あたり (USD)", 3),
    ]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in s.iterrows():
        lines.append("| " + " | ".join(_fmt(r.get(k), d) for k, _, d in cols) + " |")
    note = ("\n指標は回ごとに計算した平均。解析失敗と欠けた画像は分母から除く（CI-L11、L12）。"
            "条件 C のレイテンシ・トークン・費用は4回の合計。\n")
    return "\n".join(lines) + "\n" + note
