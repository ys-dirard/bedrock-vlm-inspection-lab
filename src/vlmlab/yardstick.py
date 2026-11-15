"""評価の物差し：11月の記録を、正答率〜見逃し率と「見逃しを決めたときの良品の通過率」で並べ直す（12/1 の記事）。

docs/contracts.md の CI-Y1〜Y6。API は呼ばない。

並べ方
  conf        : D1 のスコア（NG なら確信度、OK なら 1 − 確信度）。繰り返しのある実験は回ごとに出して平均（D2）
  conf_imgmean: 画像ごとに、有効な回の D1 スコアを平均したもの（CI-K12 の「Nova だけ」と同じ）
  ngcount     : 3回の判定で NG と言った回数（0〜3）。同点は NG 側（CI-Y2）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cascade as C
from . import metrics as M
from .report import TARGET_MISS_RATES

POINT_KEYS = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "miss_rate"]
ORDERINGS = ("conf", "conf_imgmean", "ngcount")
_miss = M._is_missing


def _pass_rows(condition: str, ordering: str, unit: str, results: list[dict[str, Any]],
               n_excluded: int = 0, note: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rows.append({"condition": condition, "ordering": ordering, "unit": unit, "n_images_excluded": n_excluded,
                     "note": note, **r})
    return rows


def _pass(scores: Any, labels: Any, target: float) -> dict[str, Any]:
    """CI-Y2 の通過率に、anomalib の PGn と同じ計算の参考値（CI-Y7）を添える。"""
    scores, labels = list(scores), list(labels)
    return {**M.pass_rate_at_miss_scores(scores, labels, target), **M.pgn_quantile(scores, labels, target)}


def _d1_scores(g: pd.DataFrame) -> tuple[list[float], list[str]]:
    """1回分の有効な行の D1 スコアと label。確信度の無い行があれば ValueError（CI-M7 の代わりに止める）。"""
    v = g[~g["pred"].map(_miss)]
    if v["confidence"].map(_miss).any():
        raise ValueError("confidence missing in some valid responses")
    return [M.anomaly_score(p, float(c)) for p, c in zip(v["pred"], v["confidence"])], v["label"].tolist()


def _image_mean_scores(df: pd.DataFrame) -> pd.DataFrame:
    """画像ごとの D1 スコアの平均（有効で確信度のある回だけ）。1回も無い画像は除く。"""
    rows = []
    for image_id, g in df.groupby("image_id", sort=True):
        sc = [M.anomaly_score(p, float(c)) for p, c in zip(g["pred"], g["confidence"])
              if not _miss(p) and not _miss(c)]
        if sc:
            # 平均の浮動小数点の誤差で同じ値が別の値にならないよう、CI-M6 と同じく小数9桁に丸める
            rows.append({"image_id": image_id, "label": g["label"].iloc[0],
                         "score": round(float(np.mean(sc)), M.SCORE_DIGITS)})
    return pd.DataFrame(rows).set_index("image_id")


def repeated_condition(name: str, df: pd.DataFrame) -> tuple[dict[str, Any], list[dict], list[pd.DataFrame]]:
    """11/11 の条件（3回）。点の指標は D2、通過率は3通り（CI-Y6）。"""
    per = M.per_repeat(df)
    agg = M.aggregate_repeats(per, keys=POINT_KEYS + ["parse_fail_rate"])
    row: dict[str, Any] = {
        "condition": name, "unit": "回ごとの平均（D2）", "n_images": int(df["image_id"].nunique()),
        "n_repeats": agg["n_repeats"], "n_calls": agg["n_total_sum"], "n_valid": agg["n_valid_sum"],
        "n_parse_fail": agg["n_parse_fail_sum"],
    }
    for k in POINT_KEYS:
        row[k] = agg[f"{k}_mean"]
        row[f"{k}_min"] = agg[f"{k}_min"]
        row[f"{k}_max"] = agg[f"{k}_max"]

    detail: list[dict] = []
    ops: list[pd.DataFrame] = []
    # 並べ方1：回ごと → 平均・最小・最大
    for target in TARGET_MISS_RATES:
        per_t = [_pass(*_d1_scores(g), target) for _, g in df.groupby("repeat", sort=True)]
        vals = [p["pass_rate"] for p in per_t if p["pass_rate"] is not None]
        key = f"pass_{int(round(target * 100))}pct"
        row[f"{key}_conf"] = float(np.mean(vals)) if vals else None
        row[f"{key}_conf_min"] = min(vals) if vals else None
        row[f"{key}_conf_max"] = max(vals) if vals else None
        for rep, p in zip(sorted(df["repeat"].unique()), per_t):
            detail += _pass_rows(name, "conf", f"repeat {rep}", [p])
    # 並べ方1'：画像ごとの平均スコア
    im = _image_mean_scores(df)
    res = [_pass(im["score"], im["label"], t) for t in TARGET_MISS_RATES]
    n_ex = int(df["image_id"].nunique() - len(im))
    detail += _pass_rows(name, "conf_imgmean", "1枚1スコア", res, n_ex)
    for t, r in zip(TARGET_MISS_RATES, res):
        row[f"pass_{int(round(t * 100))}pct_conf_imgmean"] = r["pass_rate"]
    ops.append(M.operating_points(im["score"], im["label"]).assign(condition=name, ordering="conf_imgmean"))
    # 並べ方2：NG と言った回数
    nc, excluded = M.ng_counts(df, n_repeats=C.NOVA_REPEATS)
    res = [_pass(nc["ng_count"], nc["label"], t) for t in TARGET_MISS_RATES]
    detail += _pass_rows(name, "ngcount", "1枚1スコア", res, len(excluded),
                         note=("除いた画像: " + ", ".join(excluded)) if excluded else None)
    for t, r in zip(TARGET_MISS_RATES, res):
        row[f"pass_{int(round(t * 100))}pct_ngcount"] = r["pass_rate"]
    row["ngcount_n_images_excluded"] = len(excluded)
    row["ngcount_distribution"] = _ng_distribution(nc)
    ops.append(M.operating_points(nc["ng_count"], nc["label"]).assign(condition=name, ordering="ngcount"))
    return row, detail, ops


def _ng_distribution(nc: pd.DataFrame) -> str:
    """「不良品 0回:n 1回:n …／良品 …」の文字列。"""
    parts = []
    for lab, jp in (("anomaly", "不良品"), ("normal", "良品")):
        vc = nc.loc[nc["label"] == lab, "ng_count"].value_counts()
        parts.append(jp + " " + " ".join(f"{k}回:{int(vc.get(k, 0))}" for k in range(C.NOVA_REPEATS + 1)))
    return " / ".join(parts)


def cascade_conditions(nova: pd.DataFrame, claude: pd.DataFrame, emb: pd.DataFrame,
                       names: list[str] | None = None) -> tuple[list[dict], list[dict], list[pd.DataFrame], str]:
    """11/26 の段階判定の行（1枚1判定。CI-K6・K12、CI-Y6）。names を省くと Nova だけ・Claude だけ・F1 最大の段階判定。"""
    res = C.run_cascade(nova, claude, emb)
    t, rules, results, best = res["table"], res["rules"], res["results"].set_index("rule"), res["best"]
    names = names or ["Nova だけ", "Claude だけ", best]
    nc, excluded = M.ng_counts(nova, n_repeats=C.NOVA_REPEATS)
    rows, detail, ops = [], [], []
    for name in names:
        r = results.loc[name]
        n_valid = int(r["n_valid"])
        row: dict[str, Any] = {
            "condition": f"11/26 {name}", "unit": "1枚1判定", "n_images": int(r["n_images"]), "n_repeats": None,
            "n_calls": None, "n_valid": n_valid, "n_parse_fail": int(r["n_parse_fail"]),
            "accuracy": (int(r["tp"]) + int(r["tn"])) / n_valid if n_valid else None,
            **{k: (None if _miss(r[k]) else float(r[k])) for k in POINT_KEYS if k != "accuracy"},
            "escalation_rate": float(r["escalation_rate"]),
            "cost_per_1000_usd": None if _miss(r["cost_per_1000_usd"]) else float(r["cost_per_1000_usd"]),
        }
        s = C.final_scores(t, rules[name])
        ok = s.map(lambda v: not _miss(v))
        pr = [_pass(s[ok], t.loc[ok, "label"], tg) for tg in TARGET_MISS_RATES]
        detail += _pass_rows(row["condition"], "conf", "1枚1スコア（CI-K12）", pr, int((~ok).sum()))
        for tg, p in zip(TARGET_MISS_RATES, pr):
            row[f"pass_{int(round(tg * 100))}pct_conf"] = p["pass_rate"]
        ops.append(M.operating_points(s[ok], t.loc[ok, "label"]).assign(condition=row["condition"], ordering="conf"))
        if name == "Nova だけ":
            pr2 = [_pass(nc["ng_count"], nc["label"], tg) for tg in TARGET_MISS_RATES]
            detail += _pass_rows(row["condition"], "ngcount", "1枚1スコア", pr2, len(excluded))
            for tg, p in zip(TARGET_MISS_RATES, pr2):
                row[f"pass_{int(round(tg * 100))}pct_ngcount"] = p["pass_rate"]
            row["ngcount_n_images_excluded"] = len(excluded)
            row["ngcount_distribution"] = _ng_distribution(nc)
        else:
            reason = "Claude は1枚1回しか呼んでいないので、NG の回数で並べられない"
            detail += _pass_rows(row["condition"], "ngcount", "-", [
                {"target_miss_rate": tg, "threshold": None, "pass_rate": None, "achieved_miss_rate": None,
                 "n_normal": None, "n_anomaly": None, "n_normal_passed": None, "reason": reason}
                for tg in TARGET_MISS_RATES])
            for tg in TARGET_MISS_RATES:
                row[f"pass_{int(round(tg * 100))}pct_ngcount"] = None
        rows.append(row)
    return rows, detail, ops, best


TABLE_COLS = [
    "condition", "unit", "n_images", "n_repeats", "n_calls", "n_valid", "n_parse_fail",
    "accuracy", "accuracy_min", "accuracy_max", "precision", "recall", "f1", "f1_min", "f1_max",
    "false_positive_rate", "miss_rate",
    "pass_2pct_conf", "pass_5pct_conf", "pass_2pct_conf_min", "pass_2pct_conf_max",
    "pass_5pct_conf_min", "pass_5pct_conf_max",
    "pass_2pct_conf_imgmean", "pass_5pct_conf_imgmean",
    "pass_2pct_ngcount", "pass_5pct_ngcount", "ngcount_n_images_excluded", "ngcount_distribution",
    "escalation_rate", "cost_per_1000_usd",
    "precision_min", "precision_max", "recall_min", "recall_max",
    "false_positive_rate_min", "false_positive_rate_max", "miss_rate_min", "miss_rate_max",
]


def build(repeated: dict[str, pd.DataFrame], nova: pd.DataFrame, claude: pd.DataFrame,
          emb: pd.DataFrame) -> dict[str, Any]:
    rows, detail, ops = [], [], []
    for name, df in repeated.items():
        r, d, o = repeated_condition(name, df)
        rows.append(r)
        detail += d
        ops += o
    r, d, o, best = cascade_conditions(nova, claude, emb)
    rows += r
    detail += d
    ops += o
    table = pd.DataFrame(rows).reindex(columns=TABLE_COLS)
    return {"table": table, "detail": pd.DataFrame(detail), "operating_points": pd.concat(ops, ignore_index=True),
            "best": best}


def write_outputs(res: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = {"table": out / "yardstick_table.csv", "detail": out / "yardstick_pass_rate_detail.csv",
         "operating_points": out / "yardstick_operating_points.csv"}
    res["table"].to_csv(p["table"], index=False, encoding="utf-8", lineterminator="\n")
    res["detail"].to_csv(p["detail"], index=False, encoding="utf-8", lineterminator="\n")
    op = res["operating_points"][["condition", "ordering", "threshold", "miss_rate", "pass_rate"]]
    op.to_csv(p["operating_points"], index=False, encoding="utf-8", lineterminator="\n")
    return p
