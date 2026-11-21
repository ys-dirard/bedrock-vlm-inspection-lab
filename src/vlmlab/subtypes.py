"""異常の種類（subtype）ごとの集計（12/15 の記事：MVTec LOCO AD の構造的・論理的な異常）。docs/contracts.md の CI-ST1〜ST5。

記録（calls.jsonl）には subtype が無いので、一覧（data/splits/*.csv）と image_id で結び付ける。API は呼ばない。

- 種類ごとの「NG と判定した割合」：異常の種類では再現率、良品（good）では誤検知率にあたる
- 種類ごとの AUROC：その種類の異常と、同じ範囲の良品すべてを比べる（LOCO の論文・EfficientAD の表と同じ数え方）。
  VLM のスコアは「不良らしさ」（NG なら confidence、OK なら 1 − confidence。D1）。手元の手法の異常度にもそのまま使える
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import metrics as M

NORMAL_SUBTYPE = "good"


def attach_subtype(records: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """CI-ST1：image_id で一覧の subtype を付ける。一覧に無い image_id、正解の食い違いがあれば止める。"""
    need = {"image_id", "label", "subtype"}
    if need - set(split.columns):
        raise ValueError(f"split lacks columns {sorted(need - set(split.columns))}")
    if split["image_id"].duplicated().any():
        raise ValueError("split image_id is not unique")
    m = records.merge(split[["image_id", "label", "subtype"]].rename(columns={"label": "_split_label"}),
                      on="image_id", how="left", validate="many_to_one")
    missing = m.loc[m["subtype"].isna(), "image_id"].unique().tolist()
    if missing:
        raise ValueError(f"{len(missing)} image_id not in split, e.g. {missing[:3]}")
    if (m["label"] != m["_split_label"]).any():
        raise ValueError("label in records differs from split")
    bad = m[(m["subtype"] == NORMAL_SUBTYPE) != (m["label"] == "normal")]
    if len(bad):
        raise ValueError(f"subtype '{NORMAL_SUBTYPE}' must be exactly the normal images; e.g. {bad.image_id.iloc[0]}")
    return m.drop(columns="_split_label")


def vlm_score(df: pd.DataFrame) -> pd.Series:
    """VLM の不良らしさ（D1）。解析失敗か confidence が無い行は NaN。"""
    out = []
    for p, c in zip(df["pred"], df["confidence"]):
        out.append(math.nan if M._is_missing(p) or M._is_missing(c) else M.anomaly_score(p, float(c)))
    return pd.Series(out, index=df.index, dtype=float)


def _auroc(pos: pd.Series, neg: pd.Series) -> tuple[float | None, str | None]:
    """CI-ST4：スコアが1つでも欠けていれば計算しない。どちらかが0件なら計算しない。同点は0.5として数える（roc_auc_score）。"""
    if pos.isna().any() or neg.isna().any():
        return None, "score missing in some valid rows"
    if len(pos) == 0 or len(neg) == 0:
        return None, "no anomaly or no normal samples"
    y = [1] * len(pos) + [0] * len(neg)
    return float(roc_auc_score(y, pd.concat([pos, neg]).to_numpy())), None


def subtype_rates(df: pd.DataFrame, group_keys: Iterable[str] = ("run_name", "repeat"),
                  score_col: str = "score") -> pd.DataFrame:
    """CI-ST2〜ST5：group_keys ごと、さらに範囲（全品目 "all" と品目ごと）×種類ごとの指標。

    入力は attach_subtype の出力に score_col（不良らしさ）を足したもの。pred が None の行は解析失敗として
    分母から除き（D3）、件数を書く。
    """
    M.validate(df)
    rows: list[dict[str, Any]] = []
    keys = list(group_keys)
    for gkey, g in df.groupby(keys, sort=True):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        scopes = [("all", g)] + [(c, gc) for c, gc in g.groupby("category", sort=True)]
        for scope, s in scopes:
            valid = s[~s["pred"].map(M._is_missing)]
            normal_scores = valid.loc[valid["subtype"] == NORMAL_SUBTYPE, score_col]
            for sub, ss in s.groupby("subtype", sort=True):
                v = ss[~ss["pred"].map(M._is_missing)]
                n_ng = int((v["pred"] == "NG").sum())
                is_normal = sub == NORMAL_SUBTYPE
                row = {**dict(zip(keys, gkey)), "scope": scope, "subtype": sub,
                       "n_total": len(ss), "n_valid": len(v), "n_parse_fail": len(ss) - len(v),
                       "n_ng": n_ng, "ng_rate": M._div(n_ng, len(v)),
                       "ng_rate_meaning": "false_positive_rate" if is_normal else "recall",
                       "auroc_vs_good": None, "auroc_reason": None}
                if not is_normal:
                    row["auroc_vs_good"], row["auroc_reason"] = _auroc(v[score_col], normal_scores)
                rows.append(row)
    return pd.DataFrame(rows)


def build_subtype_report(run_dirs: list[str], split_csv: str | Path, out_dir: str | Path) -> dict[str, Path]:
    from .report import load_run

    split = pd.read_csv(split_csv)
    frames = []
    for d in run_dirs:
        df = attach_subtype(load_run(d), split)
        df["score"] = vlm_score(df)
        frames.append(df)
    table = subtype_rates(pd.concat(frames, ignore_index=True))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out / "subtypes.csv", "md": out / "subtypes.md"}
    table.to_csv(paths["csv"], index=False, lineterminator="\n")
    paths["md"].write_text(_markdown(table), encoding="utf-8", newline="\n")
    return paths


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _markdown(t: pd.DataFrame) -> str:
    cols = ["run_name", "repeat", "scope", "subtype", "n_total", "n_valid", "n_parse_fail", "ng_rate",
            "ng_rate_meaning", "auroc_vs_good"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in t[cols].itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines) + "\n"
