"""学習データに混ぜた不良品を見つける（12/22 の記事）。docs/contracts.md の CI-N1〜N7。

- 混入ありのデータ：VisA の train の良品に、評価に使っていない test の不良品を 5% 混ぜる。
  判定に使う一覧（noise_train_420.csv）には中立の ID と label="unknown" だけを書き、どれが混入かは別のファイル
  （noise_truth_420.csv）に分ける。モデルに送るのは画像のバイト列だけで、ID・パス・正解は送らない
- AnomalyDINO の異常度は、テスト画像のパッチごとの「見本の画像ごとの最大の類似度」（類似度の表）から計算する。
  見本の組み合わせを変えても特徴を作り直さずに済む（自分以外を見本にする異常度、疑った画像を除いた見本など）

この file は numpy と pandas だけを使う（単体テストの対象）。特徴づくりは noise_dino.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anomalydino import image_score

SEED = 42
CATEGORIES = ("capsules", "pcb1")
RUN_COLUMNS = ["image_id", "category", "label", "split", "path"]
TRUTH_COLUMNS = ["image_id", "category", "source_image_id", "label", "injected", "path"]


# ---------------------------------------------------------------- 混入ありのデータ（CI-N1）

def make_noisy_split(table: pd.DataFrame, exclude_ids: set[str], n_normal: int = 200, n_injected: int = 10,
                     categories=CATEGORIES, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """品目ごとに train の良品 n_normal 枚と、exclude_ids に無い test の不良品 n_injected 枚を混ぜる。

    戻り値は (判定に使う一覧, 正解)。一覧の image_id は「品目/n001」の形で、並びは random_state=seed で混ぜる。
    """
    run_rows, truth_rows = [], []
    for cat in sorted(categories):
        t = table[(table["category"] == cat) & ~table["image_id"].isin(exclude_ids)]
        normal = t[(t["split"] == "train") & (t["label"] == "normal")]
        anomaly = t[(t["split"] == "test") & (t["label"] == "anomaly")]
        if len(normal) < n_normal or len(anomaly) < n_injected:
            raise ValueError(f"{cat}: need {n_normal} normal / {n_injected} anomaly, "
                             f"have {len(normal)} / {len(anomaly)}")
        mixed = pd.concat([normal.sample(n_normal, random_state=seed),
                           anomaly.sample(n_injected, random_state=seed)])
        mixed = mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for k, r in enumerate(mixed.itertuples(index=False), start=1):
            nid = f"{cat}/n{k:03d}"
            run_rows.append({"image_id": nid, "category": cat, "label": "unknown", "split": "noise", "path": r.path})
            truth_rows.append({"image_id": nid, "category": cat, "source_image_id": r.image_id, "label": r.label,
                               "injected": r.label == "anomaly", "path": r.path})
    run, truth = pd.DataFrame(run_rows, columns=RUN_COLUMNS), pd.DataFrame(truth_rows, columns=TRUTH_COLUMNS)
    if truth["source_image_id"].duplicated().any():
        raise ValueError("duplicated source image")
    if set(truth["source_image_id"]) & exclude_ids:
        raise ValueError("noisy set overlaps with excluded images")
    return run, truth


# ---------------------------------------------------------------- 見つけられたか（CI-N3・N4）

def detection_summary(suspected: pd.Series, truth: pd.DataFrame, method: str,
                      n_unjudged: pd.Series | None = None) -> list[dict[str, Any]]:
    """suspected：image_id → 疑ったか（bool）。品目ごとと全体の行を返す。

    - 発見率 = 疑った混入 / 混入、良品を疑った割合 = 疑った良品 / 良品
    - n_unjudged（image_id → 判定できなかったか）があれば、判定できなかった画像は「疑っていない」として数え、件数を併記する
    """
    if set(suspected.index) != set(truth["image_id"]):
        raise ValueError("suspected images do not match the truth")
    d = truth.set_index("image_id").join(suspected.rename("suspected").astype(bool))
    d["unjudged"] = False if n_unjudged is None else n_unjudged.reindex(d.index).fillna(False).astype(bool)
    rows = []
    for cat, g in [*d.groupby("category", sort=True), ("all", d)]:
        inj, nor = g[g["injected"]], g[~g["injected"]]
        rows.append({
            "method": method, "category": cat, "n_images": len(g), "n_injected": len(inj),
            "n_found": int(inj["suspected"].sum()),
            "detection_rate": None if len(inj) == 0 else float(inj["suspected"].mean()),
            "n_normal": len(nor), "n_normal_suspected": int(nor["suspected"].sum()),
            "normal_suspect_rate": None if len(nor) == 0 else float(nor["suspected"].mean()),
            "n_suspected": int(g["suspected"].sum()), "n_unjudged": int(g["unjudged"].sum()),
        })
    return rows


def top_fraction(scores: pd.DataFrame, pct: float) -> pd.Series:
    """品目ごとに異常度の上位 pct% を疑う。k = ceil(pct × n / 100)、同じ値なら image_id の昇順（CI-F4 と同じ）。"""
    out = {}
    for _, g in scores.groupby("category", sort=True):
        k = int(np.ceil(pct * len(g) / 100))
        order = g.sort_values(["score", "image_id"], ascending=[False, True])["image_id"].tolist()
        out.update({i: n < k for n, i in enumerate(order)})
    return pd.Series(out).reindex(scores["image_id"])


# ---------------------------------------------------------------- 類似度の表から異常度（CI-N2・N5）

@dataclass
class SimTable:
    """1品目の類似度の表。

    rows：行（画像）の image_id。bank：列（見本の候補）の image_id。
    sims[r]：行 r の前景のパッチ × 見本の候補ごとの最大のコサイン類似度（float32、(n_fg, len(bank))）
    n_patches[r]：行 r のパッチの総数（背景を含む。背景のパッチの距離は 0）
    """
    rows: list[str]
    bank: list[str]
    sims: list[np.ndarray]
    n_patches: list[int]

    def __post_init__(self):
        if len(self.rows) != len(self.sims) or len(self.rows) != len(self.n_patches):
            raise ValueError("rows, sims and n_patches must have the same length")
        for s, n in zip(self.sims, self.n_patches):
            if s.ndim != 2 or s.shape[1] != len(self.bank) or s.shape[0] > n:
                raise ValueError(f"bad sims shape {s.shape} for {len(self.bank)} bank images and {n} patches")

    def score(self, row: str, use: np.ndarray) -> float:
        """use（見本の候補ごとの bool）で選んだ見本に対する異常度（CI-A2・A3 と同じ式）。"""
        use = np.asarray(use, dtype=bool)
        if use.shape != (len(self.bank),) or not use.any():
            raise ValueError("use must select at least one bank image")
        r = self.rows.index(row)
        s = self.sims[r]
        d = np.zeros(self.n_patches[r], dtype=float)
        if s.shape[0]:
            d[: s.shape[0]] = 1.0 - s[:, use].max(axis=1).astype(float)
        return image_score(d)

    def leave_one_out(self) -> pd.Series:
        """行と見本の候補が同じ集合のとき、各画像の「自分以外のすべて」を見本にした異常度（CI-N2）。"""
        if set(self.rows) != set(self.bank):
            raise ValueError("leave-one-out needs rows == bank")
        out = {}
        for row in self.rows:
            use = np.array([b != row for b in self.bank])
            out[row] = self.score(row, use)
        return pd.Series(out)

    def scores_with_bank(self, keep: set[str]) -> pd.Series:
        """keep に入っている見本の候補だけを見本にした、全行の異常度（CI-N5）。"""
        use = np.array([b in keep for b in self.bank])
        return pd.Series({row: self.score(row, use) for row in self.rows})


def save_sim_table(t: SimTable, path: Path) -> None:
    offsets = np.cumsum([0] + [s.shape[0] for s in t.sims])
    np.savez(path, rows=np.array(t.rows), bank=np.array(t.bank), n_patches=np.array(t.n_patches),
             offsets=offsets, sims=np.concatenate(t.sims, axis=0).astype(np.float32))


def load_sim_table(path: Path) -> SimTable:
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    all_sims = z["sims"]  # 一度だけ読む（npz はキーを引くたびに配列全体を読み直す）
    sims = [all_sims[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return SimTable([str(x) for x in z["rows"]], [str(x) for x in z["bank"]], sims, [int(x) for x in z["n_patches"]])


# ---------------------------------------------------------------- 取り除いたあとの精度（CI-N6）

def threshold_at_miss(scores: np.ndarray, is_anomaly: np.ndarray, target: float) -> float:
    """不良品の異常度を候補に、「異常度 < t の不良品の割合 ≤ target」を満たす最大の t（CI-F3 と同じ。楽観的）。"""
    s_anom = np.sort(scores[is_anomaly])
    best = None
    for t in np.unique(s_anom):
        if (s_anom < t).mean() <= target:
            best = float(t)
    if best is None:
        raise ValueError("no threshold")
    return best


def eval_test_metrics(scores: pd.Series, test: pd.DataFrame, target_miss: float = 0.05) -> dict[str, Any]:
    """テスト画像の異常度から AUROC と、見逃し率 target 以下にしたときの誤検知率（しきい値はテストの正解で決める）。"""
    from sklearn.metrics import roc_auc_score

    d = test.set_index("image_id").join(scores.rename("score"), how="left")
    if d["score"].isna().any():
        raise ValueError("missing test scores")
    y = (d["label"] == "anomaly").to_numpy()
    s = d["score"].to_numpy(dtype=float)
    t = threshold_at_miss(s, y, target_miss)
    return {"n_test": len(d), "auroc": float(roc_auc_score(y, s)), "threshold": t,
            "miss_rate_at_t": float((s[y] < t).mean()), "false_positive_rate_at_t": float((s[~y] >= t).mean())}


# ---------------------------------------------------------------- まとめ（CI-N7）

DINO_PCTS = (5, 10, 20)


def build_report(truth_csv: Path, test_csv: Path, dino_dir: Path, nova_dir: Path | None, out_dir: Path,
                 target_miss: float = 0.05) -> dict[str, Path]:
    """混入の発見（noise_detection_summary.csv）と、取り除いたあとの精度（retrain_effect.csv）を出す。API は呼ばない。"""
    truth = pd.read_csv(truth_csv)
    test = pd.read_csv(test_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = sorted(truth["category"].unique())
    noisy_tables = {c: load_sim_table(dino_dir / f"{c}.noisy.npz") for c in cats}
    test_tables = {c: load_sim_table(dino_dir / f"{c}.test.npz") for c in cats}
    loo = pd.concat([noisy_tables[c].leave_one_out() for c in cats])
    if set(loo.index) != set(truth["image_id"]):
        raise ValueError("AnomalyDINO rows do not match the truth")
    scores = truth.set_index("image_id")[["category", "injected"]].join(loo.rename("score")).reset_index()

    rows: list[dict[str, Any]] = []
    removals: dict[str, set[str]] = {"all (混入あり)": set(),
                                     "oracle (混入を正解で除く)": set(truth.loc[truth["injected"], "image_id"])}
    from sklearn.metrics import roc_auc_score
    for pct in DINO_PCTS:
        sus = top_fraction(scores[["image_id", "category", "score"]], pct)
        rows += detection_summary(sus, truth, f"AnomalyDINO 上位{pct}%")
        removals[f"AnomalyDINO 上位{pct}%を除く"] = set(sus[sus].index)
    if nova_dir is not None:
        from .report import load_run
        nova = load_run(nova_dir).set_index("image_id")
        if set(nova.index) != set(truth["image_id"]):
            raise ValueError("Nova records do not match the truth")
        sus = (nova["pred"] == "NG")
        rows += detection_summary(sus, truth, "Nova 2 Lite（NG）", n_unjudged=nova["pred"].isna())
        removals["Nova 2 Lite の NG を除く"] = set(sus[sus].index)
    det = pd.DataFrame(rows)
    auroc = [{"category": c, "loo_auroc": float(roc_auc_score(g["injected"], g["score"]))} for c, g in scores.groupby("category")]

    eff = []
    for cond, removed in removals.items():
        for c in cats:
            keep = set(noisy_tables[c].bank) - removed
            s = test_tables[c].scores_with_bank(keep)
            m = eval_test_metrics(s, test[test["category"] == c], target_miss)
            bank = truth[truth["image_id"].isin(keep)]
            eff.append({"condition": cond, "category": c, "n_bank": len(bank),
                        "n_injected_in_bank": int(bank["injected"].sum()), **m})
    paths = {"detection": out_dir / "noise_detection_summary.csv", "retrain": out_dir / "retrain_effect.csv",
             "scores": out_dir / "anomalydino_loo_scores.csv", "loo_auroc": out_dir / "anomalydino_loo_auroc.csv"}
    det.to_csv(paths["detection"], index=False, lineterminator="\n")
    pd.DataFrame(eff).to_csv(paths["retrain"], index=False, lineterminator="\n")
    scores.to_csv(paths["scores"], index=False, lineterminator="\n")
    pd.DataFrame(auroc).to_csv(paths["loo_auroc"], index=False, lineterminator="\n")
    return paths
