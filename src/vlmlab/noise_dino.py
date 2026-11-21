"""手元の AnomalyDINO で、混入ありのデータの類似度の表を作る（12/22 の記事。AWS は使わない）。CI-N2。

12/10 の `anomalydino.OfficialAnomalyDINO`（公式実装の backbones.py を commit 固定で読み込む）で特徴と背景を出し、
品目ごとに次の2つの表を runs/<name>/<品目>.noisy.npz・<品目>.test.npz に残す。

- noisy：混入ありの210枚 × 見本の候補（同じ210枚）。自分以外を見本にした異常度（leave-one-out）に使う
- test：テスト（visa_test_200.csv のその品目の100枚）× 見本の候補（混入ありの210枚）。取り除いたあとの精度に使う

見本の候補の特徴は背景を除かず（公式と同じ）、行の画像だけ背景を除く。見本は回転しない（12/10 の1枚の見本とは違う。
210枚の8方向だと特徴が約 2.6 GB になるため）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .anomalydino import MASK_SEED, AdConfig, OfficialAnomalyDINO, _l2_normalize
from .noise import SimTable, save_sim_table
from .run import JST

BANK_CHUNK = 16  # 見本の候補を何枚ずつまとめて行列の積を取るか（メモリと速さの兼ね合い。値は結果に影響しない）


@dataclass
class NoiseDinoConfig:
    name: str
    noisy_split: str
    test_split: str
    anomalydino_commit: str
    dinov2_commit: str
    model_name: str = "dinov2_vits14"
    resolution: int = 448
    masking: bool = True
    device: str = "cpu"
    num_threads: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "NoiseDinoConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def ad_config(self) -> AdConfig:
        return AdConfig(name=self.name, split=self.test_split, refs="", anomalydino_commit=self.anomalydino_commit,
                        dinov2_commit=self.dinov2_commit, model_name=self.model_name, resolution=self.resolution,
                        masking=self.masking, rotation=False, device=self.device, num_threads=self.num_threads)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(cfg: NoiseDinoConfig, root: Path, limit: int | None) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """limit があれば、品目ごとに混入ありの先頭 limit 枚と、テストの先頭 limit 枚だけを使う（試運転）。"""
    noisy = pd.read_csv(root / cfg.noisy_split)
    test = pd.read_csv(root / cfg.test_split)
    test = test[test["category"].isin(noisy["category"].unique())]
    if set(noisy["image_id"]) & set(test["image_id"]):
        raise ValueError("noisy and test image ids overlap")
    if set(noisy["path"]) & set(test["path"]):
        raise ValueError("noisy and test images overlap (same path)")
    if limit is not None:
        noisy = noisy.sort_values("image_id").groupby("category").head(limit)
        test = test.groupby("category").head(limit)
    run_name = cfg.name if limit is None else f"{cfg.name}__trial{limit}"
    return run_name, noisy.reset_index(drop=True), test.reset_index(drop=True)


def _features(ad: OfficialAnomalyDINO, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """L2 正規化した特徴と、背景の除去の結果（行に使うとき）。CI-A2 と同じく画像ごとに乱数を固定する。"""
    with ad.torch.inference_mode():
        tensor, grid = ad.model.prepare_image(ad._read(path))
        feats = ad.model.extract_features(tensor)
        if ad.cfg.masking:
            np.random.seed(MASK_SEED)
            mask = np.asarray(ad.model.compute_background_mask(feats, grid, threshold=10, masking_type=True), dtype=bool)
        else:
            mask = np.ones(feats.shape[0], dtype=bool)
    return _l2_normalize(feats), mask


def sim_rows(row_feats: np.ndarray, bank: list[np.ndarray]) -> np.ndarray:
    """行の特徴（正規化済み、(n, D)）× 見本の候補ごとの最大のコサイン類似度 (n, len(bank))。"""
    out = np.empty((row_feats.shape[0], len(bank)), dtype=np.float32)
    for s in range(0, len(bank), BANK_CHUNK):
        chunk = bank[s:s + BANK_CHUNK]
        sizes = [b.shape[0] for b in chunk]
        sim = row_feats @ np.concatenate(chunk, axis=0).T
        bounds = np.cumsum([0] + sizes)
        for j in range(len(chunk)):
            out[:, s + j] = sim[:, bounds[j]:bounds[j + 1]].max(axis=1)
    return out


def execute(cfg: NoiseDinoConfig, config_path: Path, root: Path, visa_root: Path, runs_dir: Path,
            anomalydino_root: Path, limit: int | None = None) -> Path:
    run_name, noisy, test = plan(cfg, root, limit)
    out = runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    ident = {**{k: v for k, v in asdict(cfg).items() if k not in ("name", "device", "num_threads")},
             "rotation": False, "noisy_sha256": _sha(root / cfg.noisy_split), "test_sha256": _sha(root / cfg.test_split),
             "limit": limit}
    ident_path = out / "identity.json"
    if ident_path.exists() and json.loads(ident_path.read_text(encoding="utf-8")) != ident:
        raise ValueError("settings differ from the previous run in this directory")
    ident_path.write_text(json.dumps(ident, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    todo = [c for c in sorted(noisy["category"].unique()) if not (out / f"{c}.done").exists()]
    print(f"{run_name}: categories to do {todo}")
    if not todo:
        return out
    import torch
    ad = OfficialAnomalyDINO(cfg.ad_config(), anomalydino_root)
    meta: dict[str, Any] = {"run_name": run_name, "started": datetime.now(JST).isoformat(timespec="seconds"),
                            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                            "model_load_s": ad.load_time_s, "categories": {}}
    for cat in todo:
        t0 = time.perf_counter()
        n_rows = noisy[noisy["category"] == cat]
        t_rows = test[test["category"] == cat]
        feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for _, r in pd.concat([n_rows, t_rows]).iterrows():
            feats[r["image_id"]] = _features(ad, visa_root / r["path"])
        t_feat = time.perf_counter() - t0
        bank_ids = n_rows["image_id"].tolist()
        bank = [feats[i][0] for i in bank_ids]  # 見本の候補は背景を除かない
        tables = {}
        for kind, rows in (("noisy", n_rows), ("test", t_rows)):
            ids = rows["image_id"].tolist()
            sims = [sim_rows(feats[i][0][feats[i][1]], bank) for i in ids]
            tables[kind] = SimTable(ids, bank_ids, sims, [int(feats[i][0].shape[0]) for i in ids])
        t_sim = time.perf_counter() - t0 - t_feat
        save_sim_table(tables["noisy"], out / f"{cat}.noisy.npz")
        save_sim_table(tables["test"], out / f"{cat}.test.npz")
        (out / f"{cat}.done").write_text("done", encoding="utf-8")  # 品目が終わった印（再開用）
        meta["categories"][cat] = {"n_noisy": len(n_rows), "n_test": len(t_rows), "features_s": t_feat,
                                   "similarity_s": t_sim, "n_bank_patches": int(sum(b.shape[0] for b in bank))}
        print(f"  {cat}: features {t_feat:.0f}s, similarity {t_sim:.0f}s")
    meta["finished"] = datetime.now(JST).isoformat(timespec="seconds")
    with (out / "meta.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return out
