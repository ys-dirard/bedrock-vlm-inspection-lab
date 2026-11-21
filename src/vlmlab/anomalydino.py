"""手元の AnomalyDINO で、良品の見本から画像ごとの異常度を出す（12/10 の記事。AWS は使わない）。docs/contracts.md の CI-A1〜A5。

公式実装（dammsi/AnomalyDINO、Apache-2.0）の `src/backbones.py`（DINOv2 の読み込み・前処理・特徴・背景の除去）と
`src/utils.py`（見本の回転）をそのまま読み込み、`src/detection.py` の1近傍の距離と `src/post_eval.py` の
mean_top1p を写して使う。`detection.py` は CUDA の同期を必ず呼ぶため、CPU では呼べない。

- 公式実装は `git clone https://github.com/dammsi/AnomalyDINO third_party/AnomalyDINO` で置き、設定の commit に合わせる
- DINOv2 は torch.hub で、設定の commit に固定して読む（重みは TORCH_HOME の下に置く）
- 結果は runs/<name>/scores.jsonl（1枚1行）と meta.json（環境・品目ごとの見本の時間・メモリ）
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .run import JST, load_jsonl, select_limit

SCORE_DIGITS = 9
MASK_SEED = 42  # 背景の除去の PCA（svd_solver="randomized"、random_state なし）が np.random を使うので、画像ごとに固定する


# ---------------------------------------------------------------- 設定

@dataclass
class AdConfig:
    name: str
    split: str
    refs: str
    anomalydino_commit: str
    dinov2_commit: str
    n_refs: int = 1
    model_name: str = "dinov2_vits14"
    resolution: int = 448          # 短辺（公式の --resolution。既定 448）
    masking: bool = True           # テスト画像の背景を除く（公式の VisA の既定）
    rotation: bool = True          # 見本を8方向に回転して足す（公式の agnostic）
    device: str = "cpu"
    num_threads: int | None = None  # None なら torch の既定
    save_maps: bool = False        # パッチの距離の地図を runs/<name>/maps/ に残す（12/13 の記事。CI-A6）。異常度は変わらない

    @classmethod
    def load(cls, path: str | Path) -> "AdConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def identity(self, refs_sha256: str) -> dict[str, Any]:
        """異常度の中身を決める設定（CI-A5）。device と num_threads は時間だけに効くので含めない。"""
        return {"model_name": self.model_name, "resolution": self.resolution, "masking": self.masking,
                "rotation": self.rotation, "n_refs": self.n_refs, "anomalydino_commit": self.anomalydino_commit,
                "dinov2_commit": self.dinov2_commit, "refs_sha256": refs_sha256}


# ---------------------------------------------------------------- 計算（numpy だけ。単体テストの対象）

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if (n == 0).any():
        raise ValueError("zero feature vector")
    return x / n


def patch_distances(test_feats: np.ndarray, mask: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """CI-A2：各パッチの、見本のパッチ（bank）への最小コサイン距離。mask が False のパッチは 0。

    test_feats: (N, D)、mask: (N,) bool、bank: (M, D)（正規化の前でよい）。戻り値は (N,) float64。
    """
    test_feats = np.asarray(test_feats)
    mask = np.asarray(mask, dtype=bool)
    if test_feats.ndim != 2 or bank.ndim != 2 or test_feats.shape[1] != bank.shape[1]:
        raise ValueError(f"shape mismatch: test {test_feats.shape}, bank {bank.shape}")
    if mask.shape != (test_feats.shape[0],):
        raise ValueError(f"mask shape {mask.shape} does not match {test_feats.shape[0]} patches")
    out = np.zeros(test_feats.shape[0], dtype=float)
    if mask.any():
        sim = _l2_normalize(test_feats[mask]) @ _l2_normalize(bank).T
        out[mask] = 1.0 - sim.max(axis=1).astype(float)
    return out


def image_score(distances: np.ndarray) -> float:
    """CI-A3：大きい順に floor(0.01 N) 個の平均。0 個なら最大値（公式の mean_top1p）。"""
    d = np.asarray(distances, dtype=float).ravel()
    if d.size == 0:
        raise ValueError("no patches")
    if not np.isfinite(d).all():
        raise ValueError("non-finite patch distance")
    k = int(d.size * 0.01)
    if k == 0:
        return round(float(d.max()), SCORE_DIGITS)
    return round(float(np.sort(d)[::-1][:k].mean()), SCORE_DIGITS)


def select_refs(test: pd.DataFrame, refs: pd.DataFrame, n_refs: int) -> dict[str, list[pd.Series]]:
    """CI-A1：テストの品目ごとに ref_rank 1〜n_refs の見本。枚数が違う、テストと重なるなら止める。"""
    if n_refs < 1:
        raise ValueError("n_refs must be >= 1")
    overlap = set(test["image_id"]) & set(refs["image_id"])
    if overlap:
        raise ValueError(f"refs overlap with test images: {sorted(overlap)[:5]}")
    out = {}
    for cat in sorted(test["category"].unique()):
        r = refs[(refs["category"] == cat) & (refs["ref_rank"] <= n_refs)].sort_values("ref_rank")
        if len(r) != n_refs or sorted(r["ref_rank"].tolist()) != list(range(1, n_refs + 1)):
            raise ValueError(f"{cat}: need ref_rank 1..{n_refs}, got {r['ref_rank'].tolist()}")
        if (r["label"] != "normal").any():
            raise ValueError(f"{cat}: refs must be normal")
        out[cat] = [row for _, row in r.iterrows()]
    return out


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 公式実装の読み込み（torch を使う）

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


class OfficialAnomalyDINO:
    """公式の backbones.py・utils.py を使って、見本の特徴（bank）とテスト画像の異常度を出す。"""

    def __init__(self, cfg: AdConfig, anomalydino_root: Path):
        head = git_head(anomalydino_root)
        if head != cfg.anomalydino_commit:
            raise ValueError(f"AnomalyDINO commit {head} != config {cfg.anomalydino_commit}")
        import cv2
        import torch
        if cfg.num_threads:
            torch.set_num_threads(cfg.num_threads)
            cv2.setNumThreads(cfg.num_threads)
        self.cv2, self.torch, self.cfg = cv2, torch, cfg
        self.bb = _load_module("anomalydino_backbones", anomalydino_root / "src" / "backbones.py")
        self.utils = _load_module("anomalydino_utils", anomalydino_root / "src" / "utils.py")
        if cfg.model_name not in self.bb.DINOv2Wrapper.MODELS:
            raise ValueError(f"only DINOv2 backbones are supported here, got {cfg.model_name}")

        dinov2_ref = f"facebookresearch/dinov2:{cfg.dinov2_commit}"

        class PinnedDINOv2(self.bb.DINOv2Wrapper):
            """公式の DINOv2Wrapper と同じ。違いは torch.hub の読み込み先を commit に固定することだけ。"""

            def load_model(inner):
                model = torch.hub.load(dinov2_ref, inner.model_name, trust_repo=True, skip_validation=True)
                model.eval()
                inner.patch_size = model.patch_size
                inner.transform = inner._dino_transform()
                return model.to(inner.device)

        t0 = time.perf_counter()
        self.model = PinnedDINOv2(cfg.model_name, cfg.device, cfg.resolution)
        self.load_time_s = time.perf_counter() - t0

    def _read(self, path: Path) -> np.ndarray:
        img = self.cv2.imread(str(path), self.cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)

    def build_bank(self, ref_paths: list[Path]) -> np.ndarray:
        """見本（回転を含む）の全パッチの特徴。公式と同じく、見本には背景の除去をかけない（mask_ref_images=False）。"""
        feats = []
        with self.torch.inference_mode():
            for p in ref_paths:
                img = self._read(p)
                imgs = self.utils.augment_image(img) if self.cfg.rotation else [img]
                for im in imgs:
                    tensor, _ = self.model.prepare_image(im)
                    feats.append(self.model.extract_features(tensor))
        return np.concatenate(feats, axis=0).astype(np.float32)

    def score(self, path: Path, bank: np.ndarray) -> dict[str, Any]:
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            img = self._read(path)
            tensor, grid = self.model.prepare_image(img)
            feats = self.model.extract_features(tensor)
            if self.cfg.masking:
                np.random.seed(MASK_SEED)  # 画像の順番や再開に関係なく、同じ画像なら同じ背景になるように
                mask = self.model.compute_background_mask(feats, grid, threshold=10, masking_type=True)
            else:
                mask = np.ones(feats.shape[0], dtype=bool)
            d = patch_distances(feats, mask, bank)
            s = image_score(d)
        res = {"score": s, "time_s": time.perf_counter() - t0, "n_patches": int(d.size),
               "n_foreground": int(np.asarray(mask).sum()), "grid": list(grid)}
        if self.cfg.save_maps:
            res["_map"] = (d, np.asarray(mask, dtype=bool))
        return res



# ---------------------------------------------------------------- 実行

def _peak_memory_mb() -> dict[str, float | None]:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows だけ
        return {"rss_mb": mi.rss / 2**20, "peak_mb": peak / 2**20 if peak else None}
    except ImportError:
        return {"rss_mb": None, "peak_mb": None}


def map_filename(image_id: str) -> str:
    """CI-A6：地図のファイル名。image_id の / を __ に替える。"""
    if "__" in image_id:
        raise ValueError(f"image_id must not contain '__': {image_id}")
    return image_id.replace("/", "__") + ".npz"


def save_map(path: Path, distances: np.ndarray, mask: np.ndarray, grid: list[int]) -> None:
    """CI-A6：パッチの距離（float32）と前景の印を、grid（縦, 横）の形で残す。"""
    h, w = int(grid[0]), int(grid[1])
    d = np.asarray(distances, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if d.size != h * w or m.size != h * w:
        raise ValueError(f"map size {d.size}/{m.size} does not match grid {grid}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dist=d.reshape(h, w), mask=m.reshape(h, w))


def load_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["dist"].astype(np.float64), z["mask"].astype(bool)


def plan(cfg: AdConfig, root: Path, limit: int | None) -> tuple[str, pd.DataFrame, dict[str, list[pd.Series]]]:
    test = pd.read_csv(root / cfg.split)
    refs = pd.read_csv(root / cfg.refs)
    refs_by_cat = select_refs(test, refs, cfg.n_refs)
    items = select_limit(test, limit)
    run_name = cfg.name if limit is None else f"{cfg.name}__trial{limit}"
    return run_name, items, refs_by_cat


def execute(cfg: AdConfig, config_path: Path, root: Path, visa_root: Path, runs_dir: Path,
            anomalydino_root: Path, limit: int | None = None) -> Path:
    run_name, items, refs_by_cat = plan(cfg, root, limit)
    out = runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    ident = cfg.identity(sha256_file(root / cfg.refs))
    ident_path = out / "identity.json"
    if ident_path.exists():
        prev = json.loads(ident_path.read_text(encoding="utf-8"))
        if prev != ident:
            raise ValueError(f"settings differ from the previous run (CI-A5): {prev} != {ident}")
    else:
        ident_path.write_text(json.dumps(ident, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    scores_path = out / "scores.jsonl"
    done = {r["image_id"] for r in load_jsonl(scores_path)}
    todo = items[~items["image_id"].isin(done)]
    print(f"{run_name}: {len(todo)} to score ({len(done)} done)")
    if todo.empty:
        return out

    import torch
    ad = OfficialAnomalyDINO(cfg, anomalydino_root)
    meta: dict[str, Any] = {
        "run_name": run_name, "started": datetime.now(JST).isoformat(timespec="seconds"),
        "config": asdict(cfg), "python": sys.version.split()[0], "torch": torch.__version__,
        "cv2": ad.cv2.__version__, "torch_threads": torch.get_num_threads(), "device": cfg.device,
        "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "torch_home": os.environ.get("TORCH_HOME"), "model_load_s": ad.load_time_s, "bank": {},
    }
    with scores_path.open("a", encoding="utf-8") as f:
        for cat, g in todo.groupby("category", sort=True):
            ref_rows = refs_by_cat[cat]
            t0 = time.perf_counter()
            bank = ad.build_bank([visa_root / r["path"] for r in ref_rows])
            meta["bank"][cat] = {"refs": [r["image_id"] for r in ref_rows], "n_bank_patches": int(bank.shape[0]),
                                 "time_s": time.perf_counter() - t0}
            for _, row in g.iterrows():
                res = ad.score(visa_root / row["path"], bank)
                if "_map" in res:
                    d, m = res.pop("_map")
                    save_map(out / "maps" / map_filename(row["image_id"]), d, m, res["grid"])
                if not math.isfinite(res["score"]):
                    raise ValueError(f"{row['image_id']}: non-finite score")
                rec = {"run_name": run_name, "image_id": row["image_id"], "category": cat, "label": row["label"],
                       **res, "timestamp": datetime.now(JST).isoformat(timespec="seconds")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {row['image_id']}: score={res['score']:.4f} time={res['time_s']:.2f}s")
    meta["finished"] = datetime.now(JST).isoformat(timespec="seconds")
    meta["memory"] = _peak_memory_mb()
    with (out / "meta.jsonl").open("a", encoding="utf-8") as f:  # 再開ごとに1行
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return out
