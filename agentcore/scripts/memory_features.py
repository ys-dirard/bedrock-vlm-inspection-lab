"""12/17：記憶の100件と評価の50枚の特徴の文を、12/13 のコードで作る（手元の CPU。API は呼ばない。CI-AC-M4・M5）。

12/13 のコード（lab ac/12-13 の commit 8df4314 の vlmlab.anomalydino・vlmlab.egress）を、12/13 のワークツリーの .venv の
python で読むだけで使う（12/13 のワークツリーには何も書かない）。
- 記憶の100件: AnomalyDINO（12/13 の ad2_maps_* と同じ設定）で地図を作り、runs/ad3_maps_memory/ に置く
- 区切り（パッチのしきい値・異常度の区切り）は、12/13 と同じ学習用の良品（visa_calib_normal.csv）の地図
  （12/13 の runs/ad2_maps_calib）だけで決める（CI-E1・E2）
- 評価の50枚（visa_platform_claude50.csv）の地図は、12/13 の runs/ad2_maps_test（visa_test_200 の200枚）から使う

  <12-13 の .venv の python> scripts/memory_features.py --visa-root C:/.../data/visa
出力: runs/agentcore/memory_features/（memory_records.csv、eval_queries.csv、calibration.json）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
LAB_1213 = LAB.parent / "12-13"
EXPECTED_1213 = "8df4314"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_HOME", str(LAB_1213 / "data" / "torch_home"))

import pandas as pd  # noqa: E402

from agentlab.memcases import JUDGEMENT, query_text, record_text  # noqa: E402

PATCH_QUANTILE = 0.999  # 12/13 の egress-features の既定


def check_1213() -> None:
    head = subprocess.run(["git", "-C", str(LAB_1213), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(LAB_1213), "status", "--porcelain", "--", "src"], capture_output=True,
                           text=True, check=True).stdout.strip()
    if head != EXPECTED_1213 or dirty:
        sys.exit(f"12-13 のコードが {EXPECTED_1213} のままではない（HEAD {head}、src の変更 {bool(dirty)}）")
    import vlmlab
    if Path(vlmlab.__file__).resolve().parents[1] != (LAB_1213 / "src").resolve():
        sys.exit(f"vlmlab が 12-13 のものではない: {vlmlab.__file__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa-root", required=True)
    a = ap.parse_args()
    check_1213()
    from vlmlab.anomalydino import AdConfig, execute
    from vlmlab.egress import build_features, load_scores_with_maps, save_calibrations

    splits = LAB / "data" / "splits"
    cfg_path = LAB / "configs" / "ad3_maps_memory.yaml"
    run_dir = execute(AdConfig.load(cfg_path), cfg_path, LAB, Path(a.visa_root), LAB / "runs",
                      LAB_1213 / "third_party" / "AnomalyDINO")
    calib_scores = load_scores_with_maps(LAB_1213 / "runs" / "ad2_maps_calib")
    calib, refs = pd.read_csv(splits / "visa_calib_normal.csv"), pd.read_csv(splits / "visa_refs.csv")

    memory = pd.read_csv(splits / "visa_memory_100.csv")
    mem_feats, cals = build_features(load_scores_with_maps(run_dir), calib_scores, calib, refs,
                                     memory.drop(columns=["in_v20"]), PATCH_QUANTILE)
    eval50 = pd.read_csv(splits / "visa_platform_claude50.csv")
    test_scores = load_scores_with_maps(LAB_1213 / "runs" / "ad2_maps_test")
    eval_feats, cals2 = build_features(test_scores[test_scores["image_id"].isin(set(eval50["image_id"]))],
                                       calib_scores, calib, refs, eval50, PATCH_QUANTILE)
    if {c: v.patch_threshold for c, v in cals.items()} != {c: v.patch_threshold for c, v in cals2.items()}:
        sys.exit("記憶と評価で区切りが違う")

    out = ROOT / "runs" / "agentcore" / "memory_features"
    out.mkdir(parents=True, exist_ok=True)
    mem = mem_feats.merge(memory[["image_id", "in_v20"]], on="image_id", validate="one_to_one")
    mem["judgement"] = mem["label"].map(JUDGEMENT)
    mem["record_text"] = [record_text(c, t, lab) for c, t, lab in zip(mem["category"], mem["text_fine"], mem["label"])]
    mem.to_csv(out / "memory_records.csv", index=False, lineterminator="\n")
    ev = eval_feats.copy()
    ev["query_text"] = [query_text(c, t) for c, t in zip(ev["category"], ev["text_fine"])]
    ev.to_csv(out / "eval_queries.csv", index=False, lineterminator="\n")
    save_calibrations(cals, out / "calibration.json")
    print(out, len(mem), len(ev))
    print(mem.groupby(["category", "label"])["n_regions"].describe()[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
