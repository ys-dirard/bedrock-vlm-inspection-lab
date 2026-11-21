"""12/17：記憶に入れる事例100件の一覧を作る（API は呼ばない。CI-AC-M1〜M3）。

  uv run python scripts/make_memory_cases.py --visa-root C:/.../data/visa     # ../data/splits/visa_memory_100.csv に書く
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.memcases import select_memory_cases  # noqa: E402

SPLITS = LAB / "data" / "splits"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa-root", required=True, help="VisA のフォルダ（split_csv/1cls.csv を読む）")
    a = ap.parse_args()
    cases = select_memory_cases(
        pd.read_csv(Path(a.visa_root) / "split_csv" / "1cls.csv", keep_default_na=False),
        pd.read_csv(SPLITS / "visa_test_200.csv"), pd.read_csv(SPLITS / "visa_refs.csv"),
        pd.read_csv(SPLITS / "visa_calib_normal.csv"), pd.read_csv(SPLITS / "visa_platform_claude50.csv"))
    out = SPLITS / "visa_memory_100.csv"
    cases.to_csv(out, index=False, lineterminator="\n")
    print(out, len(cases), cases.groupby(["category", "label"]).size().to_dict(),
          "v20:", cases[cases.in_v20].groupby(["category", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
