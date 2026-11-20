"""11/30 の記事の画像の部分集合を作る（plan.md の条件、PL-D4）。

- visa_platform_100.csv: visa_test_200.csv から、pcb1・capsules の良品・不良品を 25 枚ずつ（random_state=42）
- visa_platform_claude50.csv: 上の100枚から、品目×正解ごとに半分ずつ。並べた順（capsules/anomaly, capsules/normal,
  pcb1/anomaly, pcb1/normal）に 13・12・13・12 枚（random_state=42）

  uv run python scripts/make_subset.py     # platform/ で実行。../data/splits/ に書く
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SEED = 42
SPLITS = Path(__file__).resolve().parents[2] / "data" / "splits"


def main() -> None:
    test = pd.read_csv(SPLITS / "visa_test_200.csv")
    groups = sorted(test[test.category.isin(["pcb1", "capsules"])].groupby(["category", "label"]))
    p100 = pd.concat([g.sample(25, random_state=SEED) for _, g in groups]).reset_index(drop=True)
    sizes = [13, 12, 13, 12]
    halves = sorted(p100.groupby(["category", "label"]))
    c50 = pd.concat([g.sample(n, random_state=SEED) for (_, g), n in zip(halves, sizes)]).reset_index(drop=True)
    assert len(p100) == 100 and len(c50) == 50 and set(c50.image_id) <= set(p100.image_id)
    p100.to_csv(SPLITS / "visa_platform_100.csv", index=False)
    c50.to_csv(SPLITS / "visa_platform_claude50.csv", index=False)
    for name, df in (("visa_platform_100", p100), ("visa_platform_claude50", c50)):
        print(name, df.groupby(["category", "label"]).size().to_dict())


if __name__ == "__main__":
    main()
