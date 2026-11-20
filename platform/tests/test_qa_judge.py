"""条件 Q の採点（scripts/qa_eval.judge）。AWS は呼ばない。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qa_eval import judge  # noqa: E402


def test_ints():
    t = [{"n": "100", "fail": "100"}]
    assert judge("ints", t, "判定は 100 件で、解析失敗は 100 件です。")[0]
    assert not judge("ints", t, "判定は 100 件で、解析失敗は 99 件です。")[0]


def test_rates_accept_decimal_or_percent():
    t = [{"recall": "0.5384615384615384", "fpr": "0.4166666666666667"}]
    assert judge("rates", t, "再現率は 0.538、誤検知率は 0.417 です。")[0]
    assert judge("rates", t, "再現率は 53.8%、誤検知率は 41.7% です。")[0]
    assert not judge("rates", t, "再現率は 0.54、誤検知率は 0.42 です。")[0]  # 2桁の丸めは不可


def test_rates_with_category_rows():
    t = [{"category": "capsules", "recall": "0.75"}, {"category": "pcb1", "recall": "0.5"}]
    assert judge("rates", t, "capsules: 0.75、pcb1: 0.5")[0]
    assert not judge("rates", t, "capsules: 0.75")[0]


def test_money_within_one_percent():
    t = [{"run_name": "p-claude", "cost": "1.94"}, {"run_name": "p-nova", "cost": "0.2412"}]
    assert judge("money", t, "p-claude は 1.945 USD、p-nova は 0.241 USD")[0]
    assert not judge("money", t, "p-claude は 2.1 USD、p-nova は 0.241 USD")[0]


def test_ids_exact_set():
    t = [{"image_id": "pcb1/anomaly/085"}, {"image_id": "capsules/anomaly/012"}]
    assert judge("ids", t, "見逃しは capsules/anomaly/012 と pcb1/anomaly/085 です")[0]
    assert not judge("ids", t, "見逃しは pcb1/anomaly/085 です")[0]
    assert not judge("ids", t, "capsules/anomaly/012, pcb1/anomaly/085, pcb1/anomaly/001")[0]


def test_best():
    t = [{"run_name": "p-claude", "recall": "0.5384615384615384"}]
    assert judge("best", t, "いちばん高いのは p-claude で、再現率は 53.8% です")[0]
    assert not judge("best", t, "いちばん高いのは p-qwen で、再現率は 53.8% です")[0]


def test_numbers_inside_names_and_ids_are_ignored():
    from qa_eval import numbers_in
    assert numbers_in("pcb1 の再現率は 1.0、capsules/anomaly/085 は 0.5、合計 1,234 件。") == [1.0, 0.5, 1234.0]
    t = [{"category": "pcb1", "recall": "1.0"}]
    assert not judge("rates", t, "pcb1 の再現率は 0.8 です")[0]
