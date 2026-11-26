import pytest

from build_cheatsheet import lookup, per_mtok


def test_per_mtok_units():
    assert per_mtok({"pricePerUnit": "0.0003960000", "unit": "1K tokens"}) == pytest.approx(0.396)
    assert per_mtok({"pricePerUnit": "3.3000000000", "unit": "1M tokens"}) == pytest.approx(3.3)
    with pytest.raises(ValueError):
        per_mtok({"pricePerUnit": "1", "unit": "image"})


def test_lookup_requires_exactly_one_row():
    rows = [
        {"model": "A", "usageType": "u1", "pricePerUnit": "0.001", "unit": "1K tokens"},
        {"model": "A", "usageType": "u2", "pricePerUnit": "1", "unit": "1M tokens"},
        {"model": "A", "usageType": "u2", "pricePerUnit": "2", "unit": "1M tokens"},
    ]
    assert lookup(rows, "A", "u1") == pytest.approx(1.0)
    assert lookup(rows, "A", "u2") is None  # 重複は採らない
    assert lookup(rows, "A", "missing") is None
    assert lookup(rows, "A", None) is None
