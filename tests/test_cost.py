import pytest

from vlmlab.cost import BudgetExceeded, BudgetGuard, Price, PricingTable, call_cost

P = Price("m", "standard", input_per_mtok=0.33, output_per_mtok=2.75, cache_read_per_mtok=0.0825)


def test_cost_formula():
    """CI-C1"""
    assert call_cost(P, 1_000_000, 0) == pytest.approx(0.33)
    assert call_cost(P, 680, 80) == pytest.approx((680 * 0.33 + 80 * 2.75) / 1e6)
    assert call_cost(P, 1000, 100, cache_read_tokens=2000) == pytest.approx((1000 * 0.33 + 100 * 2.75 + 2000 * 0.0825) / 1e6)


def test_null_price_with_zero_tokens_is_zero():
    """CI-C2：キャッシュ書きの単価が null でも、使っていなければ計算できる"""
    assert call_cost(P, 100, 10, cache_write_tokens=0) is not None


def test_null_price_with_tokens_gives_none_not_zero():
    """CI-C2"""
    assert call_cost(P, 100, 10, cache_write_tokens=5) is None
    unknown = Price("m", "standard", None, None)
    assert call_cost(unknown, 100, 10) is None
    assert call_cost(unknown, 0, 0) == 0.0


def test_negative_tokens_raise():
    """CI-C3"""
    with pytest.raises(ValueError):
        call_cost(P, -1, 0)


def test_unknown_model_raises(tmp_path):
    """CI-C3"""
    f = tmp_path / "pricing.yaml"
    f.write_text("models:\n  - {model_id: a, service_tier: standard, input_per_mtok: 1, output_per_mtok: 2}\n")
    t = PricingTable.load(f)
    assert t.get("a").input_per_mtok == 1
    with pytest.raises(KeyError):
        t.get("a", "flex")
    with pytest.raises(KeyError):
        t.get("b")


def test_runnable_requires_input_and_output_price():
    """CI-C5"""
    assert P.runnable
    assert not Price("m", "standard", 0.33, None).runnable
    assert not Price("m", "standard", None, 2.75).runnable


def test_guard_stops_before_exceeding():
    """CI-C4：累計 + 最大の1呼び出しが予算を超えるなら、次を呼ばない"""
    g = BudgetGuard(budget_usd=0.010)
    g.check_before_call()
    g.add(0.004)
    g.check_before_call()  # 0.004 + 0.004 = 0.008 <= 0.010
    g.add(0.004)
    with pytest.raises(BudgetExceeded):
        g.check_before_call()  # 0.008 + 0.004 = 0.012 > 0.010
    assert g.spent_usd <= 0.010


def test_guard_exact_budget_is_allowed():
    """CI-C4：ちょうど予算に達する呼び出しは許す"""
    g = BudgetGuard(budget_usd=0.010, spent_usd=0.005, max_call_usd=0.005)
    g.check_before_call()


def test_guard_refuses_unknown_cost():
    """CI-C2, CI-C4"""
    with pytest.raises(ValueError):
        BudgetGuard(1.0).add(None)


def test_guard_resume_includes_previous_costs():
    """CI-C6"""
    g = BudgetGuard.resume(0.01, [0.004, 0.003])
    assert g.spent_usd == pytest.approx(0.007) and g.max_call_usd == 0.004
    with pytest.raises(BudgetExceeded):
        g.check_before_call()
    with pytest.raises(ValueError):
        BudgetGuard.resume(0.01, [0.004, None])
