"""分析のツールの SQL（CI-PL-Q*）と費用の内訳（CI-PL-C*）。AWS は呼ばない。"""

import pytest

from platformlab import costbreak as cb
from platformlab import sqlbuild as sq

T = "vlmlab_platform.results"


def test_unknown_column_is_rejected():  # CI-PL-Q1
    with pytest.raises(ValueError):
        sq.count_by(T, ["label; DROP TABLE x"])
    with pytest.raises(ValueError):
        sq.list_images(T, {"path": "x"})


def test_values_are_parameters_not_sql():  # CI-PL-Q2
    q = sq.list_images(T, {"run_name": "p-nova' OR '1'='1", "label": "anomaly"}, limit=5)
    assert "p-nova" not in q.sql and "anomaly" not in q.sql.split("WHERE")[1]
    assert q.sql.count("?") == 2
    assert q.params == ("'anomaly'", "'p-nova'' OR ''1''=''1'")  # 列の名前の順。' は '' に


def test_limit_range():  # CI-PL-Q3
    for bad in (0, 51, "10", True, 2.0):
        with pytest.raises(ValueError):
            sq.list_images(T, None, bad)
    assert "LIMIT 50" in sq.list_images(T, None, 50).sql


def test_select_only_and_fixed_table():  # CI-PL-Q4
    for q in (sq.count_by(T, ["model_id"]), sq.list_images(T), sq.compare_models(T, ["a", "b"])):
        assert q.sql.lstrip().upper().startswith("SELECT")
        assert f"FROM {T}" in q.sql
        for word in ("INSERT", "DELETE", "DROP", "CREATE", "ALTER", "UNLOAD"):
            assert word not in q.sql.upper()


def test_metric_definitions_match_november():  # CI-PL-Q5
    sql = sq.count_by(T, ["run_name"]).sql
    assert "count_if(label = 'anomaly' AND pred = 'NG') AS tp" in sql
    assert "nullif(count_if(label = 'anomaly' AND pred IS NOT NULL), 0) AS recall" in sql
    assert "nullif(count_if(label = 'normal' AND pred IS NOT NULL), 0) AS false_positive_rate" in sql


def test_build_dispatch_and_unknown_tool():
    assert sq.build("compare_models", T, {"run_names": ["a", "b"]}).params == ("'a'", "'b'")
    with pytest.raises(ValueError):
        sq.build("run_sql", T, {"sql": "SELECT 1"})
    with pytest.raises(ValueError):
        sq.compare_models(T, ["only-one"])


def test_cost_line_and_none_propagation():  # CI-PL-C1, C2, C4
    a = cb.CostLine("Step Functions", "StateTransition", 1000, "transition", 0.000025, "measured")
    assert a.cost_usd == pytest.approx(0.025)
    zero_unknown = cb.CostLine("EventBridge", "S3 data event", 0, "event", None, "estimated")
    assert zero_unknown.cost_usd == 0.0
    unknown = cb.CostLine("EventBridge", "S3 data event", 260, "event", None, "estimated")
    assert unknown.cost_usd is None
    assert cb.total([a, zero_unknown]) == pytest.approx(0.025)
    assert cb.total([a, unknown]) is None
    assert cb.per_1000([a], 100) == pytest.approx(0.25)
    assert cb.per_1000([a], 0) is None


def test_negative_quantity_and_bad_source():  # CI-PL-C3
    with pytest.raises(ValueError):
        cb.CostLine("S3", "PUT", -1, "request", 0.0000047, "measured")
    with pytest.raises(ValueError):
        cb.CostLine("S3", "PUT", 1, "request", 0.0000047, "guess")


def test_runtime_lines_split_and_version():  # CI-PL-C5
    lines = cb.runtime_lines("inspector", vcpu_hours=0.5, gb_hours=10, platform_version="V1")
    assert [ln.usage_type for ln in lines] == ["vCPU-V1", "Memory-V1"]
    assert cb.total(lines) == pytest.approx(0.5 * 0.0895 + 10 * 0.00945)
    with pytest.raises(ValueError):
        cb.runtime_lines("inspector", 1, 1, "V3")
