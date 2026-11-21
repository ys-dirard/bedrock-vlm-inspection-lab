"""評価の流し直し（CI-AC-S5・S6）。AWS は呼ばない。"""

import pytest

from agentlab.evalresume import evaluated, select_targets


def rec(i, run="r2", sid=True, err=None):
    return {"image_id": f"img{i}", "run_id": run, "runtime_session_id": f"s-{run}-{i}" if sid else None,
            "api_error": err}


def test_select_targets_by_run_id():  # CI-AC-S5
    recs = [rec(1, "r1"), rec(2, "r1"), rec(1, "r2"), rec(2, "r2"), rec(3, "r2", err="x"), rec(4, "r2", sid=False)]
    got = select_targets(recs, "r2")
    assert [r["runtime_session_id"] for r in got] == ["s-r2-1", "s-r2-2"]
    with pytest.raises(ValueError, match="duplicated"):
        select_targets(recs, None)


def test_evaluated_skips_only_successes():  # CI-AC-S6
    res = [{"evaluator": "e", "session_id": "a", "cost_usd": 0.005, "error": None},
           {"evaluator": "e", "session_id": "b", "cost_usd": 0.0, "error": "boom"},
           {"evaluator": "other", "session_id": "c", "cost_usd": 0.1, "error": None}]
    done, total, worst = evaluated(res, "e")
    assert done == {"a"} and total == pytest.approx(0.005) and worst == pytest.approx(0.005)
    assert evaluated([], "e") == (set(), 0.0, 0.0)
