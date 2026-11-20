"""分析ツールの Lambda の入口（Athena は偽物に差し替える）。"""

from types import SimpleNamespace

import handler


class FakeAthena:
    def __init__(self, state="SUCCEEDED"):
        self.started, self.state = None, state

    def start_query_execution(self, **kw):
        self.started = kw
        return {"QueryExecutionId": "q1"}

    def get_query_execution(self, QueryExecutionId):
        return {"QueryExecution": {"Status": {"State": self.state, "StateChangeReason": "bad"},
                                   "Statistics": {"DataScannedInBytes": 10485760}}}

    def get_query_results(self, QueryExecutionId, MaxResults):
        return {"ResultSet": {"Rows": [
            {"Data": [{"VarCharValue": "run_name"}, {"VarCharValue": "n_calls"}]},
            {"Data": [{"VarCharValue": "p-nova"}, {"VarCharValue": "100"}]},
        ]}}


def ctx(tool):
    return SimpleNamespace(client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": f"analytics___{tool}"}))


def test_tool_name_and_parameters_passed(monkeypatch):
    fake = FakeAthena()
    monkeypatch.setattr(handler, "ATHENA", fake)
    out = handler.handler({"group_by": ["run_name"], "filters": {"label": "anomaly"}}, ctx("count_by"))
    assert out["rows"] == [{"run_name": "p-nova", "n_calls": "100"}]
    assert fake.started["ExecutionParameters"] == ["'anomaly'"] and fake.started["WorkGroup"] == handler.WORKGROUP
    assert "WHERE label = ?" in fake.started["QueryString"]  # 値は SQL に埋め込まない（指標の定義の 'anomaly' は固定の文字列）


def test_invalid_input_never_reaches_athena(monkeypatch):
    fake = FakeAthena()
    monkeypatch.setattr(handler, "ATHENA", fake)
    assert "error" in handler.handler({"group_by": ["path"]}, ctx("count_by"))
    assert "error" in handler.handler({"sql": "DROP TABLE x"}, ctx("run_sql"))
    assert fake.started is None


def test_failed_query_returns_error(monkeypatch):
    monkeypatch.setattr(handler, "ATHENA", FakeAthena(state="FAILED"))
    assert handler.handler({}, ctx("list_images"))["error"] == "bad"
