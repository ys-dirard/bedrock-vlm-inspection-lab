"""12/17 の記憶の事例の選び方と記録の文（CI-AC-M1〜M5）。AWS は呼ばない。"""

import pandas as pd
import pytest

from agentlab.memcases import (check_memory_cases, query_text, record_text, select_memory_cases)


def one_cls(n_test_anom=40, n_train_norm=40):
    rows = []
    for cat in ("capsules", "pcb1"):
        for i in range(n_test_anom):
            rows.append({"object": cat, "split": "test", "label": "anomaly",
                         "image": f"{cat}/Data/Images/Anomaly/{i:03d}.JPG", "mask": "m"})
        for i in range(n_train_norm):
            rows.append({"object": cat, "split": "train", "label": "normal",
                         "image": f"{cat}/Data/Images/Normal/{i:04d}.JPG", "mask": ""})
        rows.append({"object": cat, "split": "test", "label": "normal",
                     "image": f"{cat}/Data/Images/Normal/9999.JPG", "mask": ""})
    rows.append({"object": "candle", "split": "test", "label": "anomaly", "image": "candle/x/000.JPG", "mask": "m"})
    return pd.DataFrame(rows)


def ids(*xs):
    return pd.DataFrame({"image_id": list(xs)})


def select(**kw):
    base = dict(test200=ids("pcb1/anomaly/000", "capsules/anomaly/001"), refs=ids("pcb1/normal/0000"),
                calib=ids("pcb1/normal/0001"), eval_split=ids("capsules/anomaly/002"))
    base.update(kw)
    return select_memory_cases(one_cls(), **base)


def test_counts_sources_and_exclusions():  # CI-AC-M1, M2
    c = select()
    assert len(c) == 100 and set(c["category"]) == {"capsules", "pcb1"}
    assert c.groupby(["category", "label"]).size().eq(25).all()
    assert (c.loc[c.label == "anomaly", "split"] == "test").all()
    assert (c.loc[c.label == "normal", "split"] == "train").all()
    for x in ("pcb1/anomaly/000", "capsules/anomaly/001", "pcb1/normal/0000", "pcb1/normal/0001",
              "capsules/anomaly/002"):
        assert x not in set(c["image_id"])


def test_v20_is_subset_with_5_each_and_deterministic():  # CI-AC-M3
    a, b = select(), select()
    assert a.equals(b)
    assert a[a.in_v20].groupby(["category", "label"]).size().eq(5).all()


def test_not_enough_images_stops():
    with pytest.raises(ValueError, match="not enough"):
        select_memory_cases(one_cls(n_test_anom=24), ids(), ids(), ids(), ids())


def test_check_rejects_overlap():
    c = select()
    with pytest.raises(ValueError, match="overlap"):
        check_memory_cases(c, {c["image_id"].iloc[0]})


def test_record_and_query_text():  # CI-AC-M4, M5
    t = "異常度: どれよりも高い\n良品と違って見える領域: 1か所"
    r = record_text("pcb1", t, "anomaly")
    assert r.endswith("確定した判定: NG") and r.startswith("品目: pcb1\n")
    assert record_text("pcb1", t, "normal").endswith("確定した判定: OK")
    q = query_text("pcb1", t)
    assert "確定した判定" not in q and "NG" not in q and "OK" not in q
    with pytest.raises(ValueError):
        record_text("pcb1", t, "NG")
    with pytest.raises(ValueError):
        query_text("pcb1", "  ")


def records_frame():
    import pandas as pd
    rows = []
    for cat in ("capsules", "pcb1"):
        for label in ("anomaly", "normal"):
            for i in range(25):
                rows.append({"image_id": f"{cat}/{label}/{i:03d}", "category": cat, "label": label,
                             "record_text": f"品目: {cat}\nX\n確定した判定: {'NG' if label == 'anomaly' else 'OK'}",
                             "in_v20": i < 5})
    return pd.DataFrame(rows)


def test_memory_records_versions():  # CI-AC-M6
    from agentlab.memcases import memory_records
    df = records_frame()
    v100, v20 = memory_records(df, "v100"), memory_records(df, "v20")
    assert len(v100) == 100 and len(v20) == 20
    assert v100[0]["id"] == "v100-capsules-anomaly-000" and v100[0]["metadata"]["judgement"] == "NG"
    assert {r["metadata"]["image_id"] for r in v20} <= {r["metadata"]["image_id"] for r in v100}
    with pytest.raises(ValueError):
        memory_records(df.iloc[:99], "v100")
    with pytest.raises(ValueError):
        memory_records(df, "v50")


def test_memory_block():  # CI-AC-M8
    from agentlab.memcases import MEMORY_HEADER, memory_block
    assert memory_block([]) is None
    assert memory_block([{"content": {"text": "  "}}]) is None
    b = memory_block([{"content": {"text": "A"}, "score": 0.9}, {"content": {"text": "B"}, "score": 0.5}])
    assert b.startswith(MEMORY_HEADER) and "事例1:\nA" in b and "事例2:\nB" in b and "0.9" not in b
