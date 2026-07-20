"""Client-layer tests.  The real adapter is exercised against a stub, never a socket."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tsdb_anomaly_task import (
    FakeInfluxClient,
    Flag,
    InfluxClient,
    MetricQuery,
    Point,
    Series,
    Severity,
    WriteError,
)
from tsdb_anomaly_task.client import AsyncInfluxProtocol, InfluxProtocol, TaskRef

T0 = datetime(2024, 3, 4, tzinfo=UTC)


def series(host: str, values: list[float]) -> Series:
    return Series(
        "sensor",
        "temperature",
        {"host": host},
        tuple(Point(T0 + timedelta(minutes=5 * i), v) for i, v in enumerate(values)),
    )


def query(**kwargs) -> MetricQuery:
    kwargs.setdefault("bucket", "telemetry")
    kwargs.setdefault("measurement", "sensor")
    kwargs.setdefault("field", "temperature")
    return MetricQuery(**kwargs)


# -- protocol conformance ---------------------------------------------------


def test_fake_satisfies_both_protocols() -> None:
    fake = FakeInfluxClient()
    assert isinstance(fake, InfluxProtocol)
    assert isinstance(fake, AsyncInfluxProtocol)


def test_real_adapter_satisfies_both_protocols() -> None:
    adapter = InfluxClient(SimpleNamespace(org="acme"))
    assert isinstance(adapter, InfluxProtocol)
    assert isinstance(adapter, AsyncInfluxProtocol)


# -- FakeInfluxClient reads -------------------------------------------------


def test_read_matches_measurement_and_field() -> None:
    fake = FakeInfluxClient()
    fake.add_series("telemetry", series("a", [1.0, 2.0]))
    assert len(fake.read(query())) == 1
    assert fake.read(query(measurement="other")) == []
    assert fake.read(query(field="humidity")) == []
    assert fake.read(query(bucket="elsewhere")) == []


def test_read_records_the_queries_it_was_given() -> None:
    fake = FakeInfluxClient()
    fake.read(query())
    assert len(fake.reads) == 1


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, 2),
        ({"host": "*"}, 2),
        ({"host": "a"}, 1),
        ({"host": ["a", "b"]}, 2),
        ({"host": ["c"]}, 0),
        ({"region": "*"}, 0),
    ],
)
def test_read_applies_tag_filters(filters, expected: int) -> None:
    fake = FakeInfluxClient()
    fake.add_series("telemetry", series("a", [1.0]))
    fake.add_series("telemetry", series("b", [2.0]))
    assert len(fake.read(query(filters=filters))) == expected


def test_read_clips_to_the_range_when_data_is_recent() -> None:
    fake = FakeInfluxClient()
    fake.add_series("telemetry", series("a", [1.0, 2.0, 3.0, 4.0]))
    clipped = fake.read(query(range_start="-10m"), now=T0 + timedelta(minutes=15))
    assert clipped[0].values == (2.0, 3.0, 4.0)


def test_read_returns_everything_when_the_window_would_be_empty() -> None:
    """Historical fixtures must not silently produce an empty preview."""
    fake = FakeInfluxClient()
    fake.add_series("telemetry", series("a", [1.0, 2.0]))
    got = fake.read(query(range_start="-1h"), now=datetime(2030, 1, 1, tzinfo=UTC))
    assert got[0].values == (1.0, 2.0)


async def test_async_read_mirrors_the_sync_one() -> None:
    fake = FakeInfluxClient()
    fake.add_series("telemetry", series("a", [1.0]))
    assert await fake.aread(query()) == fake.read(query())


# -- FakeInfluxClient writes ------------------------------------------------


def flag(**kwargs) -> Flag:
    base = {
        "time": T0,
        "value": 1.0,
        "score": 2.0,
        "threshold": 1.0,
        "detector": "threshold",
        "reason": "r",
        "severity": Severity.WARNING,
        "tags": {"host": "a"},
    }
    base.update(kwargs)
    return Flag(**base)  # type: ignore[arg-type]


def test_write_flags_renders_line_protocol() -> None:
    fake = FakeInfluxClient()
    assert fake.write_flags("anomalies", "anomaly", [flag()]) == 1
    assert fake.written[0].startswith("anomaly,detector=threshold,host=a,severity=warning ")
    assert fake.writes == [("anomalies", "anomaly", 1)]


def test_write_flags_merges_extra_tags() -> None:
    fake = FakeInfluxClient()
    fake.write_flags("anomalies", "anomaly", [flag()], {"env": "prod"})
    assert "env=prod" in fake.written[0]


def test_injected_write_failures_are_consumed_one_at_a_time() -> None:
    fake = FakeInfluxClient(fail_writes=2)
    for _ in range(2):
        with pytest.raises(WriteError, match="injected failure"):
            fake.write_flags("anomalies", "anomaly", [flag()])
    assert fake.write_flags("anomalies", "anomaly", [flag()]) == 1


async def test_async_write_honours_the_delay() -> None:
    fake = FakeInfluxClient(write_delay=0.001)
    assert await fake.awrite_flags("anomalies", "anomaly", [flag()]) == 1


# -- FakeInfluxClient tasks -------------------------------------------------


def test_task_lifecycle() -> None:
    fake = FakeInfluxClient()
    assert fake.find_task("t") is None
    created = fake.upsert_task("t", "flux v1", every="5m")
    assert created.created and created.id == f"{1:016x}"

    updated = fake.upsert_task("t", "flux v2")
    assert updated.updated
    assert updated.id == created.id
    assert fake.find_task("t").flux == "flux v2"

    assert fake.delete_task("t") is True
    assert fake.delete_task("t") is False


def test_task_ids_are_unique() -> None:
    fake = FakeInfluxClient()
    assert fake.upsert_task("a", "x").id != fake.upsert_task("b", "y").id


# -- the real adapter, against a stub ---------------------------------------


class StubRecord:
    def __init__(self, values: dict) -> None:
        self.values = values

    def get_time(self):
        return self.values.get("_time")

    def get_value(self):
        return self.values.get("_value")


class StubTable:
    def __init__(self, records) -> None:
        self.records = records


class StubQueryApi:
    def __init__(self, tables) -> None:
        self.tables = tables
        self.queries: list[str] = []

    def query(self, flux, org=None):
        self.queries.append(flux)
        return self.tables


class StubWriteApi:
    def __init__(self, explode: bool = False) -> None:
        self.explode = explode
        self.records = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, bucket=None, org=None, record=None):
        if self.explode:
            raise ConnectionError("no route to host")
        self.records = record


class StubTask:
    def __init__(self, name: str, task_id: str = "abc") -> None:
        self.name = name
        self.id = task_id
        self.flux = "old"
        self.status = "active"


class StubTasksApi:
    def __init__(self, tasks=()) -> None:
        self.tasks = list(tasks)
        self.updated = None
        self.created = None
        self.deleted = None

    def find_tasks(self, name=None):
        return [t for t in self.tasks if t.name == name]

    def update_task(self, task):
        self.updated = task
        return task

    def create_task_with_script(self, name=None, flux=None, org_id=None):
        self.created = SimpleNamespace(id="new-id", name=name, flux=flux, status="active")
        return self.created

    def delete_task(self, task_id):
        self.deleted = task_id


class StubClient:
    org = "acme"

    def __init__(self, tables=(), tasks=(), explode_writes: bool = False) -> None:
        self._query = StubQueryApi(list(tables))
        self._write = StubWriteApi(explode_writes)
        self._tasks = StubTasksApi(tasks)

    def query_api(self):
        return self._query

    def write_api(self):
        return self._write

    def tasks_api(self):
        return self._tasks

    def organizations_api(self):
        return SimpleNamespace(find_organizations=lambda org=None: [SimpleNamespace(id="org-1")])


def test_adapter_requires_an_org() -> None:
    with pytest.raises(ValueError, match="org is required"):
        InfluxClient(SimpleNamespace())


def test_adapter_groups_records_by_tag_set() -> None:
    records = [
        StubRecord({"_time": T0, "_value": 1.0, "host": "a"}),
        StubRecord({"_time": T0 + timedelta(minutes=5), "_value": 2.0, "host": "a"}),
        StubRecord({"_time": T0, "_value": 3.0, "host": "b"}),
        StubRecord({"_time": None, "_value": 4.0, "host": "b"}),  # dropped
        StubRecord({"_time": T0, "_value": None, "host": "b"}),  # dropped
    ]
    stub = StubClient(tables=[StubTable(records)])
    got = InfluxClient(stub).read(query(group_by=["host"], filters={"host": "*"}))
    assert [s.tags["host"] for s in got] == ["a", "b"]
    assert got[0].values == (1.0, 2.0)
    assert got[1].values == (3.0,)
    assert 'from(bucket: "telemetry")' in stub._query.queries[0]


def test_adapter_writes_line_protocol() -> None:
    stub = StubClient()
    assert InfluxClient(stub).write_flags("anomalies", "anomaly", [flag()]) == 1
    assert stub._write.records[0].startswith("anomaly,")


def test_adapter_skips_empty_writes() -> None:
    stub = StubClient()
    assert InfluxClient(stub).write_flags("anomalies", "anomaly", []) == 0
    assert stub._write.records is None


def test_adapter_wraps_write_failures() -> None:
    stub = StubClient(explode_writes=True)
    with pytest.raises(WriteError, match="no route to host"):
        InfluxClient(stub).write_flags("anomalies", "anomaly", [flag()])


def test_adapter_creates_a_new_task() -> None:
    stub = StubClient()
    ref = InfluxClient(stub).upsert_task("t", "flux", every="5m")
    assert ref.created and ref.id == "new-id"
    assert stub._tasks.created.flux == "flux"


def test_adapter_updates_an_existing_task() -> None:
    stub = StubClient(tasks=[StubTask("t")])
    ref = InfluxClient(stub).upsert_task("t", "flux v2")
    assert ref.updated
    assert stub._tasks.updated.flux == "flux v2"


def test_adapter_finds_and_deletes() -> None:
    stub = StubClient(tasks=[StubTask("t")])
    adapter = InfluxClient(stub)
    assert adapter.find_task("t").id == "abc"
    assert adapter.find_task("missing") is None
    assert adapter.delete_task("t") is True
    assert stub._tasks.deleted == "abc"
    assert adapter.delete_task("missing") is False


def test_adapter_reports_a_missing_org() -> None:
    stub = StubClient()
    stub.organizations_api = lambda: SimpleNamespace(find_organizations=lambda org=None: [])
    with pytest.raises(ValueError, match="not found"):
        InfluxClient(stub).upsert_task("t", "flux")


async def test_adapter_async_methods_delegate_to_threads() -> None:
    stub = StubClient(tables=[StubTable([StubRecord({"_time": T0, "_value": 1.0})])])
    adapter = InfluxClient(stub)
    assert len(await adapter.aread(query())) == 1
    assert await adapter.awrite_flags("anomalies", "anomaly", [flag()]) == 1


# -- TaskRef ----------------------------------------------------------------


def test_task_ref_updated_flag() -> None:
    assert TaskRef("1", "t", "flux", created=True).updated is False
    assert TaskRef("1", "t", "flux", created=False).updated is True
