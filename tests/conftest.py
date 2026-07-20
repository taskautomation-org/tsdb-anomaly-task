"""Shared fixtures.  No test in this suite touches the network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tsdb_anomaly_task import (
    AnomalyTask,
    FakeInfluxClient,
    MetricQuery,
    ResultsBucket,
    Schedule,
    ThresholdDetector,
)
from tsdb_anomaly_task.synthetic import DEFAULT_START, Anomaly, make_series

T0 = DEFAULT_START
NOW = DEFAULT_START + timedelta(days=2)


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture
def query() -> MetricQuery:
    return MetricQuery(
        bucket="telemetry",
        measurement="sensor",
        field="temperature",
        filters={"host": "*"},
        group_by=["host"],
        range_start="-48h",
    )


@pytest.fixture
def output() -> ResultsBucket:
    return ResultsBucket("anomalies", flag_measurement="anomaly")


@pytest.fixture
def schedule() -> Schedule:
    return Schedule(every="5m", offset="30s")


@pytest.fixture
def spiky():
    """A 2-day sensor series with three known excursions."""
    return make_series(
        measurement="sensor",
        field_name="temperature",
        tags={"host": "edge-01"},
        count=576,
        interval="5m",
        base=21.0,
        noise=0.35,
        anomalies=[
            Anomaly(index=140, kind="spike", magnitude=16.0),
            Anomaly(index=305, kind="spike", magnitude=18.0, length=4),
            Anomaly(index=470, kind="dip", magnitude=16.0, length=3),
        ],
        seed=11,
    )


@pytest.fixture
def pristine():
    """The false-positive control: same shape, no anomalies at all."""
    return make_series(
        measurement="sensor",
        field_name="temperature",
        tags={"host": "edge-01"},
        count=576,
        interval="5m",
        base=21.0,
        noise=0.35,
        anomalies=(),
        seed=11,
    )


@pytest.fixture
def client(spiky) -> FakeInfluxClient:
    fake = FakeInfluxClient()
    fake.add_synthetic("telemetry", spiky)
    return fake


@pytest.fixture
def clean_client(pristine) -> FakeInfluxClient:
    fake = FakeInfluxClient()
    fake.add_synthetic("telemetry", pristine)
    return fake


@pytest.fixture
def task(query, schedule, output) -> AnomalyTask:
    return AnomalyTask(
        name="temperature-threshold",
        query=query,
        detector=ThresholdDetector(upper=30.0, lower=12.0),
        schedule=schedule,
        output=output,
    )


def at(index: int, *, start: datetime = T0, step: timedelta = timedelta(minutes=5)) -> datetime:
    """Timestamp of sample ``index`` in the default synthetic grid."""
    return start + index * step


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)
