from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tsdb_anomaly_task import (
    DetectionResult,
    Flag,
    MetricQuery,
    Point,
    ResultsBucket,
    Schedule,
    Series,
    Severity,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


# -- Point / Series ---------------------------------------------------------


def test_naive_timestamps_become_utc() -> None:
    assert Point(datetime(2024, 1, 1), 1.0).time.tzinfo is UTC


def test_series_sorts_points_and_exposes_helpers() -> None:
    series = Series(
        "cpu",
        "usage",
        {"host": "a"},
        (Point(T0 + timedelta(minutes=5), 2.0), Point(T0, 1.0)),
    )
    assert series.values == (1.0, 2.0)
    assert series.start == T0
    assert series.end == T0 + timedelta(minutes=5)
    assert series.span() == timedelta(minutes=5)
    assert len(series) == 2
    assert [p.value for p in series] == [1.0, 2.0]
    assert series.key == "cpu.usage{host=a}"


def test_series_key_without_tags() -> None:
    assert Series("cpu", "usage").key == "cpu.usage"


def test_series_span_of_short_series() -> None:
    assert Series("cpu", "usage", {}, (Point(T0, 1.0),)).span() == timedelta(0)


def test_series_slice() -> None:
    points = tuple(Point(T0 + timedelta(minutes=i), float(i)) for i in range(10))
    series = Series("cpu", "usage", {}, points)
    sliced = series.slice(T0 + timedelta(minutes=3), T0 + timedelta(minutes=6))
    assert sliced.values == (3.0, 4.0, 5.0, 6.0)


# -- Severity ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.5, Severity.INFO),
        (1.0, Severity.INFO),
        (1.25, Severity.WARNING),
        (1.9, Severity.WARNING),
        (2.0, Severity.CRITICAL),
        (float("inf"), Severity.CRITICAL),
        (float("nan"), Severity.CRITICAL),
    ],
)
def test_severity_from_ratio(ratio: float, expected: Severity) -> None:
    assert Severity.from_ratio(ratio) is expected
    assert str(expected) == expected.value


# -- Flag -------------------------------------------------------------------


def test_flag_ratio_and_label() -> None:
    flag = Flag(
        time=T0,
        value=95.0,
        score=95.0,
        threshold=50.0,
        detector="threshold",
        reason="too high",
        tags={"host": "a", "region": "eu"},
    )
    assert flag.ratio == pytest.approx(1.9)
    assert flag.label == "host=a,region=eu"


def test_flag_ratio_with_zero_threshold() -> None:
    zero = Flag(T0, 1.0, 1.0, 0.0, "d", "r")
    assert zero.ratio == float("inf")
    assert Flag(T0, 0.0, 0.0, 0.0, "d", "r").ratio == 0.0


def test_flag_label_falls_back_to_series_key() -> None:
    assert Flag(T0, 1.0, 1.0, 1.0, "d", "r", series_key="cpu.usage").label == "cpu.usage"


def test_flag_line_protocol_round_trip() -> None:
    flag = Flag(
        time=T0,
        value=95.5,
        score=4.2,
        threshold=3.5,
        detector="mad",
        reason='spike, "hard"',
        severity=Severity.CRITICAL,
        tags={"host": "edge 01"},
    )
    line = flag.to_line_protocol("anomaly")
    assert line.startswith("anomaly,detector=mad,host=edge\\ 01,severity=critical ")
    assert 'reason="spike\\, \\"hard\\""' in line or 'reason="spike, \\"hard\\""' in line
    assert line.endswith(f" {int(T0.timestamp() * 1e9)}")


def test_flag_line_protocol_millisecond_precision() -> None:
    flag = Flag(T0, 1.0, 1.0, 1.0, "d", "r")
    assert flag.to_line_protocol("anomaly", precision_ns=False).endswith(
        str(int(T0.timestamp() * 1e3))
    )


# -- DetectionResult --------------------------------------------------------


def test_detection_result_rates() -> None:
    flag = Flag(T0, 1.0, 1.0, 1.0, "d", "r")
    result = DetectionResult("d", "k", (flag,), points_evaluated=10)
    assert len(result) == 1
    assert bool(result) is True
    assert list(result) == [flag]
    assert result.flag_rate == pytest.approx(0.1)
    assert DetectionResult("d", "k").flag_rate == 0.0


# -- MetricQuery ------------------------------------------------------------


def test_metric_query_to_flux_basic() -> None:
    flux = MetricQuery(bucket="b", measurement="m", field="f").to_flux()
    assert 'from(bucket: "b")' in flux
    assert "|> range(start: -1h)" in flux
    assert '|> filter(fn: (r) => r._measurement == "m")' in flux
    assert "|> group()" in flux


def test_metric_query_filters_and_grouping() -> None:
    flux = MetricQuery(
        bucket="b",
        measurement="m",
        field="f",
        filters={"host": "*", "region": ["eu", "us"], "role": "db"},
        group_by=["host", "region"],
        aggregate_window="1m",
        aggregate_fn="median",
        range_start="-6h",
        range_stop="now()",
    ).to_flux()
    assert "exists r.host" in flux
    assert '(r.region == "eu" or r.region == "us")' in flux or 'r.region == "eu" or' in flux
    assert 'r.role == "db"' in flux
    assert "aggregateWindow(every: 1m, fn: median" in flux
    assert '|> group(columns: ["host", "region"])' in flux
    assert "stop: now()" in flux


def test_metric_query_lookback() -> None:
    assert MetricQuery(bucket="b", measurement="m", field="f", range_start="-6h").lookback == (
        timedelta(hours=6)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bucket": "", "measurement": "m", "field": "f"},
        {"bucket": "b", "measurement": " ", "field": "f"},
        {"bucket": "b", "measurement": "m", "field": ""},
    ],
)
def test_metric_query_rejects_empty_names(kwargs) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        MetricQuery(**kwargs)


def test_metric_query_rejects_bad_aggregate() -> None:
    with pytest.raises(ValueError, match="aggregate_fn"):
        MetricQuery(
            bucket="b", measurement="m", field="f", aggregate_window="1m", aggregate_fn="p99"
        )


def test_metric_query_rejects_bad_tag_key() -> None:
    query = MetricQuery(bucket="b", measurement="m", field="f", filters={"bad key": "x"})
    with pytest.raises(ValueError, match="invalid tag key"):
        query.to_flux()


def test_metric_query_rejects_empty_filter_list() -> None:
    query = MetricQuery(bucket="b", measurement="m", field="f", filters={"host": []})
    with pytest.raises(ValueError, match="empty value list"):
        query.to_flux()


def test_metric_query_escapes_quotes() -> None:
    flux = MetricQuery(bucket='b"x', measurement="m", field="f").to_flux()
    assert 'from(bucket: "b\\"x")' in flux


# -- Schedule ---------------------------------------------------------------


def test_schedule_every_and_options() -> None:
    schedule = Schedule(every="5m", offset="30s")
    assert schedule.interval == timedelta(minutes=5)
    assert schedule.to_task_options("t") == 'option task = {name: "t", every: 5m, offset: 30s}'


def test_schedule_cron() -> None:
    schedule = Schedule(cron="0 * * * *")
    assert schedule.interval is None
    assert 'cron: "0 * * * *"' in schedule.to_task_options("t")


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"every": "5m", "cron": "0 * * * *"}],
)
def test_schedule_requires_exactly_one(kwargs) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Schedule(**kwargs)


def test_schedule_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        Schedule(every="0s")


def test_schedule_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="negative"):
        Schedule(every="5m", offset="-1s")


# -- ResultsBucket ----------------------------------------------------------


def test_results_bucket_to_flux() -> None:
    assert ResultsBucket("anomalies", org="acme").to_flux().strip() == (
        '|> to(bucket: "anomalies", org: "acme")'
    )
    assert ResultsBucket("anomalies").to_flux().strip() == '|> to(bucket: "anomalies")'


@pytest.mark.parametrize(
    "kwargs",
    [{"bucket": " "}, {"bucket": "b", "flag_measurement": " "}],
)
def test_results_bucket_validation(kwargs) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ResultsBucket(**kwargs)
