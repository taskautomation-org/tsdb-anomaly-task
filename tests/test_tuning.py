from __future__ import annotations

import pytest

from tsdb_anomaly_task import (
    AnomalyTask,
    DeadmanDetector,
    FakeInfluxClient,
    MADDetector,
    MetricQuery,
    ResultsBucket,
    Schedule,
    ThresholdDetector,
    sweep_parameter,
)
from tsdb_anomaly_task.tuning import SweepResult, SweepRow


def build(detector) -> AnomalyTask:
    return AnomalyTask(
        name="t",
        query=MetricQuery(
            bucket="telemetry",
            measurement="sensor",
            field="temperature",
            filters={"host": "*"},
            group_by=["host"],
            range_start="-48h",
        ),
        detector=detector,
        schedule=Schedule(every="5m"),
        output=ResultsBucket("anomalies"),
    )


# -- rows -------------------------------------------------------------------


def test_row_rates() -> None:
    row = SweepRow(value=3.0, flags=5, points=1000, series_flagged=1, max_score=9.0)
    assert row.flag_rate == pytest.approx(0.005)
    assert row.flags_per_1k == pytest.approx(5.0)
    assert SweepRow(3.0, 0, 0, 0, 0.0).flag_rate == 0.0


# -- sweeping ---------------------------------------------------------------


def test_sweep_is_monotonic_in_k(client, now) -> None:
    task = build(MADDetector(k=3.0, window="6h", min_points=24))
    result = sweep_parameter(task, client, parameter="k", values=[2.5, 3.0, 3.5, 4.0, 5.0], now=now)
    counts = [row.flags for row in result]
    assert counts == sorted(counts, reverse=True)
    assert len(result) == 5
    assert all(row.points > 0 for row in result)


def test_sweep_queries_once_however_many_values(client, now) -> None:
    task = build(MADDetector(k=3.0, window="6h", min_points=24))
    sweep_parameter(task, client, parameter="k", values=[2.0, 3.0, 4.0, 5.0, 6.0], now=now)
    assert len(client.reads) == 1


def test_sweep_does_not_mutate_the_original_detector(client, now) -> None:
    detector = MADDetector(k=3.0, window="6h", min_points=24)
    sweep_parameter(build(detector), client, parameter="k", values=[9.0], now=now)
    assert detector.k == 3.0


def test_sweep_a_threshold_bound(client, now) -> None:
    task = build(ThresholdDetector(upper=25.0))
    result = sweep_parameter(task, client, parameter="upper", values=[22.0, 25.0, 28.0], now=now)
    assert [row.flags for row in result] == sorted([row.flags for row in result], reverse=True)


def test_sweep_consecutive_points(client, now) -> None:
    task = build(ThresholdDetector(upper=24.0))
    result = sweep_parameter(
        task, client, parameter="consecutive_points", values=[1, 2, 4], now=now
    )
    assert result.rows[0].flags >= result.rows[-1].flags


def test_sweep_a_duration_parameter(client, now) -> None:
    task = build(DeadmanDetector(tolerance="30m"))
    result = sweep_parameter(
        task, client, parameter="tolerance", values=["10m", "1h", "6h"], now=now
    )
    assert [row.value for row in result] == ["10m", "1h", "6h"]


def test_sweep_rejects_unknown_parameters(client, now) -> None:
    with pytest.raises(AttributeError, match="has no parameter 'sigma'"):
        sweep_parameter(build(MADDetector()), client, parameter="sigma", values=[1.0], now=now)


def test_sweep_rejects_an_empty_value_list(client, now) -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        sweep_parameter(build(MADDetector()), client, parameter="k", values=[], now=now)


def test_sweep_of_an_empty_bucket(now) -> None:
    result = sweep_parameter(
        build(MADDetector()), FakeInfluxClient(), parameter="k", values=[3.0], now=now
    )
    assert result.rows[0].flags == 0
    assert result.rows[0].points == 0


# -- recommendation ---------------------------------------------------------


def test_target_rate_picks_the_first_value_that_meets_it(clean_client, now) -> None:
    task = build(MADDetector(k=3.0, window="6h", min_points=24))
    result = sweep_parameter(
        task,
        clean_client,
        parameter="k",
        values=[2.0, 3.0, 4.0, 6.0, 10.0],
        now=now,
        target_rate=0.0,
    )
    assert result.recommended.flags == 0
    assert result.recommended.value in (3.0, 4.0, 6.0, 10.0)


def test_target_rate_falls_back_to_the_last_row_when_unreachable(client, now) -> None:
    task = build(MADDetector(k=3.0, window="6h", min_points=24))
    result = sweep_parameter(
        task, client, parameter="k", values=[1.0, 1.5], now=now, target_rate=0.0
    )
    assert result.recommended is result.rows[-1]


def test_knee_is_returned_without_a_target(client, now) -> None:
    task = build(MADDetector(k=3.0, window="6h", min_points=24))
    result = sweep_parameter(task, client, parameter="k", values=[2.0, 3.0, 8.0], now=now)
    assert result.recommended is not None


def test_no_recommendation_without_rows() -> None:
    assert SweepResult(parameter="k", rows=()).recommended is None


# -- rendering --------------------------------------------------------------


def test_render_marks_the_recommendation(client, now) -> None:
    task = build(MADDetector(k=3.0, window="6h", min_points=24))
    text = sweep_parameter(task, client, parameter="k", values=[2.5, 4.0], now=now).render()
    assert text.startswith("sweep: k")
    assert "per 1k" in text
    assert text.count("<-") == 1
