from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tsdb_anomaly_task import DeadmanDetector, Point, Series, Severity
from tsdb_anomaly_task.detectors.base import FluxContext, FluxUnsupportedError
from tsdb_anomaly_task.synthetic import Anomaly, make_series

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def reporting(minutes: list[int]) -> Series:
    return Series(
        "temp",
        "c",
        {"sensor": "a"},
        tuple(Point(T0 + timedelta(minutes=m), 20.0 + m / 100) for m in minutes),
    )


# -- construction -----------------------------------------------------------


def test_rejects_consecutive_points() -> None:
    with pytest.raises(ValueError, match="does not support consecutive_points"):
        DeadmanDetector(tolerance="5m", consecutive_points=2)


def test_rejects_nonpositive_tolerance() -> None:
    with pytest.raises(ValueError, match="positive duration"):
        DeadmanDetector(tolerance="0s")


def test_describe_and_str() -> None:
    detector = DeadmanDetector(tolerance="10m", flag_gaps=True)
    assert detector.describe()["tolerance"] == "10m"
    assert detector.tolerance_seconds == 600.0
    assert str(detector) == "deadman(tolerance=10m)"


# -- trailing silence -------------------------------------------------------


def test_flags_trailing_silence() -> None:
    series = reporting([0, 1, 2])
    result = DeadmanDetector(tolerance="5m").evaluate(series, now=T0 + timedelta(minutes=32))
    assert len(result) == 1
    flag = result.flags[0]
    assert flag.time == T0 + timedelta(minutes=2)
    assert flag.score == pytest.approx(1800.0)
    assert flag.threshold == 300.0
    assert flag.reason == "no data for 30m (tolerance 5m)"
    assert flag.severity is Severity.CRITICAL


def test_silence_within_tolerance_is_quiet() -> None:
    series = reporting([0, 1, 2])
    result = DeadmanDetector(tolerance="10m").evaluate(series, now=T0 + timedelta(minutes=6))
    assert len(result) == 0
    assert result.usable


def test_defaults_to_now_when_no_reference_given() -> None:
    """A series from 2024 is very silent as of today."""
    result = DeadmanDetector(tolerance="1h").evaluate(reporting([0, 1]))
    assert len(result) == 1


# -- interior gaps ----------------------------------------------------------


def test_interior_gaps_are_off_by_default() -> None:
    series = reporting([0, 1, 2, 90, 91, 92])
    result = DeadmanDetector(tolerance="10m").evaluate(series, now=T0 + timedelta(minutes=93))
    assert len(result) == 0


def test_interior_gaps_when_enabled() -> None:
    series = reporting([0, 1, 2, 90, 91, 92])
    result = DeadmanDetector(tolerance="10m", flag_gaps=True).evaluate(
        series, now=T0 + timedelta(minutes=93)
    )
    assert len(result) == 1
    assert result.flags[0].time == T0 + timedelta(minutes=90)
    assert "reporting gap of 1h28m" in result.flags[0].reason


def test_gap_and_trailing_silence_together() -> None:
    series = reporting([0, 1, 2, 90, 91, 92])
    result = DeadmanDetector(tolerance="10m", flag_gaps=True).evaluate(
        series, now=T0 + timedelta(minutes=200)
    )
    assert len(result) == 2
    assert [f.time for f in result.flags] == [
        T0 + timedelta(minutes=90),
        T0 + timedelta(minutes=92),
    ]


def test_stats_report_the_largest_gap() -> None:
    series = reporting([0, 1, 2, 90])
    result = DeadmanDetector(tolerance="10m").evaluate(series, now=T0 + timedelta(minutes=91))
    assert result.stats["max_gap_seconds"] == pytest.approx(88 * 60)
    assert result.stats["tolerance_seconds"] == 600.0


# -- edge cases -------------------------------------------------------------


def test_empty_series_reports_why_it_cannot_help() -> None:
    result = DeadmanDetector(tolerance="5m").evaluate(Series("temp", "c"), now=T0)
    assert len(result) == 0
    assert not result.usable
    assert "registry of expected series" in result.notes[0]


def test_naive_now_is_treated_as_utc() -> None:
    result = DeadmanDetector(tolerance="5m").evaluate(
        reporting([0]), now=datetime(2024, 1, 1, 1, 0)
    )
    assert len(result) == 1


# -- ground truth -----------------------------------------------------------


def test_detects_the_injected_dropout() -> None:
    generated = make_series(
        count=288,
        interval="5m",
        anomalies=[Anomaly(index=100, kind="gap", length=36)],  # 3 hours of silence
        seed=5,
    )
    end = generated.series.end
    result = DeadmanDetector(tolerance="30m", flag_gaps=True).evaluate(
        generated.series, now=end + timedelta(minutes=5)
    )
    assert len(result) == 1
    assert result.flags[0].score == pytest.approx(37 * 300.0)


def test_no_false_positives_on_a_healthy_feed() -> None:
    generated = make_series(count=288, interval="5m", anomalies=(), seed=5)
    end = generated.series.end
    result = DeadmanDetector(tolerance="30m", flag_gaps=True).evaluate(
        generated.series, now=end + timedelta(minutes=5)
    )
    assert len(result) == 0


# -- Flux -------------------------------------------------------------------


def test_trailing_silence_compiles_to_flux() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    flux = DeadmanDetector(tolerance="10m").to_flux(context)
    assert "|> last()" in flux
    assert "int(v: now()) - int(v: r._time)" in flux
    assert "r._score > 600.0" in flux


def test_gap_detection_blocks_flux_compilation() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    detector = DeadmanDetector(tolerance="10m", flag_gaps=True)
    support = detector.flux_support(context)
    assert not support
    assert "flag_gaps=False" in support.reason
    with pytest.raises(FluxUnsupportedError):
        detector.to_flux(context)


def test_flux_notes_warn_about_vanished_series() -> None:
    notes = DeadmanDetector(tolerance="10m").flux_notes()
    assert any("disappears entirely" in note for note in notes)
