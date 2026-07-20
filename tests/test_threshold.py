from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tsdb_anomaly_task import Point, Series, Severity, ThresholdDetector
from tsdb_anomaly_task.detectors.base import FluxContext, FluxUnsupportedError

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def build(values: list[float]) -> Series:
    return Series(
        "cpu",
        "usage",
        {"host": "a"},
        tuple(Point(T0 + timedelta(minutes=i), v) for i, v in enumerate(values)),
    )


def flagged_indices(result) -> list[int]:
    return [int((f.time - T0).total_seconds() // 60) for f in result.flags]


# -- construction -----------------------------------------------------------


def test_requires_a_bound() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ThresholdDetector()


def test_rejects_inverted_band() -> None:
    with pytest.raises(ValueError, match="must be below"):
        ThresholdDetector(upper=10, lower=20)


def test_rejects_negative_hysteresis() -> None:
    with pytest.raises(ValueError, match="not be negative"):
        ThresholdDetector(upper=10, hysteresis=-1)


def test_rejects_overlapping_reset_points() -> None:
    with pytest.raises(ValueError, match="wider than half"):
        ThresholdDetector(upper=10, lower=0, hysteresis=6)


def test_rejects_zero_consecutive_points() -> None:
    with pytest.raises(ValueError, match="consecutive_points"):
        ThresholdDetector(upper=10, consecutive_points=0)


def test_describe_and_repr() -> None:
    detector = ThresholdDetector(upper=90, lower=10, hysteresis=2, consecutive_points=3)
    assert detector.describe()["upper"] == 90.0
    assert "upper=90.0" in repr(detector)
    assert str(detector) == "threshold(>= 10 and <= 90)"


# -- detection --------------------------------------------------------------


def test_flags_upper_and_lower_breaches() -> None:
    result = ThresholdDetector(upper=90, lower=10).evaluate(build([50, 95, 50, 5, 50]))
    assert flagged_indices(result) == [1, 3]
    assert result.flags[0].reason == "value 95 above upper limit 90"
    assert result.flags[1].reason == "value 5 below lower limit 10"


def test_severity_scales_with_distance() -> None:
    result = ThresholdDetector(upper=10).evaluate(build([11, 13, 25]))
    assert [f.severity for f in result.flags] == [
        Severity.INFO,
        Severity.WARNING,
        Severity.CRITICAL,
    ]


def test_no_flags_on_clean_data() -> None:
    result = ThresholdDetector(upper=90, lower=10).evaluate(build([50] * 20))
    assert len(result) == 0
    assert result.flag_rate == 0.0


def test_non_finite_values_are_skipped() -> None:
    result = ThresholdDetector(upper=90).evaluate(build([50, float("nan"), 95]))
    assert flagged_indices(result) == [2]


def test_empty_series_is_unusable_not_an_error() -> None:
    result = ThresholdDetector(upper=90).evaluate(Series("cpu", "usage"))
    assert not result.usable
    assert result.notes == ("series is empty; nothing to evaluate",)


# -- consecutive points -----------------------------------------------------


def test_consecutive_points_suppresses_lone_spikes() -> None:
    detector = ThresholdDetector(upper=90, consecutive_points=2)
    result = detector.evaluate(build([10, 10, 95, 10, 96, 97, 10]))
    assert flagged_indices(result) == [4, 5]


def test_consecutive_points_keeps_the_whole_run() -> None:
    detector = ThresholdDetector(upper=90, consecutive_points=3)
    result = detector.evaluate(build([10, 95, 96, 97, 98, 10]))
    assert flagged_indices(result) == [1, 2, 3, 4]


def test_consecutive_points_run_at_end_of_series() -> None:
    detector = ThresholdDetector(upper=90, consecutive_points=2)
    assert flagged_indices(detector.evaluate(build([10, 10, 95, 96]))) == [2, 3]


def test_consecutive_points_run_too_short_at_end() -> None:
    detector = ThresholdDetector(upper=90, consecutive_points=3)
    assert flagged_indices(detector.evaluate(build([10, 10, 95, 96]))) == []


# -- hysteresis -------------------------------------------------------------


def test_hysteresis_holds_the_alarm_through_dips_below_the_limit() -> None:
    # 88 and 89 are below the 90 limit but above the 85 reset point.
    values = [80, 95, 88, 89, 96, 80]
    plain = ThresholdDetector(upper=90).evaluate(build(values))
    latched = ThresholdDetector(upper=90, hysteresis=5).evaluate(build(values))
    assert flagged_indices(plain) == [1, 4]
    assert flagged_indices(latched) == [1, 2, 3, 4]


def test_hysteresis_clears_once_below_the_reset_point() -> None:
    result = ThresholdDetector(upper=90, hysteresis=5).evaluate(build([95, 84, 88]))
    assert flagged_indices(result) == [0]


def test_hysteresis_on_the_lower_bound() -> None:
    result = ThresholdDetector(lower=10, hysteresis=5).evaluate(build([5, 12, 14, 20]))
    assert flagged_indices(result) == [0, 1, 2]


def test_hysteresis_can_switch_directly_between_alarms() -> None:
    result = ThresholdDetector(upper=90, lower=10, hysteresis=2).evaluate(build([95, 5, 50]))
    assert flagged_indices(result) == [0, 1]
    assert result.flags[1].threshold == 10.0


def test_hysteresis_stops_chattering_across_a_noisy_crossing() -> None:
    noisy = [89, 91, 89, 91, 89, 91, 89]
    plain = ThresholdDetector(upper=90).evaluate(build(noisy))
    latched = ThresholdDetector(upper=90, hysteresis=3).evaluate(build(noisy))
    # Plain flags three separate excursions; hysteresis reports one.
    assert flagged_indices(plain) == [1, 3, 5]
    assert flagged_indices(latched) == [1, 2, 3, 4, 5, 6]


# -- ground truth against synthetic data ------------------------------------


def test_catches_injected_spikes_with_zero_false_positives(spiky, pristine) -> None:
    detector = ThresholdDetector(upper=25.0, lower=17.0)

    caught = detector.evaluate(spiky.series)
    assert spiky.caught([f.time for f in caught.flags]) == len(spiky.anomalies)
    assert spiky.false_positives([f.time for f in caught.flags]) == ()

    control = detector.evaluate(pristine.series)
    assert len(control) == 0


# -- Flux -------------------------------------------------------------------


def test_flux_support_blocked_by_hysteresis() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    support = ThresholdDetector(upper=90, hysteresis=1).flux_support(context)
    assert not support
    assert "hysteresis" in support.reason
    with pytest.raises(FluxUnsupportedError, match="hysteresis"):
        ThresholdDetector(upper=90, hysteresis=1).to_flux(context)


def test_flux_uses_state_count_for_consecutive_points() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    flux = ThresholdDetector(upper=90, consecutive_points=3).to_flux(context)
    assert 'stateCount(fn: (r) => r._value > 90.0, column: "_breachRun")' in flux
    assert "r._breachRun >= 3" in flux


def test_flux_single_point_uses_a_plain_filter() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    flux = ThresholdDetector(lower=10).to_flux(context)
    assert "|> filter(fn: (r) => r._value < 10.0)" in flux
    assert "stateCount" not in flux
    assert '_reason: "value below lower limit 10"' in flux


def test_flux_notes_mention_the_back_fill_difference() -> None:
    assert ThresholdDetector(upper=1, consecutive_points=2).flux_notes()
    assert ThresholdDetector(upper=1).flux_notes() == ()
