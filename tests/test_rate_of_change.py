from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tsdb_anomaly_task import Point, RateOfChangeDetector, Series
from tsdb_anomaly_task.detectors.base import FluxContext, FluxUnsupportedError
from tsdb_anomaly_task.synthetic import Anomaly, make_series

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def build(values: list[float], step_seconds: int = 10) -> Series:
    return Series(
        "temp",
        "c",
        {"sensor": "a"},
        tuple(Point(T0 + timedelta(seconds=step_seconds * i), v) for i, v in enumerate(values)),
    )


# -- construction -----------------------------------------------------------


def test_requires_a_bound() -> None:
    with pytest.raises(ValueError, match="requires one of"):
        RateOfChangeDetector()


@pytest.mark.parametrize("bound", ["max_rate", "max_increase", "max_decrease"])
def test_bounds_must_be_positive(bound: str) -> None:
    with pytest.raises(ValueError, match=f"{bound} must be positive"):
        RateOfChangeDetector(**{bound: -1.0})


def test_per_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive duration"):
        RateOfChangeDetector(max_rate=1.0, per="0s")


def test_describe_and_str() -> None:
    detector = RateOfChangeDetector(max_rate=2.0, per="1m")
    assert detector.describe()["max_increase"] == 2.0
    assert str(detector) == "rate_of_change(max=2/1m)"


# -- detection --------------------------------------------------------------


def test_flags_the_glitch_and_the_recovery() -> None:
    detector = RateOfChangeDetector(max_rate=1.0, per="1s")
    result = detector.evaluate(build([20.0, 20.2, 20.1, 85.0, 20.3]))
    assert [round(f.score, 2) for f in result.flags] == [6.49, -6.47]
    assert "rise of +6.49 per 1s" in result.flags[0].reason
    assert "fall of" in result.flags[1].reason


def test_a_glitch_inside_the_threshold_band_is_still_caught() -> None:
    """The case a min/max check cannot see: a legal value reached illegally."""
    values = [20.0, 20.1, 20.0, 24.9, 20.1, 20.0]
    assert all(0.0 < v < 25.0 for v in values)  # never leaves the allowed band
    result = RateOfChangeDetector(max_rate=0.1, per="1s").evaluate(build(values))
    assert len(result) == 2


def test_asymmetric_bounds() -> None:
    detector = RateOfChangeDetector(max_increase=10.0, max_decrease=0.05, per="1s")
    result = detector.evaluate(build([100.0, 150.0, 149.0]))
    # The +5/s rise is allowed; the -0.1/s fall is not.
    assert [round(f.score, 3) for f in result.flags] == [-0.1]


def test_only_increase_bounded() -> None:
    detector = RateOfChangeDetector(max_increase=0.1, per="1s")
    result = detector.evaluate(build([20.0, 40.0, 10.0]))
    assert len(result) == 1
    assert result.flags[0].score > 0


def test_only_decrease_bounded() -> None:
    detector = RateOfChangeDetector(max_decrease=0.1, per="1s")
    result = detector.evaluate(build([20.0, 40.0, 10.0]))
    assert len(result) == 1
    assert result.flags[0].score < 0


def test_per_unit_scales_the_rate() -> None:
    per_second = RateOfChangeDetector(max_rate=1.0, per="1s").evaluate(build([0.0, 30.0]))
    per_minute = RateOfChangeDetector(max_rate=1.0, per="1m").evaluate(build([0.0, 30.0]))
    assert per_second.flags[0].score == pytest.approx(3.0)
    assert per_minute.flags[0].score == pytest.approx(180.0)


def test_clean_data_produces_nothing() -> None:
    result = RateOfChangeDetector(max_rate=1.0, per="1s").evaluate(build([20.0, 20.1, 20.05]))
    assert len(result) == 0


def test_too_short_to_differentiate() -> None:
    result = RateOfChangeDetector(max_rate=1.0).evaluate(build([20.0]))
    assert not result.usable
    assert "at least two points" in result.notes[0]


def test_duplicate_timestamps_are_skipped_not_divided_by_zero() -> None:
    series = Series("temp", "c", {}, (Point(T0, 20.0), Point(T0, 90.0), Point(T0, 20.0)))
    result = RateOfChangeDetector(max_rate=0.001, per="1s").evaluate(series)
    assert len(result) == 0
    assert "sharing a timestamp" in result.notes[0]


def test_min_interval_skips_bursty_pairs() -> None:
    detector = RateOfChangeDetector(max_rate=0.001, per="1s", min_interval="1m")
    result = detector.evaluate(build([20.0, 90.0, 20.0], step_seconds=10))
    assert len(result) == 0
    assert result.stats["evaluated"] == 0


def test_non_finite_values_are_ignored() -> None:
    result = RateOfChangeDetector(max_rate=1.0, per="1s").evaluate(
        build([20.0, float("nan"), 20.1])
    )
    assert len(result) == 0


def test_consecutive_points_requires_a_sustained_ramp() -> None:
    ramp = [20.0, 40.0, 40.1, 40.2, 40.3, 60.0, 80.0, 100.0]
    lone = RateOfChangeDetector(max_rate=1.0, per="1s").evaluate(build(ramp))
    sustained = RateOfChangeDetector(max_rate=1.0, per="1s", consecutive_points=3).evaluate(
        build(ramp)
    )
    assert len(lone) == 4  # one isolated jump plus a three-sample ramp
    assert len(sustained) == 3  # the isolated jump is suppressed


# -- ground truth -----------------------------------------------------------


def test_catches_injected_sensor_glitches() -> None:
    generated = make_series(
        count=288,
        interval="5m",
        base=20.0,
        noise=0.3,
        anomalies=[
            Anomaly(index=50, kind="spike", magnitude=60.0),
            Anomaly(index=180, kind="dip", magnitude=70.0),
        ],
        seed=17,
    )
    detector = RateOfChangeDetector(max_rate=2.0, per="1m")
    result = detector.evaluate(generated.series)
    assert generated.caught([f.time for f in result.flags]) == 2


def test_zero_false_positives_on_a_smooth_feed() -> None:
    generated = make_series(count=288, interval="5m", base=20.0, noise=0.3, seed=17)
    result = RateOfChangeDetector(max_rate=2.0, per="1m").evaluate(generated.series)
    assert len(result) == 0, [f.reason for f in result.flags]


# -- Flux -------------------------------------------------------------------


def test_compiles_to_flux() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    flux = RateOfChangeDetector(max_rate=1.5, per="1m").to_flux(context)
    assert "derivative(unit: 1m, nonNegative: false" in flux
    assert 'rename(columns: {_value: "_score"})' in flux
    assert "r._score > 1.5 or r._score < -1.5" in flux


def test_asymmetric_flux_threshold_expression() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    flux = RateOfChangeDetector(max_increase=2.0, max_decrease=0.5).to_flux(context)
    assert "if r._score > 0.0 then 2.0 else -0.5" in flux


def test_single_sided_flux_thresholds() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    assert "_threshold: 2.0" in RateOfChangeDetector(max_increase=2.0).to_flux(context)
    assert "_threshold: -0.5" in RateOfChangeDetector(max_decrease=0.5).to_flux(context)


def test_consecutive_points_block_flux() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    detector = RateOfChangeDetector(max_rate=1.0, consecutive_points=2)
    assert not detector.flux_support(context)
    with pytest.raises(FluxUnsupportedError, match="stateCount"):
        detector.to_flux(context)


def test_flux_notes() -> None:
    assert RateOfChangeDetector(max_rate=1.0).flux_notes()
