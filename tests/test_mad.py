from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from tsdb_anomaly_task import MADDetector, Point, Series
from tsdb_anomaly_task.detectors.base import FluxContext, FluxUnsupportedError
from tsdb_anomaly_task.detectors.mad import NORMAL_CONSISTENCY, robust_scale

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def build(values: list[float], step_minutes: int = 5) -> Series:
    return Series(
        "temp",
        "c",
        {"sensor": "a"},
        tuple(Point(T0 + timedelta(minutes=step_minutes * i), v) for i, v in enumerate(values)),
    )


# -- the statistic itself ---------------------------------------------------


def test_normal_consistency_constant_matches_the_normal_quantile() -> None:
    from statistics import NormalDist

    assert pytest.approx(1.0 / NormalDist().inv_cdf(0.75), rel=1e-12) == NORMAL_CONSISTENCY


def test_scaled_mad_estimates_sigma_for_gaussian_data() -> None:
    sample = np.random.default_rng(7).normal(100.0, 5.0, 20000)
    _, scale = robust_scale(sample)
    assert scale == pytest.approx(5.0, rel=0.03)


def test_mad_resists_contamination_that_destroys_stddev() -> None:
    """The property the whole detector rests on: one bad sample must not move it."""
    clean = list(np.random.default_rng(3).normal(20.0, 1.0, 200))
    contaminated = [*clean, -9999.0]

    _, clean_scale = robust_scale(clean)
    _, dirty_scale = robust_scale(contaminated)
    assert dirty_scale == pytest.approx(clean_scale, rel=0.05)

    # The classical estimator moves by orders of magnitude on the same input.
    assert np.std(contaminated) > 50 * np.std(clean)


def test_robust_scale_of_empty_input() -> None:
    median, scale = robust_scale([])
    assert math.isnan(median)
    assert scale == 0.0


def test_robust_scale_ignores_non_finite_values() -> None:
    median, _ = robust_scale([1.0, 2.0, 3.0, float("nan"), float("inf")])
    assert median == 2.0


# -- construction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": 0}, "k must be positive"),
        ({"min_points": 2}, "min_points"),
        ({"window": "0s"}, "positive duration"),
    ],
)
def test_construction_validation(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MADDetector(**kwargs)


def test_describe_and_str() -> None:
    detector = MADDetector(k=3.5, window="1h")
    assert detector.describe() == {
        "detector": "mad",
        "k": 3.5,
        "window": "1h",
        "min_points": 8,
        "consecutive_points": 1,
    }
    assert str(detector) == "mad(k=3.5, window=1h)"
    assert str(MADDetector(k=3.0)) == "mad(k=3, window=all)"


# -- detection --------------------------------------------------------------


def test_flags_a_single_outlier() -> None:
    values = [20.0, 20.1, 19.9, 20.0, 20.2, 19.8, 20.1, 45.0, 20.0, 19.9]
    result = MADDetector(k=3.5, min_points=5).evaluate(build(values))
    assert [f.value for f in result.flags] == [45.0]
    assert result.flags[0].score > 3.5
    assert "robust z-score" in result.flags[0].reason


def test_leave_one_out_keeps_the_outlier_out_of_its_own_baseline() -> None:
    """A long excursion must not raise the scale that judges it.

    Half the window is at 30.0, so an inclusive baseline would treat 30.0 as
    ordinary.  Leave-one-out plus the median's 50% breakdown point still calls
    the shorter arm of the series the normal one.
    """
    values = [20.0, 20.1, 19.9, 20.0, 20.2, 19.8, 20.1, 20.0, 30.0, 20.05, 19.95]
    result = MADDetector(k=3.0, min_points=5).evaluate(build(values))
    assert [f.value for f in result.flags] == [30.0]


def test_flat_window_falls_back_to_the_whole_series_scale() -> None:
    """A locally flat window borrows the series-wide scale rather than skipping."""
    varied = [20.0, 22.0, 18.0, 21.0, 19.0, 23.0, 17.0, 24.0, 16.0, 21.5, 18.5, 22.5]
    values = varied + [20.0] * 5 + [26.0]
    result = MADDetector(k=2.5, window="30m", min_points=5).evaluate(build(values))
    assert 26.0 in [f.value for f in result.flags]


def test_entirely_constant_baseline_is_skipped_rather_than_guessed() -> None:
    """With no scale anywhere in the series there is nothing to measure against."""
    values = [20.0] * 8 + [24.0] + [20.0] * 8
    result = MADDetector(k=3.0, min_points=5).evaluate(build(values))
    assert len(result) == 0
    assert any("MAD is zero" in note for note in result.notes)


def test_perfectly_constant_series_is_skipped_not_flagged() -> None:
    result = MADDetector(k=3.0, min_points=5).evaluate(build([20.0] * 30))
    assert len(result) == 0
    assert not result.usable
    assert any("MAD is zero" in note for note in result.notes)


def test_rolling_window_skips_points_without_enough_history() -> None:
    values = [20.0 + 0.1 * (i % 3) for i in range(40)]
    result = MADDetector(k=3.0, window="30m", min_points=6).evaluate(build(values))
    assert any("min_points" in note for note in result.notes)
    assert result.points_evaluated < len(values)


def test_rolling_window_beats_a_global_window_on_a_level_shift() -> None:
    """After a step change the rolling baseline re-learns; a global one keeps alarming."""
    values = [20.0 + 0.05 * (i % 4) for i in range(60)] + [40.0 + 0.05 * (i % 4) for i in range(60)]
    rolling = MADDetector(k=4.0, window="1h", min_points=8).evaluate(build(values))
    globalw = MADDetector(k=4.0, min_points=8).evaluate(build(values))
    assert len(rolling) < len(globalw)


def test_empty_series() -> None:
    result = MADDetector().evaluate(Series("temp", "c"))
    assert not result.usable
    assert result.notes == ("series is empty",)


def test_non_finite_values_are_ignored() -> None:
    values = [20.0, 20.1, float("nan"), 20.0, 20.2, 19.8, 20.1, 45.0, 20.0, 19.9]
    result = MADDetector(k=3.5, min_points=5).evaluate(build(values))
    assert [f.value for f in result.flags] == [45.0]


def test_stats_are_reported() -> None:
    result = MADDetector(k=3.5, min_points=5).evaluate(build([20.0, 20.1, 19.9, 20.0, 20.2, 30.0]))
    assert result.stats["evaluated"] > 0
    assert result.stats["max_abs_score"] > 0


# -- ground truth -----------------------------------------------------------


def test_catches_every_injected_anomaly(spiky) -> None:
    detector = MADDetector(k=5.0, window="6h", min_points=24)
    result = detector.evaluate(spiky.series)
    times = [f.time for f in result.flags]
    assert spiky.caught(times) == len(spiky.anomalies)


def test_zero_false_positives_on_clean_data(pristine) -> None:
    detector = MADDetector(k=5.0, window="6h", min_points=24)
    result = detector.evaluate(pristine.series)
    assert len(result) == 0, [f.reason for f in result.flags]


def test_consecutive_points_reduces_the_flag_count(spiky) -> None:
    lenient = MADDetector(k=3.0, window="6h", min_points=24).evaluate(spiky.series)
    strict = MADDetector(k=3.0, window="6h", min_points=24, consecutive_points=3).evaluate(
        spiky.series
    )
    assert len(strict) < len(lenient)


# -- Flux -------------------------------------------------------------------


def context(**kwargs) -> FluxContext:
    kwargs.setdefault("query_flux", "data")
    kwargs.setdefault("flag_measurement", "anomaly")
    return FluxContext(**kwargs)


def test_simple_form_compiles_to_flux() -> None:
    flux = MADDetector(k=3.5).to_flux(context())
    assert "quantile(q: 0.5" in flux
    assert "findRecord" in flux
    assert "1.4826" in flux
    assert "math.abs(x: r._score) >= 3.5" in flux


def test_rolling_window_does_not_compile() -> None:
    support = MADDetector(k=3.0, window="1h").flux_support(context())
    assert not support
    assert "rolling window" in support.reason


def test_grouped_series_do_not_compile() -> None:
    support = MADDetector(k=3.0).flux_support(context(group_by=["host"]))
    assert not support
    assert "findRecord" in support.reason


def test_consecutive_points_do_not_compile() -> None:
    support = MADDetector(k=3.0, consecutive_points=2).flux_support(context())
    assert not support
    assert "stateCount" in support.reason


def test_unsupported_to_flux_raises_with_all_reasons() -> None:
    detector = MADDetector(k=3.0, window="1h", consecutive_points=2)
    with pytest.raises(FluxUnsupportedError) as excinfo:
        detector.to_flux(context(group_by=["host"]))
    message = str(excinfo.value)
    assert "rolling window" in message
    assert "findRecord" in message
    assert "stateCount" in message


def test_flux_notes_flag_the_semantic_difference() -> None:
    notes = MADDetector(k=3.0).flux_notes()
    assert any("whole task window" in note for note in notes)
