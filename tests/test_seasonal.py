from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from tsdb_anomaly_task import Point, SeasonalDetector, SeasonalPeriod, Series
from tsdb_anomaly_task.detectors.base import FluxContext, FluxUnsupportedError
from tsdb_anomaly_task.synthetic import Anomaly, make_series

MONDAY = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)


def daily(days: int, *, step_minutes: int = 60, amplitude: float = 10.0) -> Series:
    """A clean series with a strong 24-hour shape and almost no noise."""
    points = []
    steps = days * 24 * 60 // step_minutes
    for i in range(steps):
        moment = MONDAY + timedelta(minutes=step_minutes * i)
        seconds = moment.hour * 3600 + moment.minute * 60
        value = 50.0 + amplitude * math.sin(2 * math.pi * seconds / 86400.0)
        value += 0.05 * ((i * 7) % 5 - 2)  # tiny deterministic wobble
        points.append(Point(moment, value))
    return Series("traffic", "rps", {"site": "a"}, tuple(points))


# -- period arithmetic ------------------------------------------------------


def test_bucket_of_hour_of_day() -> None:
    moment = datetime(2024, 3, 6, 14, 30, tzinfo=UTC)
    assert SeasonalPeriod.bucket_of(SeasonalPeriod.HOUR_OF_DAY, moment) == 14
    assert SeasonalPeriod.bucket_of(SeasonalPeriod.DAY_OF_WEEK, moment) == 2
    assert SeasonalPeriod.bucket_of(SeasonalPeriod.HOUR_OF_WEEK, moment) == 2 * 24 + 14


def test_bucket_of_assumes_utc_for_naive_input() -> None:
    assert SeasonalPeriod.bucket_of("hour-of-day", datetime(2024, 3, 6, 9)) == 9


def test_bucket_of_rejects_unknown_period() -> None:
    with pytest.raises(ValueError, match="unknown seasonal period"):
        SeasonalPeriod.bucket_of("fortnight", MONDAY)


def test_labels() -> None:
    assert SeasonalPeriod.label("hour-of-day", 7) == "07:00"
    assert SeasonalPeriod.label("day-of-week", 0) == "Mon"
    assert SeasonalPeriod.label("hour-of-week", 26) == "Tue 02:00"


# -- construction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"period": "monthly"}, "unknown seasonal period"),
        ({"k": -1}, "k must be positive"),
        ({"min_samples_per_bucket": 1}, "min_samples_per_bucket"),
        ({"training": "0s"}, "positive duration"),
    ],
)
def test_construction_validation(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SeasonalDetector(**kwargs)


def test_describe_and_str() -> None:
    detector = SeasonalDetector(period="day-of-week", k=3.5, training="21d")
    assert detector.buckets == 7
    assert detector.describe()["period"] == "day-of-week"
    assert str(detector) == "seasonal(day-of-week, k=3.5, training=3w)"


# -- profile ----------------------------------------------------------------


def test_profile_learns_the_daily_shape() -> None:
    detector = SeasonalDetector(period="hour-of-day", training="7d")
    profile = detector.profile(daily(5))
    assert len(profile) == 24
    assert all(b.count == 5 for b in profile)
    peak = max(profile, key=lambda b: b.median)
    trough = min(profile, key=lambda b: b.median)
    assert peak.label == "06:00"  # sin peaks a quarter of the way through the day
    assert trough.label == "18:00"
    assert peak.median - trough.median == pytest.approx(20.0, abs=0.5)
    assert peak.trained


def test_profile_marks_untrained_buckets() -> None:
    detector = SeasonalDetector(period="hour-of-day", training="7d")
    short = daily(1, step_minutes=60)
    truncated = Series("traffic", "rps", {}, short.points[:6])
    profile = detector.profile(truncated)
    assert profile[10].count == 0
    assert not profile[10].trained
    assert math.isnan(profile[10].median)


def test_coverage() -> None:
    detector = SeasonalDetector(period="hour-of-day", training="7d", min_samples_per_bucket=3)
    assert detector.coverage(daily(4)) == 1.0
    assert detector.coverage(daily(1)) == 0.0


# -- insufficient history ---------------------------------------------------


def test_insufficient_history_skips_rather_than_guessing() -> None:
    detector = SeasonalDetector(period="hour-of-day", k=3.0, training="14d")
    result = detector.evaluate(daily(1))  # one sample per bucket: nothing to learn
    assert len(result) == 0
    assert not result.usable
    assert any("insufficient history" in note for note in result.notes)
    assert any("full hour-of-day cycles" in note for note in result.notes)


def test_partial_history_evaluates_the_buckets_it_can() -> None:
    detector = SeasonalDetector(
        period="hour-of-day", k=3.0, training="14d", min_samples_per_bucket=3
    )
    series = daily(5)
    result = detector.evaluate(series)
    assert result.usable
    assert result.points_evaluated > 0
    # The first three days cannot be judged; the note says so.
    assert result.points_evaluated < len(series)
    assert any("insufficient history" in note for note in result.notes)


def test_empty_series() -> None:
    result = SeasonalDetector().evaluate(Series("traffic", "rps"))
    assert not result.usable
    assert result.notes == ("series is empty",)


def test_constant_bucket_history_is_skipped() -> None:
    points = tuple(Point(MONDAY + timedelta(hours=i), 42.0) for i in range(24 * 6))
    result = SeasonalDetector(period="hour-of-day", training="14d").evaluate(
        Series("traffic", "rps", {}, points)
    )
    assert len(result) == 0
    assert any("zero MAD" in note for note in result.notes)


# -- detection --------------------------------------------------------------


def test_flags_a_night_time_value_that_a_flat_threshold_would_miss() -> None:
    """The point of seasonality: 55 at 18:00 is abnormal, 55 at 06:00 is not."""
    series = daily(6)
    points = list(series.points)
    night = next(i for i, p in enumerate(points) if p.time.hour == 18 and p.time.day == 9)
    points[night] = Point(points[night].time, 55.0)
    perturbed = Series("traffic", "rps", {"site": "a"}, tuple(points))

    result = SeasonalDetector(period="hour-of-day", k=3.0, training="14d").evaluate(perturbed)
    flagged = [f.time for f in result.flags]
    assert points[night].time in flagged
    # 55 sits comfortably inside the series' overall range, so it is not an
    # outlier in absolute terms at all.
    assert 40.0 < 55.0 < 60.0
    assert "18:00 baseline" in result.flags[0].reason


def test_clean_seasonal_data_produces_no_flags() -> None:
    result = SeasonalDetector(period="hour-of-day", k=3.0, training="14d").evaluate(daily(6))
    assert len(result) == 0


def test_catches_injected_anomalies_in_a_seasonal_series() -> None:
    generated = make_series(
        count=24 * 12,
        interval="1h",
        base=50.0,
        noise=0.6,
        daily_amplitude=10.0,
        anomalies=[
            Anomaly(index=24 * 9 + 3, kind="spike", magnitude=20.0),
            Anomaly(index=24 * 10 + 17, kind="dip", magnitude=22.0),
        ],
        seed=99,
    )
    detector = SeasonalDetector(period="hour-of-day", k=4.0, training="14d")
    result = detector.evaluate(generated.series)
    times = [f.time for f in result.flags]
    assert generated.caught(times) == 2


def test_zero_false_positives_on_clean_seasonal_data() -> None:
    generated = make_series(
        count=24 * 12,
        interval="1h",
        base=50.0,
        noise=0.6,
        daily_amplitude=10.0,
        anomalies=(),
        seed=99,
    )
    result = SeasonalDetector(
        period="hour-of-day", k=6.0, training="21d", min_samples_per_bucket=7
    ).evaluate(generated.series)
    assert len(result) == 0, [f.reason for f in result.flags]


def test_day_of_week_period() -> None:
    detector = SeasonalDetector(period="day-of-week", k=3.0, training="60d")
    assert detector.buckets == 7
    points = tuple(
        Point(MONDAY + timedelta(days=i), 10.0 + (i % 7) + 0.3 * ((i // 7) % 4)) for i in range(56)
    )
    result = detector.evaluate(Series("x", "y", {}, points))
    assert result.usable


# -- Flux -------------------------------------------------------------------


def test_seasonal_never_compiles_to_flux() -> None:
    context = FluxContext(query_flux="data", flag_measurement="anomaly")
    support = SeasonalDetector().flux_support(context)
    assert not support
    assert "client-side" in support.reason
    with pytest.raises(FluxUnsupportedError):
        SeasonalDetector().to_flux(context)
