from __future__ import annotations

from datetime import timedelta

import pytest

from tsdb_anomaly_task.synthetic import DEFAULT_START, Anomaly, clean_series, make_series


def test_generation_is_deterministic() -> None:
    a = make_series(count=50, seed=4)
    b = make_series(count=50, seed=4)
    assert a.series.values == b.series.values


def test_different_seeds_differ() -> None:
    assert (
        make_series(count=50, seed=1).series.values != make_series(count=50, seed=2).series.values
    )


def test_shape_and_grid() -> None:
    generated = make_series(count=10, interval="5m", start=DEFAULT_START)
    assert len(generated) == 10
    assert generated.series.start == DEFAULT_START
    assert generated.series.span() == timedelta(minutes=45)


def test_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="count must be"):
        make_series(count=0)


@pytest.mark.parametrize("kind", ["spike", "dip", "drift", "stuck"])
def test_each_anomaly_kind_moves_the_data(kind: str) -> None:
    baseline = make_series(count=60, seed=8)
    perturbed = make_series(
        count=60, seed=8, anomalies=[Anomaly(index=30, kind=kind, magnitude=20.0, length=3)]
    )
    assert baseline.series.values != perturbed.series.values
    assert perturbed.anomalous_indices == {30, 31, 32}


def test_spike_goes_up_and_dip_goes_down() -> None:
    up = make_series(count=40, seed=8, anomalies=[Anomaly(index=20, kind="spike", magnitude=20.0)])
    down = make_series(count=40, seed=8, anomalies=[Anomaly(index=20, kind="dip", magnitude=20.0)])
    assert up.series.values[20] > down.series.values[20]


def test_drift_ramps_rather_than_jumping() -> None:
    generated = make_series(
        count=40,
        noise=0.1,
        seed=8,
        anomalies=[Anomaly(index=10, kind="drift", magnitude=30.0, length=10)],
    )
    affected = generated.series.values[10:20]
    assert affected == tuple(sorted(affected))


def test_stuck_holds_the_previous_value() -> None:
    generated = make_series(count=20, seed=8, anomalies=[Anomaly(index=10, kind="stuck", length=4)])
    held = generated.series.values[10:14]
    assert len(set(held)) == 1
    assert held[0] == pytest.approx(generated.series.values[9])


def test_stuck_at_index_zero_holds_its_own_value() -> None:
    generated = make_series(count=10, seed=8, anomalies=[Anomaly(index=0, kind="stuck", length=3)])
    assert len(set(generated.series.values[:3])) == 1


def test_gap_removes_points() -> None:
    generated = make_series(count=50, anomalies=[Anomaly(index=20, kind="gap", length=10)])
    assert len(generated) == 40
    assert generated.anomalous_times == ()  # dropped points have no timestamp to flag


def test_daily_amplitude_and_trend() -> None:
    flat = make_series(count=288, interval="5m", noise=0.0)
    wavy = make_series(count=288, interval="5m", noise=0.0, daily_amplitude=5.0)
    rising = make_series(count=288, interval="5m", noise=0.0, trend_per_day=10.0)
    assert max(flat.series.values) - min(flat.series.values) < 1e-9
    assert max(wavy.series.values) - min(wavy.series.values) == pytest.approx(10.0, abs=0.2)
    assert rising.series.values[-1] > rising.series.values[0] + 9.0


def test_anomalies_beyond_the_end_are_ignored() -> None:
    generated = make_series(count=10, anomalies=[Anomaly(index=50, kind="spike")])
    assert generated.anomalous_indices == set()


def test_anomaly_validation() -> None:
    with pytest.raises(ValueError, match="unknown anomaly kind"):
        Anomaly(index=1, kind="wobble")
    with pytest.raises(ValueError, match="length"):
        Anomaly(index=1, length=0)
    with pytest.raises(ValueError, match="index"):
        Anomaly(index=-1)


# -- ground-truth accounting ------------------------------------------------


def test_caught_and_false_positives() -> None:
    generated = make_series(
        count=40,
        seed=8,
        anomalies=[Anomaly(index=10, kind="spike"), Anomaly(index=30, kind="dip")],
    )
    times = generated.series.times
    assert generated.caught([times[10]]) == 1
    assert generated.caught([times[10], times[30]]) == 2
    assert generated.caught([]) == 0
    assert generated.false_positives([times[5]]) == (times[5],)
    assert generated.false_positives([times[10]]) == ()


def test_normal_times_excludes_the_planted_ones() -> None:
    generated = make_series(count=20, anomalies=[Anomaly(index=5, kind="spike", length=2)])
    assert len(generated.normal_times) == 18


def test_clean_series_helper_drops_any_anomalies() -> None:
    generated = clean_series(count=20, anomalies=[Anomaly(index=5)])
    assert generated.anomalies == ()
    assert generated.anomalous_times == ()
