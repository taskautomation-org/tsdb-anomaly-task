"""Deterministic synthetic sensor data with *known* injected anomalies.

The whole test suite, the demo and the chart run on this module.  Because the
generator records exactly which samples it corrupted, tests can assert both
halves of the quality question: did the detector catch the anomalies that were
planted, and did it stay silent everywhere else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from .duration import parse_duration
from .models import Point, Series

__all__ = ["Anomaly", "SyntheticSeries", "clean_series", "make_series"]

DEFAULT_START = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)  # a Monday, 00:00 UTC


@dataclass(frozen=True, slots=True)
class Anomaly:
    """An anomaly to inject at a known sample index.

    Args:
        index: Index of the first affected sample.
        kind: One of ``spike``, ``dip``, ``drift``, ``stuck``, ``gap``.
        magnitude: Size of the excursion, in multiples of the noise sigma for
            ``spike``/``dip``/``drift``, and ignored for ``stuck``/``gap``.
        length: Number of consecutive samples affected.
    """

    index: int
    kind: str = "spike"
    magnitude: float = 12.0
    length: int = 1

    VALID = ("spike", "dip", "drift", "stuck", "gap")

    def __post_init__(self) -> None:
        if self.kind not in Anomaly.VALID:
            raise ValueError(f"unknown anomaly kind {self.kind!r}; expected one of {Anomaly.VALID}")
        if self.length < 1:
            raise ValueError("anomaly length must be >= 1")
        if self.index < 0:
            raise ValueError("anomaly index must be >= 0")

    def indices(self) -> range:
        return range(self.index, self.index + self.length)


@dataclass(frozen=True, slots=True)
class SyntheticSeries:
    """A generated series plus the ground truth about what was injected."""

    series: Series
    anomalies: tuple[Anomaly, ...] = ()
    anomalous_indices: frozenset[int] = frozenset()
    anomalous_times: tuple[datetime, ...] = ()
    clean_values: tuple[float, ...] = ()

    def __len__(self) -> int:
        return len(self.series)

    @property
    def normal_times(self) -> tuple[datetime, ...]:
        planted = set(self.anomalous_times)
        return tuple(p.time for p in self.series.points if p.time not in planted)

    def caught(self, flagged_times: Sequence[datetime]) -> int:
        """How many injected anomalies have at least one flagged sample."""
        flagged = set(flagged_times)
        times = self.series.times
        hits = 0
        for anomaly in self.anomalies:
            window = {times[i] for i in anomaly.indices() if i < len(times)}
            if window & flagged:
                hits += 1
        return hits

    def false_positives(self, flagged_times: Sequence[datetime]) -> tuple[datetime, ...]:
        """Flagged samples that were not part of any injected anomaly."""
        planted = set(self.anomalous_times)
        return tuple(t for t in flagged_times if t not in planted)


def make_series(
    *,
    measurement: str = "sensor",
    field_name: str = "value",
    tags: dict[str, str] | None = None,
    start: datetime = DEFAULT_START,
    count: int = 480,
    interval: str | timedelta = "5m",
    base: float = 20.0,
    noise: float = 0.4,
    daily_amplitude: float = 0.0,
    trend_per_day: float = 0.0,
    anomalies: Sequence[Anomaly] = (),
    seed: int = 20240304,
) -> SyntheticSeries:
    """Generate a reproducible sensor series with injected anomalies.

    The clean signal is ``base`` plus optional daily seasonality and linear
    trend, perturbed by Gaussian noise of standard deviation ``noise``.
    Anomalies are then stamped on top at known indices.

    Args:
        count: Number of samples.
        interval: Spacing between samples.
        base: Mean level of the clean signal.
        noise: Standard deviation of the Gaussian noise.
        daily_amplitude: Peak-to-mean amplitude of a 24-hour sinusoid.
        trend_per_day: Linear drift added per day.
        anomalies: Anomalies to inject.
        seed: RNG seed; the same seed always produces the same series.

    Returns:
        A :class:`SyntheticSeries` carrying both the data and the ground truth.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = np.random.default_rng(seed)
    step = parse_duration(interval)
    step_seconds = step.total_seconds()

    times = [start + i * step for i in range(count)]
    values = np.full(count, float(base))
    if daily_amplitude:
        for i, moment in enumerate(times):
            seconds = moment.hour * 3600 + moment.minute * 60 + moment.second
            values[i] += daily_amplitude * math.sin(2 * math.pi * seconds / 86400.0)
    if trend_per_day:
        for i in range(count):
            values[i] += trend_per_day * (i * step_seconds) / 86400.0
    values += rng.normal(0.0, noise, count)

    clean = tuple(float(v) for v in values)
    affected: set[int] = set()
    dropped: set[int] = set()

    for anomaly in anomalies:
        idx = [i for i in anomaly.indices() if i < count]
        if not idx:
            continue
        if anomaly.kind == "spike":
            for i in idx:
                values[i] += anomaly.magnitude * max(noise, 1e-9)
        elif anomaly.kind == "dip":
            for i in idx:
                values[i] -= anomaly.magnitude * max(noise, 1e-9)
        elif anomaly.kind == "drift":
            span = len(idx)
            for offset, i in enumerate(idx):
                values[i] += anomaly.magnitude * max(noise, 1e-9) * (offset + 1) / span
        elif anomaly.kind == "stuck":
            held = values[idx[0] - 1] if idx[0] > 0 else values[idx[0]]
            for i in idx:
                values[i] = held
        elif anomaly.kind == "gap":
            dropped.update(idx)
        affected.update(idx)

    points = tuple(Point(times[i], float(values[i])) for i in range(count) if i not in dropped)
    series = Series(
        measurement=measurement,
        field=field_name,
        tags=tags or {},
        points=points,
    )
    anomalous_times = tuple(times[i] for i in sorted(affected) if i not in dropped)
    return SyntheticSeries(
        series=series,
        anomalies=tuple(anomalies),
        anomalous_indices=frozenset(affected),
        anomalous_times=anomalous_times,
        clean_values=clean,
    )


def clean_series(**kwargs: object) -> SyntheticSeries:
    """Generate a series with no anomalies at all — the false-positive control."""
    kwargs.pop("anomalies", None)
    return make_series(anomalies=(), **kwargs)  # type: ignore[arg-type]
