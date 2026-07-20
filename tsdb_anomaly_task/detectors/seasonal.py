"""Seasonal-profile detection: compare each point to its own time-of-day slot."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import numpy as np

from ..duration import format_duration, parse_duration
from ..models import Series
from .base import Candidate, Detector, FluxContext, FluxSupport, Judgement
from .mad import robust_scale

__all__ = ["BucketBaseline", "SeasonalDetector", "SeasonalPeriod"]


class SeasonalPeriod:
    """Named seasonal decompositions and their bucket-index functions."""

    HOUR_OF_DAY = "hour-of-day"
    DAY_OF_WEEK = "day-of-week"
    HOUR_OF_WEEK = "hour-of-week"

    SIZES: ClassVar[dict[str, int]] = {
        HOUR_OF_DAY: 24,
        DAY_OF_WEEK: 7,
        HOUR_OF_WEEK: 168,
    }

    @staticmethod
    def bucket_of(period: str, moment: datetime) -> int:
        """Return the seasonal bucket index for ``moment`` under ``period``."""
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        if period == SeasonalPeriod.HOUR_OF_DAY:
            return moment.hour
        if period == SeasonalPeriod.DAY_OF_WEEK:
            return moment.weekday()
        if period == SeasonalPeriod.HOUR_OF_WEEK:
            return moment.weekday() * 24 + moment.hour
        raise ValueError(f"unknown seasonal period {period!r}")

    @staticmethod
    def label(period: str, bucket: int) -> str:
        """Human-readable name for a bucket, used in flag reasons."""
        days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        if period == SeasonalPeriod.HOUR_OF_DAY:
            return f"{bucket:02d}:00"
        if period == SeasonalPeriod.DAY_OF_WEEK:
            return days[bucket]
        return f"{days[bucket // 24]} {bucket % 24:02d}:00"


@dataclass(frozen=True, slots=True)
class BucketBaseline:
    """The learned profile for one seasonal bucket."""

    bucket: int
    label: str
    count: int
    median: float
    scaled_mad: float

    @property
    def trained(self) -> bool:
        return self.count > 0 and math.isfinite(self.median)


class SeasonalDetector(Detector):
    """Flag deviation from a per-slot baseline learned from history.

    A flat threshold is wrong for anything with a daily rhythm.  Traffic that
    is perfectly normal at 14:00 is a serious incident at 04:00, and a single
    limit either misses the night-time anomaly or pages all afternoon.  This
    detector decomposes time into repeating buckets — hour-of-day, day-of-week
    or hour-of-week — and judges each point only against *other points from the
    same bucket*.

    Each bucket's baseline is a median plus a consistency-scaled MAD, computed
    leave-one-out over a trailing training window, so the profile stays robust
    to the very anomalies it is meant to find.  A point is flagged when its
    robust z-score against its own bucket exceeds ``k``.

    **Insufficient history is handled explicitly, never guessed at.**  A bucket
    with fewer than ``min_samples_per_bucket`` training points cannot support a
    baseline, so points landing in it are skipped and the reason is recorded in
    :attr:`~tsdb_anomaly_task.models.DetectionResult.notes`.  A result whose
    every point was skipped comes back with ``usable=False``.  That is
    deliberate: silently falling back to a global baseline would report
    confident nonsense for the first week of a new deployment.

    Args:
        period: One of ``"hour-of-day"``, ``"day-of-week"``, ``"hour-of-week"``.
        k: Robust z-score threshold against the bucket baseline.
        training: Trailing window from which each point's baseline is drawn.
        min_samples_per_bucket: Minimum training samples a bucket needs.  Five
            is a floor, not a recommendation: a robust scale built from five
            points is still noisy, so prefer a training window covering ten or
            more cycles and raise ``k`` when you cannot.
        consecutive_points: Adjacent breaches required before flagging.

    Note:
        Seasonal detection does not compile to Flux — see :meth:`flux_support`.
        Deploy it with the client-side runner instead.
    """

    name: ClassVar[str] = "seasonal"

    def __init__(
        self,
        *,
        period: str = SeasonalPeriod.HOUR_OF_DAY,
        k: float = 3.0,
        training: str | timedelta = "14d",
        min_samples_per_bucket: int = 5,
        consecutive_points: int = 1,
    ) -> None:
        super().__init__(consecutive_points=consecutive_points)
        if period not in SeasonalPeriod.SIZES:
            raise ValueError(
                f"unknown seasonal period {period!r}; "
                f"expected one of {sorted(SeasonalPeriod.SIZES)}"
            )
        if k <= 0:
            raise ValueError("k must be positive")
        if min_samples_per_bucket < 2:
            raise ValueError("min_samples_per_bucket must be >= 2")
        self.period = period
        self.k = float(k)
        self.training = training
        self.min_samples_per_bucket = int(min_samples_per_bucket)
        self._training_delta = parse_duration(training)
        if self._training_delta.total_seconds() <= 0:
            raise ValueError("training must be a positive duration")

    def describe(self) -> Mapping[str, object]:
        return {
            "detector": self.name,
            "period": self.period,
            "k": self.k,
            "training": format_duration(self.training),
            "min_samples_per_bucket": self.min_samples_per_bucket,
            "consecutive_points": self.consecutive_points,
        }

    @property
    def buckets(self) -> int:
        """Number of buckets in the configured period."""
        return SeasonalPeriod.SIZES[self.period]

    # -- profile -----------------------------------------------------------

    def profile(self, series: Series) -> list[BucketBaseline]:
        """Return the seasonal profile learned from the whole of ``series``.

        Useful for plotting the expected shape next to the observed data, and
        for checking coverage before trusting the detector in production.
        """
        by_bucket: dict[int, list[float]] = {}
        for point in series.points:
            if math.isfinite(point.value):
                bucket = SeasonalPeriod.bucket_of(self.period, point.time)
                by_bucket.setdefault(bucket, []).append(point.value)

        baselines: list[BucketBaseline] = []
        for bucket in range(self.buckets):
            values = by_bucket.get(bucket, [])
            if values:
                median, scale = robust_scale(values)
            else:
                median, scale = float("nan"), 0.0
            baselines.append(
                BucketBaseline(
                    bucket=bucket,
                    label=SeasonalPeriod.label(self.period, bucket),
                    count=len(values),
                    median=median,
                    scaled_mad=scale,
                )
            )
        return baselines

    def coverage(self, series: Series) -> float:
        """Fraction of buckets with enough training samples, in ``[0, 1]``."""
        trained = sum(1 for b in self.profile(series) if b.count >= self.min_samples_per_bucket)
        return trained / self.buckets

    # -- detection ---------------------------------------------------------

    def _judge(self, series: Series, now: datetime) -> Judgement:
        n = len(series.points)
        if n == 0:
            return Judgement(candidates=[], usable=False, notes=("series is empty",))

        times = np.array([p.time.timestamp() for p in series.points])
        values = np.array([p.value for p in series.points], dtype=float)
        buckets = np.array([SeasonalPeriod.bucket_of(self.period, p.time) for p in series.points])
        training_seconds = self._training_delta.total_seconds()

        candidates: list[Candidate | None] = [None] * n
        evaluated = 0
        thin_buckets: set[int] = set()
        flat_buckets: set[int] = set()

        for i in range(n):
            value = values[i]
            if not math.isfinite(value):
                continue
            bucket = int(buckets[i])
            mask = (
                (buckets == bucket)
                & (times >= times[i] - training_seconds)
                & (times <= times[i])
                & np.isfinite(values)
            )
            mask[i] = False
            sample = values[mask]
            if sample.size < self.min_samples_per_bucket:
                thin_buckets.add(bucket)
                continue

            median, scale = robust_scale(sample)
            if scale <= 0.0:
                flat_buckets.add(bucket)
                continue

            evaluated += 1
            score = (value - median) / scale
            if abs(score) >= self.k:
                label = SeasonalPeriod.label(self.period, bucket)
                direction = "above" if score > 0 else "below"
                candidates[i] = Candidate(
                    index=i,
                    time=series.points[i].time,
                    value=value,
                    score=score,
                    threshold=self.k if score > 0 else -self.k,
                    reason=(
                        f"{value:g} is {abs(score):.2f} scaled MADs {direction} the "
                        f"{label} baseline of {median:g} "
                        f"({sample.size} training samples)"
                    ),
                )

        notes: list[str] = []
        if thin_buckets:
            notes.append(
                f"insufficient history: {len(thin_buckets)} of {self.buckets} "
                f"{self.period} bucket(s) had fewer than "
                f"{self.min_samples_per_bucket} training samples within "
                f"{format_duration(self.training)}; points in those buckets were "
                f"skipped rather than judged against an unlearned baseline"
            )
        if flat_buckets:
            notes.append(
                f"{len(flat_buckets)} bucket(s) had a zero MAD (constant history) and were skipped"
            )
        if evaluated == 0:
            notes.append(
                "no point could be evaluated; extend the query range so it covers "
                "at least "
                f"{self.min_samples_per_bucket} full {self.period} cycles"
            )

        return Judgement(
            candidates=candidates,
            notes=tuple(notes),
            stats={
                "points": float(n),
                "evaluated": float(evaluated),
                "buckets": float(self.buckets),
                "coverage": self.coverage(series),
            },
            usable=evaluated > 0,
            points_evaluated=evaluated,
        )

    # -- Flux --------------------------------------------------------------

    def flux_support(self, context: FluxContext) -> FluxSupport:
        return FluxSupport(
            False,
            "seasonal detection needs a per-bucket median learned from a training "
            "window that is longer than the task's evaluation window, plus a join "
            "of each point back onto its own bucket baseline; Flux can express "
            "neither the windowed quantile nor the leave-one-out baseline, so this "
            "detector runs client-side",
        )

    def __str__(self) -> str:
        return f"seasonal({self.period}, k={self.k:g}, training={format_duration(self.training)})"
