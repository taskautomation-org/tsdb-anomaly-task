"""Rate-of-change detection: flag physically implausible jumps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import ClassVar

from ..duration import format_duration, parse_duration
from ..flux import flux_float
from ..models import Series
from .base import Candidate, Detector, FluxContext, FluxSupport, FluxUnsupportedError, Judgement

__all__ = ["RateOfChangeDetector"]


class RateOfChangeDetector(Detector):
    """Flag points whose first derivative exceeds a plausible bound.

    Physical quantities have inertia.  A room does not warm by 30 °C in one
    second, a water tank does not lose half its level between two reads, a
    flow meter does not go from 0 to 400 l/min instantaneously.  When the
    numbers say otherwise, the sensor is usually the thing that broke: a
    corrupted I²C read, an ADC glitch, a reset that zeroed a counter, a unit
    change in new firmware.

    Those glitches often stay comfortably *inside* the configured min/max
    limits, so a threshold check never sees them, and they are frequently too
    brief to move a MAD baseline.  Bounding the derivative catches them
    directly, and it is the cheapest sanity filter you can put in front of a
    downstream rollup — a single ``-9999`` sailing into an hourly mean is
    exactly the kind of contamination that makes people distrust a dashboard.

    The rate is computed between adjacent samples as
    ``(vᵢ - vᵢ₋₁) / Δt``, expressed per ``per`` (default: per second), and the
    flag is attached to the later of the two points.  Bounds may be symmetric
    (``max_rate``) or asymmetric (``max_increase`` / ``max_decrease``) — useful
    for counters that may legitimately jump up but should never fall, or for
    tank levels that drain fast and fill slowly.

    Args:
        max_rate: Symmetric bound on ``|rate|``.
        max_increase: Bound on positive rate; overrides ``max_rate`` upward.
        max_decrease: Bound on the magnitude of negative rate.
        per: Time unit the rates are expressed in (``"1s"``, ``"1m"`` …).
        min_interval: Ignore pairs closer together than this, which stops a
            duplicated timestamp from producing a near-infinite rate.
        consecutive_points: Adjacent breaches required before flagging.

    Example:
        >>> from datetime import datetime, timedelta, timezone
        >>> from tsdb_anomaly_task import Point, RateOfChangeDetector, Series
        >>> t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> vals = [20.0, 20.2, 20.1, 85.0, 20.3]
        >>> s = Series("temp", "c", {}, tuple(
        ...     Point(t0 + timedelta(seconds=10 * i), v) for i, v in enumerate(vals)))
        >>> det = RateOfChangeDetector(max_rate=1.0, per="1s")
        >>> [round(f.score, 2) for f in det.evaluate(s)]
        [6.49, -6.47]
    """

    name: ClassVar[str] = "rate_of_change"

    def __init__(
        self,
        *,
        max_rate: float | None = None,
        max_increase: float | None = None,
        max_decrease: float | None = None,
        per: str | timedelta = "1s",
        min_interval: str | timedelta | None = None,
        consecutive_points: int = 1,
    ) -> None:
        super().__init__(consecutive_points=consecutive_points)
        if max_rate is None and max_increase is None and max_decrease is None:
            raise ValueError(
                "RateOfChangeDetector requires one of 'max_rate', 'max_increase' or 'max_decrease'"
            )
        for label, value in (
            ("max_rate", max_rate),
            ("max_increase", max_increase),
            ("max_decrease", max_decrease),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive")

        self.max_rate = None if max_rate is None else float(max_rate)
        self.max_increase = float(max_increase) if max_increase is not None else self.max_rate
        self.max_decrease = float(max_decrease) if max_decrease is not None else self.max_rate
        self.per = per
        self.min_interval = min_interval
        self._per_seconds = parse_duration(per).total_seconds()
        if self._per_seconds <= 0:
            raise ValueError("per must be a positive duration")
        self._min_interval_seconds = (
            parse_duration(min_interval).total_seconds() if min_interval is not None else 0.0
        )

    def describe(self) -> Mapping[str, object]:
        return {
            "detector": self.name,
            "max_rate": self.max_rate,
            "max_increase": self.max_increase,
            "max_decrease": self.max_decrease,
            "per": format_duration(self.per),
            "consecutive_points": self.consecutive_points,
        }

    # -- detection ---------------------------------------------------------

    def _judge(self, series: Series, now: datetime) -> Judgement:
        points = series.points
        n = len(points)
        candidates: list[Candidate | None] = [None] * n
        if n < 2:
            return Judgement(
                candidates=candidates,
                usable=False,
                notes=("need at least two points to compute a rate of change",),
                stats={"points": float(n)},
                points_evaluated=0,
            )

        unit = format_duration(self.per)
        evaluated = 0
        skipped_close = 0
        max_seen = 0.0

        for i in range(1, n):
            previous, current = points[i - 1], points[i]
            if not (math.isfinite(previous.value) and math.isfinite(current.value)):
                continue
            delta_t = (current.time - previous.time).total_seconds()
            if delta_t <= 0 or delta_t < self._min_interval_seconds:
                skipped_close += 1
                continue

            rate = (current.value - previous.value) / delta_t * self._per_seconds
            evaluated += 1
            max_seen = max(max_seen, abs(rate))

            limit: float | None = None
            if rate > 0 and self.max_increase is not None and rate > self.max_increase:
                limit = self.max_increase
            elif rate < 0 and self.max_decrease is not None and -rate > self.max_decrease:
                limit = -self.max_decrease
            if limit is None:
                continue

            direction = "rise" if rate > 0 else "fall"
            candidates[i] = Candidate(
                index=i,
                time=current.time,
                value=current.value,
                score=rate,
                threshold=limit,
                reason=(
                    f"{direction} of {rate:+.4g} per {unit} exceeds the "
                    f"{abs(limit):g} per {unit} bound "
                    f"({previous.value:g} → {current.value:g} in "
                    f"{format_duration(timedelta(seconds=delta_t))})"
                ),
            )

        notes: list[str] = []
        if skipped_close:
            notes.append(
                f"{skipped_close} pair(s) skipped: samples closer together than "
                f"min_interval, or sharing a timestamp"
            )

        return Judgement(
            candidates=candidates,
            notes=tuple(notes),
            stats={
                "points": float(n),
                "evaluated": float(evaluated),
                "max_abs_rate": max_seen,
            },
            usable=evaluated > 0,
            points_evaluated=evaluated,
        )

    # -- Flux --------------------------------------------------------------

    def flux_support(self, context: FluxContext) -> FluxSupport:
        if self.consecutive_points > 1:
            return FluxSupport(
                False,
                "consecutive-point de-bouncing of a derivative column cannot be "
                "expressed with stateCount over a mapped value",
            )
        return FluxSupport(True)

    def to_flux(self, context: FluxContext) -> str:
        support = self.flux_support(context)
        if not support:
            raise FluxUnsupportedError(support.reason)

        indent = context.indent
        unit = format_duration(self.per)
        tests: list[str] = []
        if self.max_increase is not None:
            tests.append(f"r._score > {flux_float(self.max_increase)}")
        if self.max_decrease is not None:
            tests.append(f"r._score < {flux_float(-self.max_decrease)}")
        breach = " or ".join(tests)

        if self.max_increase is not None and self.max_decrease is not None:
            threshold = (
                f"if r._score > 0.0 then {flux_float(self.max_increase)} "
                f"else {flux_float(-self.max_decrease)}"
            )
        elif self.max_increase is not None:
            threshold = flux_float(self.max_increase)
        else:
            assert self.max_decrease is not None
            threshold = flux_float(-self.max_decrease)

        return "\n".join(
            [
                f"originals = {context.query_flux}",
                "",
                "originals",
                f"{indent}|> derivative(unit: {unit}, nonNegative: false, "
                'columns: ["_value"], timeColumn: "_time")',
                f'{indent}|> rename(columns: {{_value: "_score"}})',
                f"{indent}|> filter(fn: (r) => {breach})",
                f"{indent}|> map(fn: (r) => ({{ r with",
                f"{indent}    _value: r._score,",
                f"{indent}    _threshold: {threshold},",
                f'{indent}    _reason: "rate of change exceeds the configured bound per {unit}",',
                f"{indent}}}))",
            ]
        )

    def flux_notes(self) -> tuple[str, ...]:
        return (
            "derivative() reports the rate at the later of each sample pair, matching"
            " the client-side runner.",
        )

    def __str__(self) -> str:
        unit = format_duration(self.per)
        return f"rate_of_change(max={self.max_rate or self.max_increase:g}/{unit})"
