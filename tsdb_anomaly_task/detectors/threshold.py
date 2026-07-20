"""Static threshold detection with hysteresis and run-length de-bouncing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import ClassVar

from ..flux import escape_flux_string, flux_float
from ..models import Series
from .base import Candidate, Detector, FluxContext, FluxSupport, FluxUnsupportedError, Judgement

__all__ = ["ThresholdDetector"]


class ThresholdDetector(Detector):
    """Flag points outside a fixed band, with hysteresis to stop chattering.

    A naive ``value > limit`` check is the most common alert in time-series
    monitoring and also the most common source of alert fatigue: a signal that
    hovers around the limit crosses it dozens of times an hour, and each
    crossing is a page.  Two mechanisms fix that, and this detector implements
    both.

    **Hysteresis** splits the single limit into a trip point and a *reset*
    point.  Once ``value > upper`` the series is considered in-alarm and stays
    in-alarm until it falls back below ``upper - hysteresis``.  A signal
    oscillating inside the hysteresis band therefore produces one continuous
    excursion rather than a burst of separate ones.

    **``consecutive_points``** requires N adjacent samples to breach before any
    of them is reported, which discards the single-sample spikes that dominate
    real sensor feeds.

    Choosing the limits themselves is a data question, not a code question —
    and on skewed, noisy sensor data the mean-based limits people reach for
    first are usually the wrong ones.  See
    `Choosing mean vs median for noisy sensor rollups
    <https://taskautomation.org/downsampling-aggregation-pipeline-design/threshold-tuning-for-aggregation/choosing-mean-vs-median-for-noisy-sensor-rollups/>`_
    and `Threshold-based alerting with Flux and Python hooks
    <https://taskautomation.org/automated-task-scheduling-orchestration/anomaly-detection-and-alerting/threshold-based-alerting-with-flux-and-python-hooks/>`_.

    Args:
        upper: Upper limit; ``None`` disables the upper check.
        lower: Lower limit; ``None`` disables the lower check.
        hysteresis: Width of the reset band, in the metric's own units.  Zero
            disables hysteresis (and keeps the detector Flux-compilable).
        consecutive_points: Adjacent breaches required before flagging.

    Example:
        >>> from datetime import datetime, timedelta, timezone
        >>> from tsdb_anomaly_task import Point, Series, ThresholdDetector
        >>> t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> pts = [Point(t0 + timedelta(minutes=i), v)
        ...        for i, v in enumerate([10, 10, 95, 10, 96, 97, 10])]
        >>> s = Series("cpu", "usage", {"host": "a"}, tuple(pts))
        >>> result = ThresholdDetector(upper=90, consecutive_points=2).evaluate(s)
        >>> len(result)  # the lone 95 is suppressed, the 96/97 pair is not
        2
    """

    name: ClassVar[str] = "threshold"

    def __init__(
        self,
        *,
        upper: float | None = None,
        lower: float | None = None,
        hysteresis: float = 0.0,
        consecutive_points: int = 1,
    ) -> None:
        super().__init__(consecutive_points=consecutive_points)
        if upper is None and lower is None:
            raise ValueError("ThresholdDetector requires at least one of 'upper' or 'lower'")
        if upper is not None and lower is not None and lower >= upper:
            raise ValueError(f"lower ({lower}) must be below upper ({upper})")
        if hysteresis < 0:
            raise ValueError("hysteresis must not be negative")
        if upper is not None and lower is not None and hysteresis > (upper - lower) / 2:
            raise ValueError(
                "hysteresis is wider than half the band; the reset points would overlap"
            )
        self.upper = None if upper is None else float(upper)
        self.lower = None if lower is None else float(lower)
        self.hysteresis = float(hysteresis)

    def describe(self) -> Mapping[str, object]:
        return {
            "detector": self.name,
            "upper": self.upper,
            "lower": self.lower,
            "hysteresis": self.hysteresis,
            "consecutive_points": self.consecutive_points,
        }

    # -- detection ---------------------------------------------------------

    def _judge(self, series: Series, now: datetime) -> Judgement:
        candidates: list[Candidate | None] = []
        state: str | None = None  # None | "high" | "low"

        for index, point in enumerate(series.points):
            value = point.value
            if not math.isfinite(value):
                candidates.append(None)
                continue

            state = self._next_state(state, value)
            if state == "high":
                assert self.upper is not None
                candidates.append(
                    Candidate(
                        index=index,
                        time=point.time,
                        value=value,
                        score=value,
                        threshold=self.upper,
                        reason=f"value {value:g} above upper limit {self.upper:g}",
                    )
                )
            elif state == "low":
                assert self.lower is not None
                candidates.append(
                    Candidate(
                        index=index,
                        time=point.time,
                        value=value,
                        score=value,
                        threshold=self.lower,
                        reason=f"value {value:g} below lower limit {self.lower:g}",
                    )
                )
            else:
                candidates.append(None)

        return Judgement(
            candidates=candidates,
            stats={
                "points": float(len(series)),
                "hysteresis": self.hysteresis,
            },
            usable=bool(series.points),
            notes=() if series.points else ("series is empty; nothing to evaluate",),
        )

    def _next_state(self, state: str | None, value: float) -> str | None:
        """Advance the alarm state machine by one sample."""
        if state == "high":
            assert self.upper is not None
            return "high" if value > self.upper - self.hysteresis else self._trip(value)
        if state == "low":
            assert self.lower is not None
            return "low" if value < self.lower + self.hysteresis else self._trip(value)
        return self._trip(value)

    def _trip(self, value: float) -> str | None:
        if self.upper is not None and value > self.upper:
            return "high"
        if self.lower is not None and value < self.lower:
            return "low"
        return None

    # -- Flux --------------------------------------------------------------

    def flux_support(self, context: FluxContext) -> FluxSupport:
        if self.hysteresis > 0:
            return FluxSupport(
                False,
                "hysteresis needs alarm state carried across rows, which Flux's "
                "row-wise pipeline cannot express; run this detector client-side",
            )
        return FluxSupport(True)

    def to_flux(self, context: FluxContext) -> str:
        support = self.flux_support(context)
        if not support:
            raise FluxUnsupportedError(support.reason)

        indent = context.indent
        tests: list[str] = []
        if self.upper is not None:
            tests.append(f"r._value > {flux_float(self.upper)}")
        if self.lower is not None:
            tests.append(f"r._value < {flux_float(self.lower)}")
        breach = " or ".join(tests)

        if self.upper is not None and self.lower is not None:
            threshold = (
                f"if r._value > {flux_float(self.upper)} "
                f"then {flux_float(self.upper)} else {flux_float(self.lower)}"
            )
            reason = (
                f"if r._value > {flux_float(self.upper)} "
                f'then "value above upper limit {self.upper:g}" '
                f'else "value below lower limit {self.lower:g}"'
            )
        elif self.upper is not None:
            threshold = flux_float(self.upper)
            reason = f'"value above upper limit {self.upper:g}"'
        else:
            assert self.lower is not None
            threshold = flux_float(self.lower)
            reason = f'"value below lower limit {self.lower:g}"'

        lines = [context.query_flux]
        if self.consecutive_points > 1:
            lines.append(f'{indent}|> stateCount(fn: (r) => {breach}, column: "_breachRun")')
            lines.append(f"{indent}|> filter(fn: (r) => r._breachRun >= {self.consecutive_points})")
        else:
            lines.append(f"{indent}|> filter(fn: (r) => {breach})")
        lines.append(f"{indent}|> map(fn: (r) => ({{ r with")
        lines.append(f"{indent}    _score: r._value,")
        lines.append(f"{indent}    _threshold: {threshold},")
        lines.append(f"{indent}    _reason: {reason},")
        lines.append(f"{indent}}}))")
        return "\n".join(lines)

    def flux_notes(self) -> tuple[str, ...]:
        """Caveats worth putting in the generated script's header."""
        if self.consecutive_points > 1:
            return (
                "stateCount emits a run only once it reaches "
                f"consecutive_points={self.consecutive_points}; the Python runner "
                "back-fills the earlier samples of the same run.",
            )
        return ()

    def __str__(self) -> str:
        bounds = []
        if self.lower is not None:
            bounds.append(f">= {escape_flux_string(f'{self.lower:g}')}")
        if self.upper is not None:
            bounds.append(f"<= {escape_flux_string(f'{self.upper:g}')}")
        return f"threshold({' and '.join(bounds)})"
