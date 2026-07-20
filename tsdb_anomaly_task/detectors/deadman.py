"""Deadman detection: flag series that have gone quiet."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import ClassVar

from ..duration import format_duration, parse_duration
from ..flux import flux_float
from ..models import Series
from .base import Candidate, Detector, FluxContext, FluxSupport, FluxUnsupportedError, Judgement

__all__ = ["DeadmanDetector"]


class DeadmanDetector(Detector):
    """Flag a series that has stopped reporting for longer than ``tolerance``.

    Value-based detectors are blind to the most common IoT failure mode: the
    sensor does not report a wrong number, it reports *nothing*.  A battery
    dies, a gateway loses its uplink, a firmware watchdog reboots into a loop —
    and every threshold and z-score check stays perfectly quiet, because
    silence never crosses a limit.  A deadman check inverts the question and
    alerts on absence.  See `Deadman checks for detecting silent IoT sensors
    <https://taskautomation.org/automated-task-scheduling-orchestration/anomaly-detection-and-alerting/deadman-checks-for-detecting-silent-iot-sensors/>`_
    for the operational side of running these at fleet scale.

    Two kinds of silence are reported:

    * **Trailing silence** — the newest point is older than ``tolerance``
      relative to the evaluation instant.  This is the live "sensor is down"
      alert, flagged at the last known point.
    * **Interior gaps** — two consecutive points more than ``tolerance`` apart,
      flagged at the point where reporting resumed.  Opt in with ``flag_gaps``.
      These catch intermittent uplinks that a trailing check misses entirely,
      because by the time the task runs the sensor is talking again.  Interior
      gaps are the one part of this detector Flux cannot express, so enabling
      them moves the task to client-side execution.

    Set ``tolerance`` from the reporting interval, not from how long you are
    willing to wait: a sensor writing every 60s needs a tolerance of several
    intervals (``5m``, say) so that one dropped write is not an incident.

    Args:
        tolerance: Maximum acceptable silence.
        flag_gaps: Also flag interior gaps, not just trailing silence.
            Defaults to off, which keeps the detector Flux-compilable.

    Note:
        ``consecutive_points`` is meaningless here — deadman flags are isolated
        by construction — and passing anything but 1 raises.

    Example:
        >>> from datetime import datetime, timedelta, timezone
        >>> from tsdb_anomaly_task import DeadmanDetector, Point, Series
        >>> t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> s = Series("temp", "c", {"sensor": "a"},
        ...            (Point(t0, 20.0), Point(t0 + timedelta(minutes=1), 20.1)))
        >>> result = DeadmanDetector(tolerance="5m").evaluate(
        ...     s, now=t0 + timedelta(minutes=30))
        >>> result.flags[0].reason
        'no data for 29m (tolerance 5m)'
    """

    name: ClassVar[str] = "deadman"

    def __init__(
        self,
        *,
        tolerance: str | timedelta = "10m",
        flag_gaps: bool = False,
        consecutive_points: int = 1,
    ) -> None:
        if consecutive_points != 1:
            raise ValueError(
                "DeadmanDetector does not support consecutive_points; silence is "
                "reported as a single flag per gap"
            )
        super().__init__(consecutive_points=1)
        self.tolerance = tolerance
        self.flag_gaps = bool(flag_gaps)
        self._tolerance_delta = parse_duration(tolerance)
        if self._tolerance_delta.total_seconds() <= 0:
            raise ValueError("tolerance must be a positive duration")

    def describe(self) -> Mapping[str, object]:
        return {
            "detector": self.name,
            "tolerance": format_duration(self.tolerance),
            "flag_gaps": self.flag_gaps,
            "consecutive_points": self.consecutive_points,
        }

    @property
    def tolerance_seconds(self) -> float:
        return self._tolerance_delta.total_seconds()

    # -- detection ---------------------------------------------------------

    def _judge(self, series: Series, now: datetime) -> Judgement:
        points = series.points
        if not points:
            return Judgement(
                candidates=[],
                usable=False,
                notes=(
                    "series returned no points at all; a deadman check cannot "
                    "distinguish a silent sensor from one that never existed. "
                    "Compare against a registry of expected series instead.",
                ),
                stats={"points": 0.0},
            )

        tolerance = self.tolerance_seconds
        candidates: list[Candidate | None] = [None] * len(points)

        if self.flag_gaps:
            for i in range(1, len(points)):
                gap = (points[i].time - points[i - 1].time).total_seconds()
                if gap > tolerance:
                    candidates[i] = Candidate(
                        index=i,
                        time=points[i].time,
                        value=points[i].value,
                        score=gap,
                        threshold=tolerance,
                        reason=(
                            f"reporting gap of {format_duration(timedelta(seconds=gap))} "
                            f"(tolerance {format_duration(self.tolerance)})"
                        ),
                    )

        last = points[-1]
        silence = (now - last.time).total_seconds()
        if silence > tolerance:
            candidates[-1] = Candidate(
                index=len(points) - 1,
                time=last.time,
                value=last.value,
                score=silence,
                threshold=tolerance,
                reason=(
                    f"no data for {format_duration(timedelta(seconds=silence))} "
                    f"(tolerance {format_duration(self.tolerance)})"
                ),
            )

        return Judgement(
            candidates=candidates,
            stats={
                "points": float(len(points)),
                "silence_seconds": silence,
                "tolerance_seconds": tolerance,
                "max_gap_seconds": max(
                    (
                        (points[i].time - points[i - 1].time).total_seconds()
                        for i in range(1, len(points))
                    ),
                    default=0.0,
                ),
            },
            usable=True,
            points_evaluated=len(points),
        )

    # -- Flux --------------------------------------------------------------

    def flux_support(self, context: FluxContext) -> FluxSupport:
        if self.flag_gaps:
            return FluxSupport(
                False,
                "interior-gap detection needs the delta between adjacent rows "
                "combined with a per-row emit, which the server-side form does "
                "not express; set flag_gaps=False to compile the trailing-silence "
                "check to Flux, or run client-side to keep gap detection",
            )
        return FluxSupport(True)

    def to_flux(self, context: FluxContext) -> str:
        support = self.flux_support(context)
        if not support:
            raise FluxUnsupportedError(support.reason)

        indent = context.indent
        tolerance = self.tolerance_seconds
        return "\n".join(
            [
                context.query_flux,
                f"{indent}|> last()",
                f"{indent}|> map(fn: (r) => ({{ r with",
                f"{indent}    _score: float(v: int(v: now()) - int(v: r._time)) / 1000000000.0,",
                f"{indent}    _threshold: {flux_float(tolerance)},",
                f'{indent}    _reason: "series stopped reporting for longer than '
                f'{format_duration(self.tolerance)}",',
                f"{indent}}}))",
                f"{indent}|> filter(fn: (r) => r._score > {flux_float(tolerance)})",
            ]
        )

    def flux_notes(self) -> tuple[str, ...]:
        return (
            "A server-side deadman only sees series that returned at least one row in"
            " the task window. A sensor silent for longer than the window disappears"
            " entirely and must be caught by comparing against an expected-series list.",
        )

    def __str__(self) -> str:
        return f"deadman(tolerance={format_duration(self.tolerance)})"
