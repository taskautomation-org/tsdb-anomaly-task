"""Parameter sweeps: pick a threshold from data instead of from a hunch.

The usual failure mode of an anomaly detector is not a bad algorithm, it is a
number someone typed once.  ``k=3`` feels principled until it produces 400
alerts a day on a noisy flow meter, and ``k=6`` feels safe until it misses the
event you built the check for.

:func:`sweep_parameter` replays real history at a range of parameter values and
reports what each one would have cost you.  Pick the knee of the curve, then
confirm it with :meth:`~tsdb_anomaly_task.task.AnomalyTask.preview`.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from .client import InfluxProtocol
from .models import Series
from .task import AnomalyTask

__all__ = ["SweepResult", "SweepRow", "sweep_parameter"]


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One parameter value and the alert load it would have produced."""

    value: float | int | str
    flags: int
    points: int
    series_flagged: int
    max_score: float

    @property
    def flag_rate(self) -> float:
        return self.flags / self.points if self.points else 0.0

    @property
    def flags_per_1k(self) -> float:
        """Flags per thousand evaluated points — the readable form of the rate."""
        return self.flag_rate * 1000.0


@dataclass(frozen=True, slots=True)
class SweepResult:
    """A whole sweep, plus a recommendation."""

    parameter: str
    rows: tuple[SweepRow, ...]
    target_rate: float | None = None

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def recommended(self) -> SweepRow | None:
        """The least aggressive value whose flag rate is at or below the target.

        With no target, the knee of the curve is returned instead: the value
        after which tightening the parameter stops buying much quiet.
        """
        if not self.rows:
            return None
        if self.target_rate is not None:
            for row in self.rows:
                if row.flag_rate <= self.target_rate:
                    return row
            return self.rows[-1]

        best, best_drop = self.rows[0], -1.0
        for previous, row in zip(self.rows, self.rows[1:], strict=False):
            drop = previous.flag_rate - row.flag_rate
            if drop > best_drop:
                best, best_drop = row, drop
        return best

    def render(self) -> str:
        """Render the sweep as a fixed-width table."""
        lines = [
            f"sweep: {self.parameter}",
            f"  {'value':>10}{'flags':>8}{'points':>9}{'per 1k':>9}{'series':>8}{'max score':>11}",
            "  " + "-" * 55,
        ]
        pick = self.recommended
        for row in self.rows:
            marker = " <-" if pick is not None and row is pick else ""
            lines.append(
                f"  {row.value!s:>10}{row.flags:>8}{row.points:>9}"
                f"{row.flags_per_1k:>9.1f}{row.series_flagged:>8}"
                f"{row.max_score:>11.2f}{marker}"
            )
        return "\n".join(lines)


def sweep_parameter(
    task: AnomalyTask,
    client: InfluxProtocol,
    *,
    parameter: str = "k",
    values: Sequence[float | int | str],
    now: datetime | None = None,
    target_rate: float | None = None,
) -> SweepResult:
    """Re-run ``task``'s detector over the same data at several parameter values.

    The data is read **once** and replayed for every value, so a sweep costs one
    query no matter how many candidates you try — and every row is comparable
    because they all saw exactly the same points.

    Args:
        task: The task whose detector is being tuned.
        client: Source of the history to replay.
        parameter: Detector attribute to vary (``"k"``, ``"upper"``,
            ``"consecutive_points"``, ``"tolerance"`` …).
        values: Candidate values, in the order you want them reported.
        now: Reference instant for the read.
        target_rate: If given, :attr:`SweepResult.recommended` returns the first
            value whose flag rate falls at or below this fraction.

    Raises:
        AttributeError: if the detector has no such parameter.

    Example:
        >>> from tsdb_anomaly_task import *          # doctest: +SKIP
        >>> sweep_parameter(task, client, parameter="k",
        ...                 values=[2.5, 3.0, 3.5, 4.0]).render()  # doctest: +SKIP
    """
    if not values:
        raise ValueError("sweep needs at least one candidate value")
    if not hasattr(task.detector, parameter):
        raise AttributeError(
            f"{type(task.detector).__name__} has no parameter {parameter!r}; "
            f"available: {sorted(task.detector.describe())}"
        )

    moment = now or datetime.now(UTC)
    series: list[Series] = list(client.read(task.query, now=moment))

    rows: list[SweepRow] = []
    for value in values:
        detector = _respec(task.detector, parameter, value)
        variant = replace(task, detector=detector)
        results = variant.detector.evaluate_all(series, now=moment)
        flags = [f for r in results for f in r.flags]
        rows.append(
            SweepRow(
                value=value,
                flags=len(flags),
                points=sum(r.points_evaluated for r in results),
                series_flagged=sum(1 for r in results if r.flags),
                max_score=max((abs(f.score) for f in flags), default=0.0),
            )
        )
    return SweepResult(parameter=parameter, rows=tuple(rows), target_rate=target_rate)


def _respec(detector, parameter: str, value):  # type: ignore[no-untyped-def]
    """Return a copy of ``detector`` with one constructor parameter changed.

    Detectors do validation and derive cached state in ``__init__``, so the
    copy is rebuilt through the constructor rather than mutated in place.
    """
    params = dict(detector.describe())
    params.pop("detector", None)
    params[parameter] = value
    try:
        return type(detector)(**params)
    except TypeError:
        # A detector whose describe() is not constructor-shaped: fall back to a
        # shallow copy with the attribute replaced, re-running __init__ hooks.
        clone = copy.copy(detector)
        setattr(clone, parameter, value)
        return clone
