"""Detector base class, shared run-length logic, and the Flux support contract.

Every detector implements two things:

* :meth:`Detector._judge` — a pure, per-point verdict over a :class:`Series`.
* :meth:`Detector.flux_support` / :meth:`Detector.to_flux` — whether and how the
  detector compiles to a server-side Flux task.

The ``consecutive_points`` de-bounce is implemented once, here, so every
detector suppresses single-sample flapping the same way.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

from ..models import DetectionResult, Flag, Series, Severity

__all__ = ["Candidate", "Detector", "FluxContext", "FluxSupport", "FluxUnsupportedError"]


class FluxUnsupportedError(RuntimeError):
    """Raised when a detector cannot be compiled to a server-side Flux task."""


@dataclass(frozen=True, slots=True)
class FluxSupport:
    """Whether a detector compiles to Flux, and if not, why not."""

    supported: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.supported


@dataclass(frozen=True, slots=True)
class FluxContext:
    """Everything the Flux emitter needs that lives outside the detector.

    Detectors emit only the *detection* stage of the pipeline; the query
    prologue and the ``to()`` epilogue are supplied by the task.
    """

    query_flux: str
    flag_measurement: str
    indent: str = "  "
    group_by: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class Candidate:
    """A per-point verdict before de-bouncing.

    ``score`` and ``threshold`` are in the detector's own units; ``reason`` is
    the human-readable sentence that ends up on the written flag.
    """

    index: int
    time: datetime
    value: float
    score: float
    threshold: float
    reason: str

    def __post_init__(self) -> None:
        # Detectors compute with numpy; the public API only ever sees floats.
        for name in ("value", "score", "threshold"):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class Judgement:
    """Raw detector output: one optional candidate per evaluated point."""

    candidates: Sequence[Candidate | None]
    notes: tuple[str, ...] = ()
    stats: Mapping[str, float] = field(default_factory=dict)
    usable: bool = True
    points_evaluated: int | None = None


class Detector(abc.ABC):
    """Base class for all anomaly detectors.

    Subclasses set :attr:`name` and implement :meth:`_judge`.  Consecutive-point
    de-bouncing, severity assignment and result assembly are handled here.

    Args:
        consecutive_points: How many *adjacent* points must be judged anomalous
            before any of them is flagged.  ``1`` flags every offending sample;
            ``2`` or more suppresses lone spikes, which on real sensor feeds are
            far more often a dropped packet or an ADC glitch than a real event.
    """

    name: ClassVar[str] = "detector"

    def __init__(self, *, consecutive_points: int = 1) -> None:
        if consecutive_points < 1:
            raise ValueError("consecutive_points must be >= 1")
        self.consecutive_points = int(consecutive_points)

    # -- subclass contract -------------------------------------------------

    @abc.abstractmethod
    def _judge(self, series: Series, now: datetime) -> Judgement:
        """Return a per-point verdict for ``series`` as of ``now``."""

    def flux_support(self, context: FluxContext) -> FluxSupport:
        """Whether this detector, as configured, compiles to a Flux task."""
        return FluxSupport(False, f"{self.name} has no Flux implementation")

    def to_flux(self, context: FluxContext) -> str:
        """Emit the detection stage of the Flux pipeline.

        Raises:
            FluxUnsupportedError: if :meth:`flux_support` is false.
        """
        support = self.flux_support(context)
        if not support:
            raise FluxUnsupportedError(support.reason)
        raise NotImplementedError  # pragma: no cover - unreachable by contract

    def explain(self) -> str:
        """One-paragraph description of what this detector does, as configured."""
        return self.__class__.__doc__ or self.name

    def describe(self) -> Mapping[str, object]:
        """Machine-readable parameter dump, used by the tuning helpers."""
        return {"detector": self.name, "consecutive_points": self.consecutive_points}

    # -- public API --------------------------------------------------------

    def evaluate(self, series: Series, *, now: datetime | None = None) -> DetectionResult:
        """Run the detector over ``series`` and return flags plus diagnostics.

        Args:
            series: The series to evaluate.
            now: Evaluation time, used by time-relative detectors such as
                :class:`~tsdb_anomaly_task.DeadmanDetector`.  Defaults to the
                current UTC time; pass it explicitly to make a batch of series
                share one consistent reference instant.

        Never raises on thin or degenerate input: a series that cannot be
        evaluated comes back with ``usable=False`` and an explanatory note.
        """
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        judgement = self._judge(series, moment)
        kept = self._apply_consecutive(judgement.candidates)
        flags = tuple(self._to_flag(series, c) for c in kept)
        evaluated = (
            judgement.points_evaluated
            if judgement.points_evaluated is not None
            else len(judgement.candidates)
        )
        return DetectionResult(
            detector=self.name,
            series_key=series.key,
            flags=flags,
            points_evaluated=evaluated,
            notes=judgement.notes,
            stats=judgement.stats,
            usable=judgement.usable,
        )

    def evaluate_all(
        self, series: Sequence[Series], *, now: datetime | None = None
    ) -> list[DetectionResult]:
        """Evaluate several series against one shared reference instant."""
        moment = now or datetime.now(UTC)
        return [self.evaluate(s, now=moment) for s in series]

    # -- helpers -----------------------------------------------------------

    def _apply_consecutive(self, candidates: Sequence[Candidate | None]) -> list[Candidate]:
        """Keep only candidates belonging to a run of at least ``n`` in a row.

        Every point of a qualifying run is kept, including the ones before the
        threshold was reached, so an operator sees the full excursion rather
        than a truncated tail.
        """
        if self.consecutive_points <= 1:
            return [c for c in candidates if c is not None]

        kept: list[Candidate] = []
        run: list[Candidate] = []
        for candidate in candidates:
            if candidate is None:
                if len(run) >= self.consecutive_points:
                    kept.extend(run)
                run = []
            else:
                run.append(candidate)
        if len(run) >= self.consecutive_points:
            kept.extend(run)
        return kept

    def _to_flag(self, series: Series, candidate: Candidate) -> Flag:
        threshold = candidate.threshold
        ratio = abs(candidate.score) / abs(threshold) if threshold else float("inf")
        return Flag(
            time=candidate.time,
            value=candidate.value,
            score=candidate.score,
            threshold=threshold,
            detector=self.name,
            reason=candidate.reason,
            severity=Severity.from_ratio(ratio),
            series_key=series.key,
            tags=series.tags,
        )

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.describe().items() if k != "detector")
        return f"{self.__class__.__name__}({params})"
