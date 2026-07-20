"""Robust z-scores via the median absolute deviation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np

from ..duration import format_duration, parse_duration
from ..flux import flux_float
from ..models import Series
from .base import Candidate, Detector, FluxContext, FluxSupport, FluxUnsupportedError, Judgement

__all__ = ["NORMAL_CONSISTENCY", "MADDetector", "robust_scale"]

#: 1 / Φ⁻¹(0.75).  Scales the MAD so that, for normally distributed data, it
#: estimates the same quantity as the standard deviation.
NORMAL_CONSISTENCY = 1.4826022185056018


def small_sample_correction(n: int) -> float:
    """Croux–Rousseeuw finite-sample correction ``n / (n - 0.8)`` for the MAD.

    The 1.4826 constant is *asymptotic*.  On a short baseline the raw MAD is
    biased low, which inflates every z-score computed against it and is the
    single most common cause of spurious alerts from a seasonal detector, where
    each bucket may only hold a handful of training points.  Multiplying by
    ``n / (n - 0.8)`` removes most of that bias and costs nothing.
    """
    if n < 2:
        return 1.0
    return n / (n - 0.8)


def robust_scale(
    values: np.ndarray | list[float], *, finite_sample: bool = True
) -> tuple[float, float]:
    """Return ``(median, consistency-scaled MAD)`` for ``values``.

    The scaled MAD is ``1.4826 * median(|x - median(x)|)``.  The constant is
    ``1/Φ⁻¹(0.75)``, chosen so that for Gaussian data the result converges on
    the standard deviation — which means a MAD-based z-score can be read with
    the same intuition as a classical one (3.0 is "far out"), while keeping the
    median's resistance to contamination.

    Args:
        values: The sample.  Non-finite entries are dropped.
        finite_sample: Apply :func:`small_sample_correction`.  Leave this on
            unless you are reproducing a textbook asymptotic figure.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), 0.0)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = NORMAL_CONSISTENCY * mad
    if finite_sample:
        scale *= small_sample_correction(int(arr.size))
    return (median, scale)


class MADDetector(Detector):
    """Flag points whose robust z-score exceeds ``k``.

    The textbook outlier test is ``|x - mean| / stddev > k``.  It has a fatal
    flaw on machine data: both the mean and the standard deviation are computed
    *from the data being tested*, and both are unbounded — a single bad sample
    drags the mean toward itself and inflates the standard deviation.  A sensor
    that reports ``-9999`` on a read error will therefore raise the very
    threshold that was supposed to catch it, and can mask genuine anomalies
    around it.  The breakdown point of the mean/stddev pair is 0%: one bad
    value in a million is enough to move them arbitrarily far.

    The median and the MAD have a breakdown point of 50%: up to half the window
    can be garbage before the estimate moves at all.  Multiplying the MAD by
    :data:`NORMAL_CONSISTENCY` (1.4826) rescales it onto the standard
    deviation's scale for Gaussian data, so ``k=3.0`` keeps its familiar
    meaning while the estimator stops being hijacked by the outliers.  This is
    the same reasoning that makes the median the better aggregate for noisy
    sensor rollups — see `Choosing mean vs median for noisy sensor rollups
    <https://taskautomation.org/downsampling-aggregation-pipeline-design/threshold-tuning-for-aggregation/choosing-mean-vs-median-for-noisy-sensor-rollups/>`_.

    The baseline is computed **leave-one-out**: the point under test is
    excluded from its own median and MAD, so a large excursion cannot dilute
    the very statistic that judges it.

    Args:
        k: Robust z-score above which a point is flagged.  3.0 is a sensible
            default; 3.5 is quieter.  Use :func:`tsdb_anomaly_task.sweep_parameter`
            to pick one against real history instead of guessing.
        window: Trailing window used to build the baseline (for example
            ``"1h"``).  ``None`` uses the whole series.
        min_points: Minimum baseline samples required before a point can be
            judged; points with a thinner window are skipped, not flagged.
        consecutive_points: Adjacent breaches required before flagging.

    Example:
        >>> from datetime import datetime, timedelta, timezone
        >>> from tsdb_anomaly_task import MADDetector, Point, Series
        >>> t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> vals = [20.0, 20.1, 19.9, 20.0, 20.2, 19.8, 20.1, 45.0, 20.0, 19.9]
        >>> s = Series("temp", "c", {}, tuple(
        ...     Point(t0 + timedelta(minutes=i), v) for i, v in enumerate(vals)))
        >>> [f.value for f in MADDetector(k=3.5, min_points=5).evaluate(s)]
        [45.0]
    """

    name: ClassVar[str] = "mad"

    def __init__(
        self,
        *,
        k: float = 3.0,
        window: str | timedelta | None = None,
        min_points: int = 8,
        consecutive_points: int = 1,
    ) -> None:
        super().__init__(consecutive_points=consecutive_points)
        if k <= 0:
            raise ValueError("k must be positive")
        if min_points < 3:
            raise ValueError("min_points must be >= 3; a MAD needs a real sample to work with")
        self.k = float(k)
        self.window = window
        self.min_points = int(min_points)
        self._window_delta = parse_duration(window) if window is not None else None
        if self._window_delta is not None and self._window_delta.total_seconds() <= 0:
            raise ValueError("window must be a positive duration")

    def describe(self) -> Mapping[str, object]:
        return {
            "detector": self.name,
            "k": self.k,
            "window": format_duration(self.window) if self.window else None,
            "min_points": self.min_points,
            "consecutive_points": self.consecutive_points,
        }

    # -- detection ---------------------------------------------------------

    def _judge(self, series: Series, now: datetime) -> Judgement:
        n = len(series.points)
        candidates: list[Candidate | None] = [None] * n
        if n == 0:
            return Judgement(candidates=[], usable=False, notes=("series is empty",))

        times = np.array([p.time.timestamp() for p in series.points])
        values = np.array([p.value for p in series.points], dtype=float)

        evaluated = 0
        skipped_thin = 0
        skipped_degenerate = 0
        scores: list[float] = []

        global_median, global_scale = robust_scale(values)

        for i in range(n):
            value = values[i]
            if not math.isfinite(value):
                continue
            baseline = self._baseline_values(times, values, i)
            if baseline.size < self.min_points:
                skipped_thin += 1
                continue

            median, scale = robust_scale(baseline)
            if scale <= 0.0:
                # A perfectly flat baseline gives a zero MAD, which would make
                # every deviation infinitely anomalous.  Fall back to the scale
                # of the whole series; if that is flat too, the series carries
                # no information and we skip rather than invent an alert.
                scale = global_scale
                if scale <= 0.0:
                    skipped_degenerate += 1
                    continue

            evaluated += 1
            score = (value - median) / scale
            scores.append(abs(score))
            if abs(score) >= self.k:
                direction = "above" if score > 0 else "below"
                candidates[i] = Candidate(
                    index=i,
                    time=series.points[i].time,
                    value=value,
                    score=score,
                    threshold=self.k if score > 0 else -self.k,
                    reason=(
                        f"robust z-score {score:+.2f} is {direction} the median "
                        f"{median:g} by more than k={self.k:g} scaled MADs "
                        f"(scaled MAD {scale:.4g})"
                    ),
                )

        notes: list[str] = []
        if skipped_thin:
            notes.append(
                f"{skipped_thin} point(s) skipped: fewer than min_points={self.min_points} "
                f"baseline samples available"
            )
        if skipped_degenerate:
            notes.append(
                f"{skipped_degenerate} point(s) skipped: baseline MAD is zero "
                f"(perfectly constant series)"
            )

        stats = {
            "points": float(n),
            "evaluated": float(evaluated),
            "median": global_median,
            "scaled_mad": global_scale,
            "max_abs_score": max(scores) if scores else 0.0,
        }
        return Judgement(
            candidates=candidates,
            notes=tuple(notes),
            stats=stats,
            usable=evaluated > 0,
            points_evaluated=evaluated,
        )

    def _baseline_values(self, times: np.ndarray, values: np.ndarray, i: int) -> np.ndarray:
        """Leave-one-out baseline sample for point ``i``."""
        if self._window_delta is None:
            mask = np.ones(values.size, dtype=bool)
        else:
            lo = times[i] - self._window_delta.total_seconds()
            mask = (times >= lo) & (times <= times[i])
        mask[i] = False
        selected = values[mask]
        return selected[np.isfinite(selected)]

    # -- Flux --------------------------------------------------------------

    def flux_support(self, context: FluxContext) -> FluxSupport:
        reasons: list[str] = []
        if self._window_delta is not None:
            reasons.append(
                "a rolling window needs a per-row median, which Flux has no "
                "windowed-quantile primitive for"
            )
        if context.group_by:
            reasons.append(
                "grouped series need one median scalar per group, and findRecord "
                "extracts a single scalar from a single table"
            )
        if self.consecutive_points > 1:
            reasons.append(
                "consecutive-point de-bouncing of a computed score cannot be "
                "expressed with stateCount over a mapped column"
            )
        if reasons:
            return FluxSupport(False, "; ".join(reasons))
        return FluxSupport(True)

    def to_flux(self, context: FluxContext) -> str:
        support = self.flux_support(context)
        if not support:
            raise FluxUnsupportedError(support.reason)

        indent = context.indent
        query = context.query_flux
        return "\n".join(
            [
                "// Whole-window MAD: the median and MAD are computed once over the",
                "// task's range and applied to every row (unlike the client-side",
                "// runner, which excludes each point from its own baseline).",
                f"data = {query}",
                "",
                "_median = (data",
                f'{indent}|> quantile(q: 0.5, method: "exact_selector")',
                f"{indent}|> findRecord(fn: (key) => true, idx: 0))._value",
                "",
                "_scaledMad = (data",
                f"{indent}|> map(fn: (r) => "
                "({ r with _value: math.abs(x: r._value - _median) }))",
                f'{indent}|> quantile(q: 0.5, method: "exact_selector")',
                f"{indent}|> findRecord(fn: (key) => true, idx: 0))._value * "
                f"{flux_float(NORMAL_CONSISTENCY)}",
                "",
                "data",
                f"{indent}|> filter(fn: (r) => _scaledMad > 0.0)",
                f"{indent}|> map(fn: (r) => ({{ r with",
                f"{indent}    _score: (r._value - _median) / _scaledMad,",
                f"{indent}    _threshold: {flux_float(self.k)},",
                f'{indent}    _reason: "robust z-score exceeds k={self.k:g} scaled MADs",',
                f"{indent}}}))",
                f"{indent}|> filter(fn: (r) => math.abs(x: r._score) >= {flux_float(self.k)})",
            ]
        )

    def flux_notes(self) -> tuple[str, ...]:
        return (
            "MAD is evaluated over the whole task window, not a rolling window.",
            f"Consistency constant {NORMAL_CONSISTENCY:.4f} = 1/inverse-normal-CDF(0.75).",
        )

    def __str__(self) -> str:
        window = format_duration(self.window) if self.window else "all"
        return f"mad(k={self.k:g}, window={window})"
