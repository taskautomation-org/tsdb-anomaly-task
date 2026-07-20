"""Core value types: queries, schedules, series, and anomaly flags.

Everything here is an immutable dataclass.  A detector receives a
:class:`Series` and returns a :class:`DetectionResult`; the task layer turns
those results into line-protocol records for the results bucket.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from .duration import format_duration, parse_duration

__all__ = [
    "DetectionResult",
    "Flag",
    "MetricQuery",
    "Point",
    "ResultsBucket",
    "Schedule",
    "Series",
    "Severity",
]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def _escape_flux_string(value: str) -> str:
    """Escape a Python string for safe interpolation into a Flux string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Severity(enum.Enum):
    """How badly a point violated the detector's expectation.

    Severity is derived from the ratio of the anomaly score to the threshold
    that fired, so it is comparable across detectors: ``1.0x`` is exactly at
    the limit, ``2.0x`` is twice as far out as the limit allows.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @classmethod
    def from_ratio(cls, ratio: float) -> Severity:
        """Map a score/threshold ratio onto a severity level."""
        if not math.isfinite(ratio):
            return cls.CRITICAL
        if ratio >= 2.0:
            return cls.CRITICAL
        if ratio >= 1.25:
            return cls.WARNING
        return cls.INFO

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Point:
    """A single ``(timestamp, value)`` sample of a metric."""

    time: datetime
    value: float

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            object.__setattr__(self, "time", self.time.replace(tzinfo=UTC))


@dataclass(frozen=True, slots=True)
class Series:
    """One time series: a tag set plus its points, sorted by time.

    A :class:`Series` is what a detector sees.  ``tags`` identifies the series
    (for example ``{"host": "edge-01"}``) and is copied verbatim onto every
    flag the detector emits, so downstream alert routing keeps working.
    """

    measurement: str
    field: str
    tags: Mapping[str, str] = field(default_factory=dict)
    points: tuple[Point, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", dict(self.tags))
        pts = tuple(sorted(self.points, key=lambda p: p.time))
        object.__setattr__(self, "points", pts)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[Point]:
        return iter(self.points)

    @property
    def key(self) -> str:
        """A stable, human-readable identifier for this series."""
        if not self.tags:
            return f"{self.measurement}.{self.field}"
        tagstr = ",".join(f"{k}={v}" for k, v in sorted(self.tags.items()))
        return f"{self.measurement}.{self.field}{{{tagstr}}}"

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(p.value for p in self.points)

    @property
    def times(self) -> tuple[datetime, ...]:
        return tuple(p.time for p in self.points)

    @property
    def start(self) -> datetime | None:
        return self.points[0].time if self.points else None

    @property
    def end(self) -> datetime | None:
        return self.points[-1].time if self.points else None

    def span(self) -> timedelta:
        """Wall-clock duration covered by the series (zero if fewer than 2 points)."""
        if len(self.points) < 2:
            return timedelta(0)
        return self.points[-1].time - self.points[0].time

    def slice(self, start: datetime, end: datetime) -> Series:
        """Return a copy containing only points in ``[start, end]``."""
        return Series(
            measurement=self.measurement,
            field=self.field,
            tags=self.tags,
            points=tuple(p for p in self.points if start <= p.time <= end),
        )


@dataclass(frozen=True, slots=True)
class Flag:
    """A single flagged point, with everything needed to explain the alert.

    ``score`` and ``threshold`` are in the detector's own units (a robust
    z-score for MAD, the raw value for a threshold check, seconds of silence
    for a deadman check).  ``ratio`` normalises them so severity is comparable.
    """

    time: datetime
    value: float
    score: float
    threshold: float
    detector: str
    reason: str
    severity: Severity = Severity.WARNING
    series_key: str = ""
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", dict(self.tags))
        if self.time.tzinfo is None:
            object.__setattr__(self, "time", self.time.replace(tzinfo=UTC))

    @property
    def label(self) -> str:
        """Compact identifier for terminal output: the tag set, or the field."""
        if self.tags:
            return ",".join(f"{k}={v}" for k, v in sorted(self.tags.items()))
        return self.series_key or self.detector

    @property
    def ratio(self) -> float:
        """Score expressed as a multiple of the threshold that fired."""
        if self.threshold == 0:
            return math.inf if self.score != 0 else 0.0
        return abs(self.score) / abs(self.threshold)

    def to_line_protocol(self, measurement: str, precision_ns: bool = True) -> str:
        """Render the flag as an InfluxDB line-protocol record."""
        tags = dict(self.tags)
        tags["detector"] = self.detector
        tags["severity"] = str(self.severity)
        tag_part = "".join(
            f",{_escape_lp_key(k)}={_escape_lp_key(str(v))}"
            for k, v in sorted(tags.items())
            if v != ""
        )
        fields = (
            f"value={self.value!r},"
            f"score={self.score!r},"
            f"threshold={self.threshold!r},"
            f'reason="{_escape_lp_field(self.reason)}"'
        )
        ts = int(self.time.timestamp() * (1e9 if precision_ns else 1e3))
        return f"{_escape_lp_key(measurement)}{tag_part} {fields} {ts}"


def _escape_lp_key(value: str) -> str:
    return value.replace("\\", "\\\\").replace(" ", r"\ ").replace(",", r"\,").replace("=", r"\=")


def _escape_lp_field(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Flags plus the diagnostics that explain how the detector behaved.

    Detectors never raise on thin data — they return a result with an empty
    ``flags`` list and a note in ``notes`` saying why.  Callers that want to
    treat "not enough history" as an error can check :attr:`usable`.
    """

    detector: str
    series_key: str
    flags: tuple[Flag, ...] = ()
    points_evaluated: int = 0
    notes: tuple[str, ...] = ()
    stats: Mapping[str, float] = field(default_factory=dict)
    usable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "flags", tuple(self.flags))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "stats", dict(self.stats))

    def __iter__(self) -> Iterator[Flag]:
        return iter(self.flags)

    def __len__(self) -> int:
        return len(self.flags)

    def __bool__(self) -> bool:
        return bool(self.flags)

    @property
    def flag_rate(self) -> float:
        """Fraction of evaluated points that were flagged."""
        if not self.points_evaluated:
            return 0.0
        return len(self.flags) / self.points_evaluated


@dataclass(frozen=True, slots=True)
class MetricQuery:
    """Declares *which* metric to watch.

    Args:
        bucket: Source bucket name.
        measurement: Measurement to read.
        field: Field key to evaluate.
        filters: Extra tag predicates.  A value of ``"*"`` means "the tag must
            exist but may hold any value" and compiles to ``exists r.<tag>``.
            A list/tuple value compiles to an OR-group.
        group_by: Tag keys that separate independent series.  Each group is
            evaluated on its own, which is what you want for per-host or
            per-sensor detection.
        range_start: How far back the task/preview reads, relative to now.
        aggregate_window: Optional pre-aggregation (for example ``"1m"``),
            applied with ``aggregateWindow`` before detection.  Useful for
            taming high-frequency sensors before a threshold check.
        aggregate_fn: Flux aggregate applied by ``aggregate_window``.
    """

    bucket: str
    measurement: str
    field: str
    filters: Mapping[str, str | Sequence[str]] = field(default_factory=dict)
    group_by: Sequence[str] = ()
    range_start: str = "-1h"
    range_stop: str | None = None
    aggregate_window: str | None = None
    aggregate_fn: str = "mean"

    def __post_init__(self) -> None:
        for name, value in (
            ("bucket", self.bucket),
            ("measurement", self.measurement),
            ("field", self.field),
        ):
            if not value or not str(value).strip():
                raise ValueError(f"MetricQuery.{name} must not be empty")
        object.__setattr__(self, "filters", dict(self.filters))
        object.__setattr__(self, "group_by", tuple(self.group_by))
        if self.aggregate_window is not None:
            parse_duration(self.aggregate_window)
            if self.aggregate_fn not in {"mean", "median", "max", "min", "sum", "last", "first"}:
                raise ValueError(f"unsupported aggregate_fn {self.aggregate_fn!r}")
        # A range start of "-1h" is relative; anything else must still parse.
        parse_duration(self.range_start)

    @property
    def lookback(self) -> timedelta:
        """Absolute size of the read window."""
        return abs(parse_duration(self.range_start))

    def to_flux(self, indent: str = "  ") -> str:
        """Compile the query half of the pipeline to Flux."""
        lines = [f'from(bucket: "{_escape_flux_string(self.bucket)}")']
        stop = f", stop: {self.range_stop}" if self.range_stop else ""
        lines.append(f"{indent}|> range(start: {format_duration(self.range_start)}{stop})")
        lines.append(
            f"{indent}|> filter(fn: (r) => r._measurement == "
            f'"{_escape_flux_string(self.measurement)}")'
        )
        lines.append(
            f'{indent}|> filter(fn: (r) => r._field == "{_escape_flux_string(self.field)}")'
        )
        for key, value in sorted(self.filters.items()):
            lines.append(f"{indent}|> filter(fn: (r) => {self._filter_expr(key, value)})")
        if self.aggregate_window:
            lines.append(
                f"{indent}|> aggregateWindow(every: "
                f"{format_duration(self.aggregate_window)}, "
                f"fn: {self.aggregate_fn}, createEmpty: false)"
            )
        if self.group_by:
            cols = ", ".join(f'"{_escape_flux_string(g)}"' for g in self.group_by)
            lines.append(f"{indent}|> group(columns: [{cols}])")
        else:
            lines.append(f"{indent}|> group()")
        lines.append(f'{indent}|> sort(columns: ["_time"])')
        return "\n".join(lines)

    def _filter_expr(self, key: str, value: str | Sequence[str]) -> str:
        if not _IDENT.match(key):
            raise ValueError(f"invalid tag key {key!r} in MetricQuery.filters")
        ref = f"r.{key}" if key.replace("_", "").isalnum() else f'r["{key}"]'
        if isinstance(value, str):
            if value == "*":
                return f"exists {ref}"
            return f'{ref} == "{_escape_flux_string(value)}"'
        options = list(value)
        if not options:
            raise ValueError(f"filter {key!r} has an empty value list")
        return " or ".join(f'{ref} == "{_escape_flux_string(v)}"' for v in options)


@dataclass(frozen=True, slots=True)
class Schedule:
    """When the task runs.

    Exactly one of ``every`` or ``cron`` must be set.  ``offset`` delays the
    run relative to its schedule so late-arriving sensor data has landed before
    the window is read — without it, an interval task reliably evaluates a
    window that the slowest writers have not finished filling.
    """

    every: str | None = None
    cron: str | None = None
    offset: str | None = None

    def __post_init__(self) -> None:
        if bool(self.every) == bool(self.cron):
            raise ValueError("Schedule requires exactly one of 'every' or 'cron'")
        if self.every and parse_duration(self.every).total_seconds() <= 0:
            raise ValueError("Schedule.every must be a positive duration")
        if self.offset and parse_duration(self.offset).total_seconds() < 0:
            raise ValueError("Schedule.offset must not be negative")

    @property
    def interval(self) -> timedelta | None:
        """The run interval, or ``None`` for cron schedules."""
        return parse_duration(self.every) if self.every else None

    def to_task_options(self, name: str) -> str:
        """Render the Flux ``option task = {...}`` header."""
        parts = [f'name: "{_escape_flux_string(name)}"']
        if self.every:
            parts.append(f"every: {format_duration(self.every)}")
        else:
            parts.append(f'cron: "{_escape_flux_string(str(self.cron))}"')
        if self.offset:
            parts.append(f"offset: {format_duration(self.offset)}")
        return "option task = {" + ", ".join(parts) + "}"


@dataclass(frozen=True, slots=True)
class ResultsBucket:
    """Where flags are written.

    Args:
        bucket: Destination bucket for anomaly records.
        flag_measurement: Measurement name used for written flags.
        org: Optional org override; defaults to the client's org.
        extra_tags: Static tags stamped onto every flag (e.g. ``{"env": "prod"}``).
    """

    bucket: str
    flag_measurement: str = "anomaly"
    org: str | None = None
    extra_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise ValueError("ResultsBucket.bucket must not be empty")
        if not self.flag_measurement.strip():
            raise ValueError("ResultsBucket.flag_measurement must not be empty")
        object.__setattr__(self, "extra_tags", dict(self.extra_tags))

    def to_flux(self, indent: str = "  ") -> str:
        """Render the ``to()`` call that writes flags back."""
        org = f', org: "{_escape_flux_string(self.org)}"' if self.org else ""
        return f'{indent}|> to(bucket: "{_escape_flux_string(self.bucket)}"{org})'
