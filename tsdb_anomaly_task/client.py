"""The narrow client layer.

The library never talks to ``influxdb_client`` directly.  Everything goes
through :class:`InfluxProtocol`, which has five methods.  That keeps the
detection code testable without a server, and it means a
:class:`FakeInfluxClient` is a complete stand-in rather than a partial mock.

Three implementations ship here:

* :class:`InfluxClient` — the real adapter over ``influxdb_client``.
* :class:`FakeInfluxClient` — in-memory, deterministic, used by every test.
* :class:`RecordingClient` — wraps another client and logs the calls made.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .models import Flag, MetricQuery, Point, Series

if TYPE_CHECKING:  # pragma: no cover
    from .synthetic import SyntheticSeries

__all__ = [
    "AsyncInfluxProtocol",
    "FakeInfluxClient",
    "InfluxClient",
    "InfluxProtocol",
    "RecordingClient",
    "TaskRef",
    "WriteError",
]


class WriteError(RuntimeError):
    """Raised when a write to InfluxDB fails; retried by the async runner."""


@dataclass(frozen=True, slots=True)
class TaskRef:
    """A server-side task as the client sees it."""

    id: str
    name: str
    flux: str
    status: str = "active"
    every: str | None = None
    offset: str | None = None
    created: bool = True

    @property
    def updated(self) -> bool:
        """True when :meth:`InfluxProtocol.upsert_task` replaced an existing task."""
        return not self.created


@runtime_checkable
class InfluxProtocol(Protocol):
    """The synchronous surface the library needs from InfluxDB."""

    def read(self, query: MetricQuery, *, now: datetime | None = None) -> list[Series]:
        """Execute ``query`` and return one :class:`Series` per group."""
        ...

    def write_flags(
        self,
        bucket: str,
        measurement: str,
        flags: Sequence[Flag],
        extra_tags: Mapping[str, str] | None = None,
    ) -> int:
        """Write flags to ``bucket``; returns the number of records written."""
        ...

    def upsert_task(
        self, name: str, flux: str, *, every: str | None = None, offset: str | None = None
    ) -> TaskRef:
        """Create the task, or replace the script of an existing one."""
        ...

    def find_task(self, name: str) -> TaskRef | None:
        """Return the task with this name, or ``None``."""
        ...

    def delete_task(self, name: str) -> bool:
        """Delete the task; returns whether anything was deleted."""
        ...


@runtime_checkable
class AsyncInfluxProtocol(Protocol):
    """The asynchronous surface used by :mod:`tsdb_anomaly_task.runner`."""

    async def aread(self, query: MetricQuery, *, now: datetime | None = None) -> list[Series]: ...

    async def awrite_flags(
        self,
        bucket: str,
        measurement: str,
        flags: Sequence[Flag],
        extra_tags: Mapping[str, str] | None = None,
    ) -> int: ...


# ---------------------------------------------------------------------------
# Real adapter
# ---------------------------------------------------------------------------


class InfluxClient:
    """Adapter over ``influxdb_client.InfluxDBClient``.

    Constructed from an existing client so token handling, TLS and connection
    pooling stay the caller's concern:

    .. code-block:: python

        from influxdb_client import InfluxDBClient
        from tsdb_anomaly_task import InfluxClient

        client = InfluxClient(InfluxDBClient(url=..., token=..., org="acme"))

    Async methods run the blocking client in a worker thread, which is the
    honest way to use ``influxdb_client`` from asyncio without pretending its
    HTTP layer is non-blocking.
    """

    def __init__(self, client: Any, *, org: str | None = None) -> None:
        self._client = client
        self.org = org or getattr(client, "org", None)
        if not self.org:
            raise ValueError("an org is required; pass org= or use a client that carries one")

    # -- reads -------------------------------------------------------------

    def read(self, query: MetricQuery, *, now: datetime | None = None) -> list[Series]:
        flux = query.to_flux()
        tables = self._client.query_api().query(flux, org=self.org)
        return self._tables_to_series(tables, query)

    async def aread(self, query: MetricQuery, *, now: datetime | None = None) -> list[Series]:
        return await asyncio.to_thread(self.read, query, now=now)

    @staticmethod
    def _tables_to_series(tables: Iterable[Any], query: MetricQuery) -> list[Series]:
        """Fold Flux tables into :class:`Series`, one per distinct tag set."""
        grouped: dict[tuple[tuple[str, str], ...], list[Point]] = {}
        for table in tables:
            for record in table.records:
                values = record.values
                tags = tuple(
                    sorted(
                        (str(k), str(v))
                        for k, v in values.items()
                        if k in query.group_by and v is not None
                    )
                )
                moment = record.get_time()
                value = record.get_value()
                if moment is None or value is None:
                    continue
                grouped.setdefault(tags, []).append(Point(moment, float(value)))

        return [
            Series(
                measurement=query.measurement,
                field=query.field,
                tags=dict(tags),
                points=tuple(points),
            )
            for tags, points in sorted(grouped.items())
        ]

    # -- writes ------------------------------------------------------------

    def write_flags(
        self,
        bucket: str,
        measurement: str,
        flags: Sequence[Flag],
        extra_tags: Mapping[str, str] | None = None,
    ) -> int:
        if not flags:
            return 0
        records = [_render(flag, measurement, extra_tags) for flag in flags]
        try:
            with self._client.write_api() as writer:
                writer.write(bucket=bucket, org=self.org, record=records)
        except Exception as exc:
            raise WriteError(
                f"write of {len(records)} flag(s) to {bucket!r} failed: {exc}"
            ) from exc
        return len(records)

    async def awrite_flags(
        self,
        bucket: str,
        measurement: str,
        flags: Sequence[Flag],
        extra_tags: Mapping[str, str] | None = None,
    ) -> int:
        return await asyncio.to_thread(self.write_flags, bucket, measurement, flags, extra_tags)

    # -- tasks -------------------------------------------------------------

    def upsert_task(
        self, name: str, flux: str, *, every: str | None = None, offset: str | None = None
    ) -> TaskRef:
        api = self._client.tasks_api()
        existing = self._find_raw(name)
        if existing is not None:
            existing.flux = flux
            updated = api.update_task(existing)
            return TaskRef(
                id=str(updated.id),
                name=name,
                flux=flux,
                status=str(getattr(updated, "status", "active")),
                every=every,
                offset=offset,
                created=False,
            )
        created = api.create_task_with_script(name=name, flux=flux, org_id=self._org_id())
        return TaskRef(
            id=str(created.id),
            name=name,
            flux=flux,
            status=str(getattr(created, "status", "active")),
            every=every,
            offset=offset,
            created=True,
        )

    def find_task(self, name: str) -> TaskRef | None:
        raw = self._find_raw(name)
        if raw is None:
            return None
        return TaskRef(
            id=str(raw.id),
            name=name,
            flux=str(getattr(raw, "flux", "")),
            status=str(getattr(raw, "status", "active")),
            created=False,
        )

    def delete_task(self, name: str) -> bool:
        raw = self._find_raw(name)
        if raw is None:
            return False
        self._client.tasks_api().delete_task(raw.id)
        return True

    def _find_raw(self, name: str) -> Any | None:
        for task in self._client.tasks_api().find_tasks(name=name) or []:
            if task.name == name:
                return task
        return None

    def _org_id(self) -> str:
        orgs = self._client.organizations_api().find_organizations(org=self.org)
        if not orgs:
            raise ValueError(f"organization {self.org!r} not found")
        return str(orgs[0].id)


def _render(flag: Flag, measurement: str, extra_tags: Mapping[str, str] | None) -> str:
    if extra_tags:
        merged = dict(flag.tags)
        merged.update(extra_tags)
        flag = Flag(
            time=flag.time,
            value=flag.value,
            score=flag.score,
            threshold=flag.threshold,
            detector=flag.detector,
            reason=flag.reason,
            severity=flag.severity,
            series_key=flag.series_key,
            tags=merged,
        )
    return flag.to_line_protocol(measurement)


# ---------------------------------------------------------------------------
# Fake
# ---------------------------------------------------------------------------


@dataclass
class FakeInfluxClient:
    """A complete in-memory InfluxDB stand-in.

    Load it with series (usually from :mod:`tsdb_anomaly_task.synthetic`), then
    hand it to ``preview()``, ``deploy()`` or the async runner exactly as you
    would a real client.  Every write and task operation is recorded so tests
    can assert on them, and both the sync and async halves of the protocol are
    implemented.

    Failure injection is first-class: set :attr:`fail_writes` to make the next
    N write attempts raise :class:`WriteError`, which is how the retry and
    backoff behaviour of the runner is tested without a network.

    Example:
        >>> from tsdb_anomaly_task import FakeInfluxClient, MetricQuery
        >>> from tsdb_anomaly_task.synthetic import make_series
        >>> client = FakeInfluxClient()
        >>> client.add_series("telemetry", make_series(count=10).series)
        >>> q = MetricQuery(bucket="telemetry", measurement="sensor", field="value")
        >>> len(client.read(q)[0])
        10
    """

    series: dict[str, list[Series]] = field(default_factory=dict)
    tasks: dict[str, TaskRef] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    writes: list[tuple[str, str, int]] = field(default_factory=list)
    reads: list[MetricQuery] = field(default_factory=list)
    fail_writes: int = 0
    write_delay: float = 0.0
    _next_id: int = 1

    # -- fixtures ----------------------------------------------------------

    def add_series(self, bucket: str, series: Series) -> None:
        """Register one series in ``bucket``."""
        self.series.setdefault(bucket, []).append(series)

    def add_synthetic(self, bucket: str, generated: SyntheticSeries) -> None:
        """Register the series half of a :class:`~tsdb_anomaly_task.synthetic.SyntheticSeries`."""
        self.add_series(bucket, generated.series)

    # -- reads -------------------------------------------------------------

    def read(self, query: MetricQuery, *, now: datetime | None = None) -> list[Series]:
        self.reads.append(query)
        moment = now or datetime.now(UTC)
        matches: list[Series] = []
        for candidate in self.series.get(query.bucket, []):
            if candidate.measurement != query.measurement or candidate.field != query.field:
                continue
            if not _tags_match(candidate.tags, query.filters):
                continue
            matches.append(self._apply_range(candidate, query, moment))
        return matches

    async def aread(self, query: MetricQuery, *, now: datetime | None = None) -> list[Series]:
        await asyncio.sleep(0)
        return self.read(query, now=now)

    @staticmethod
    def _apply_range(series: Series, query: MetricQuery, now: datetime) -> Series:
        """Honour ``range_start`` relative to the reference instant.

        Synthetic fixtures usually sit at a fixed historical timestamp, so a
        window that would select nothing is treated as "return everything"
        rather than silently yielding an empty preview.
        """
        start = now - query.lookback
        clipped = tuple(p for p in series.points if p.time >= start)
        if not clipped:
            return series
        return Series(
            measurement=series.measurement,
            field=series.field,
            tags=series.tags,
            points=clipped,
        )

    # -- writes ------------------------------------------------------------

    def write_flags(
        self,
        bucket: str,
        measurement: str,
        flags: Sequence[Flag],
        extra_tags: Mapping[str, str] | None = None,
    ) -> int:
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise WriteError(f"injected failure writing to {bucket!r}")
        records = [_render(flag, measurement, extra_tags) for flag in flags]
        self.written.extend(records)
        self.writes.append((bucket, measurement, len(records)))
        return len(records)

    async def awrite_flags(
        self,
        bucket: str,
        measurement: str,
        flags: Sequence[Flag],
        extra_tags: Mapping[str, str] | None = None,
    ) -> int:
        if self.write_delay:
            await asyncio.sleep(self.write_delay)
        else:
            await asyncio.sleep(0)
        return self.write_flags(bucket, measurement, flags, extra_tags)

    # -- tasks -------------------------------------------------------------

    def upsert_task(
        self, name: str, flux: str, *, every: str | None = None, offset: str | None = None
    ) -> TaskRef:
        existing = self.tasks.get(name)
        ref = TaskRef(
            id=existing.id if existing else self._allocate_id(),
            name=name,
            flux=flux,
            every=every,
            offset=offset,
            created=existing is None,
        )
        self.tasks[name] = ref
        return ref

    def find_task(self, name: str) -> TaskRef | None:
        return self.tasks.get(name)

    def delete_task(self, name: str) -> bool:
        return self.tasks.pop(name, None) is not None

    def _allocate_id(self) -> str:
        task_id = f"{self._next_id:016x}"
        self._next_id += 1
        return task_id


def _tags_match(tags: Mapping[str, str], filters: Mapping[str, str | Sequence[str]]) -> bool:
    for key, expected in filters.items():
        if key not in tags:
            return False
        if isinstance(expected, str):
            if expected != "*" and tags[key] != expected:
                return False
        elif tags[key] not in list(expected):
            return False
    return True


@dataclass
class RecordingClient:
    """Wraps another client and records every call, for debugging deployments."""

    inner: Any
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def read(self, query: MetricQuery, *, now: datetime | None = None) -> list[Series]:
        self.calls.append(("read", (query.bucket, query.measurement, query.field)))
        return self.inner.read(query, now=now)

    def write_flags(
        self,
        bucket: str,
        measurement: str,
        flags: Sequence[Flag],
        extra_tags: Mapping[str, str] | None = None,
    ) -> int:
        self.calls.append(("write_flags", (bucket, measurement, len(flags))))
        return self.inner.write_flags(bucket, measurement, flags, extra_tags)

    def upsert_task(
        self, name: str, flux: str, *, every: str | None = None, offset: str | None = None
    ) -> TaskRef:
        self.calls.append(("upsert_task", (name,)))
        return self.inner.upsert_task(name, flux, every=every, offset=offset)

    def find_task(self, name: str) -> TaskRef | None:
        self.calls.append(("find_task", (name,)))
        return self.inner.find_task(name)

    def delete_task(self, name: str) -> bool:
        self.calls.append(("delete_task", (name,)))
        return self.inner.delete_task(name)
