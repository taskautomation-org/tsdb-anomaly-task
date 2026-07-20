"""The :class:`AnomalyTask` façade: preview, compile, deploy, run."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .client import InfluxProtocol, TaskRef
from .detectors.base import Detector, FluxContext, FluxSupport, FluxUnsupportedError
from .flux import flux_header, write_stage
from .models import DetectionResult, Flag, MetricQuery, ResultsBucket, Schedule, Series

__all__ = ["AnomalyTask", "PreviewResult", "RunReport"]


@dataclass(frozen=True, slots=True)
class PreviewResult:
    """What a dry run found, without writing anything.

    ``preview()`` is the feature that makes a detector tunable: you point it at
    real history, look at what *would* have fired, and adjust before a single
    alert reaches anyone.
    """

    task: str
    detector: str
    series: tuple[Series, ...]
    results: tuple[DetectionResult, ...]
    evaluated_at: datetime

    @property
    def flags(self) -> tuple[Flag, ...]:
        """Every flag across every series, oldest first."""
        out: list[Flag] = []
        for result in self.results:
            out.extend(result.flags)
        return tuple(sorted(out, key=lambda f: (f.time, f.series_key)))

    @property
    def points_evaluated(self) -> int:
        return sum(r.points_evaluated for r in self.results)

    @property
    def flag_rate(self) -> float:
        """Fraction of evaluated points that were flagged."""
        total = self.points_evaluated
        return len(self.flags) / total if total else 0.0

    @property
    def notes(self) -> tuple[str, ...]:
        seen: list[str] = []
        for result in self.results:
            for note in result.notes:
                if note not in seen:
                    seen.append(note)
        return tuple(seen)

    @property
    def usable(self) -> bool:
        """True when at least one series produced a judgeable point."""
        return any(r.usable for r in self.results)

    def render(self, limit: int = 12, width: int = 74) -> str:
        """Render a terminal-friendly summary of the dry run."""
        lines = [
            f"preview: {self.task}  [{self.detector}]",
            f"  series {len(self.series)}   points {self.points_evaluated}   "
            f"flags {len(self.flags)}   rate {self.flag_rate * 100:.2f}%",
            "",
        ]
        flags = self.flags
        if not flags:
            lines.append("  no anomalies in the previewed window")
        else:
            lines.append(f"  {'time (UTC)':<21}{'series':<18}{'value':>10}{'score':>10}  severity")
            lines.append("  " + "-" * (width - 2))
            for flag in flags[:limit]:
                key = flag.label
                if len(key) > 17:
                    key = key[:16] + "…"
                lines.append(
                    f"  {flag.time:%Y-%m-%d %H:%M:%S}  {key:<18}"
                    f"{flag.value:>10.2f}{flag.score:>10.2f}  {flag.severity}"
                )
            if len(flags) > limit:
                lines.append(f"  … and {len(flags) - limit} more")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.flags)


@dataclass(frozen=True, slots=True)
class RunReport:
    """The outcome of one client-side execution."""

    task: str
    series_read: int
    flags: tuple[Flag, ...]
    written: int
    notes: tuple[str, ...] = ()
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __len__(self) -> int:
        return len(self.flags)


@dataclass(frozen=True, slots=True)
class AnomalyTask:
    """A metric, a detector, a schedule and a destination — as one object.

    The same definition drives both execution modes:

    * :meth:`to_flux` / :meth:`deploy` compile the detector into a Flux script
      and register it as a native InfluxDB task, where it runs on the server
      with no Python process involved.
    * :meth:`run_once` and :class:`~tsdb_anomaly_task.runner.AsyncAnomalyRunner`
      execute the same detector in Python, for detectors whose statistics Flux
      cannot express.

    :attr:`execution_mode` reports which one applies, and
    :attr:`flux_support` explains why when the answer is ``"client"``.

    Args:
        name: Task name; also the name used on the server.
        query: The metric to watch.
        detector: How to judge it.
        schedule: When to run.
        output: Where flags are written.
        description: Optional human note, included in the generated header.
    """

    name: str
    query: MetricQuery
    detector: Detector
    schedule: Schedule
    output: ResultsBucket
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AnomalyTask.name must not be empty")
        if self.output.bucket == self.query.bucket:
            # Writing flags into the source bucket makes the task read its own
            # output on the next run, which is a feedback loop, not a feature.
            raise ValueError(
                f"output bucket {self.output.bucket!r} must differ from the source "
                f"bucket; writing flags back into the queried bucket makes the task "
                f"consume its own output"
            )

    # -- compilation -------------------------------------------------------

    def _context(self) -> FluxContext:
        return FluxContext(
            query_flux=self.query.to_flux(),
            flag_measurement=self.output.flag_measurement,
            group_by=self.query.group_by,
        )

    @property
    def flux_support(self) -> FluxSupport:
        """Whether this task, as configured, compiles to a server-side script."""
        return self.detector.flux_support(self._context())

    @property
    def execution_mode(self) -> str:
        """``"server"`` if the detector compiles to Flux, otherwise ``"client"``."""
        return "server" if self.flux_support else "client"

    def to_flux(self) -> str:
        """Compile the task to a complete, deployable Flux script.

        Raises:
            FluxUnsupportedError: if the detector cannot be expressed in Flux.
                The message states exactly which part is the obstacle; run the
                task client-side instead.
        """
        context = self._context()
        support = self.detector.flux_support(context)
        if not support:
            raise FluxUnsupportedError(
                f"{self.detector.name} cannot be compiled to Flux: {support.reason}"
            )

        notes = list(getattr(self.detector, "flux_notes", tuple)())
        if self.description:
            notes.insert(0, self.description)

        blocks = [
            flux_header(self.name, str(self.detector), notes),
            'import "math"',
            self.schedule.to_task_options(self.name),
            self.detector.to_flux(context)
            + "\n"
            + write_stage(
                flag_measurement=self.output.flag_measurement,
                detector=self.detector.name,
                bucket=self.output.bucket,
                org=self.output.org,
                group_by=self.query.group_by,
                extra_tags=dict(self.output.extra_tags),
            ),
        ]
        return "\n\n".join(blocks) + "\n"

    # -- dry run -----------------------------------------------------------

    def preview(self, client: InfluxProtocol, *, now: datetime | None = None) -> PreviewResult:
        """Evaluate the detector against real data without writing anything.

        Use this before every deploy, and again whenever you change a
        parameter.  It answers the only question that matters at tuning time:
        *how many alerts would this configuration have produced yesterday?*

        Args:
            client: Any :class:`~tsdb_anomaly_task.client.InfluxProtocol`.
            now: Reference instant; defaults to the current UTC time.
        """
        moment = now or datetime.now(UTC)
        series = tuple(client.read(self.query, now=moment))
        results = tuple(self.detector.evaluate_all(series, now=moment))
        return PreviewResult(
            task=self.name,
            detector=str(self.detector),
            series=series,
            results=results,
            evaluated_at=moment,
        )

    # -- execution ---------------------------------------------------------

    def run_once(
        self, client: InfluxProtocol, *, now: datetime | None = None, dry_run: bool = False
    ) -> RunReport:
        """Execute the detector client-side and write any flags found."""
        preview = self.preview(client, now=now)
        flags = preview.flags
        written = 0
        if flags and not dry_run:
            written = client.write_flags(
                self.output.bucket,
                self.output.flag_measurement,
                flags,
                self.output.extra_tags,
            )
        return RunReport(
            task=self.name,
            series_read=len(preview.series),
            flags=flags,
            written=written,
            notes=preview.notes,
            evaluated_at=preview.evaluated_at,
        )

    # -- deployment --------------------------------------------------------

    def deploy(self, client: InfluxProtocol) -> TaskRef:
        """Create or update the server-side InfluxDB task.

        Raises:
            FluxUnsupportedError: if the detector only runs client-side.  Use
                :class:`~tsdb_anomaly_task.runner.AsyncAnomalyRunner` for those.
        """
        flux = self.to_flux()  # raises with the reason if unsupported
        return client.upsert_task(
            self.name,
            flux,
            every=self.schedule.every,
            offset=self.schedule.offset,
        )

    def undeploy(self, client: InfluxProtocol) -> bool:
        """Delete the server-side task, if it exists."""
        return client.delete_task(self.name)

    def deployed(self, client: InfluxProtocol) -> TaskRef | None:
        """Return the currently deployed task, if any."""
        return client.find_task(self.name)

    # -- introspection -----------------------------------------------------

    def summary(self, width: int = 78) -> str:
        """A one-screen human description of the task, wrapped to ``width``."""
        schedule = (
            f"every {self.schedule.every}" if self.schedule.every else f"cron {self.schedule.cron}"
        )
        if self.schedule.offset:
            schedule += f" offset {self.schedule.offset}"
        lines = [
            f"task     {self.name}",
            f"metric   {self.query.bucket}/{self.query.measurement}.{self.query.field}",
            f"detector {self.detector}",
            f"schedule {schedule}",
            f"output   {self.output.bucket}/{self.output.flag_measurement}",
            f"mode     {self.execution_mode}",
        ]
        if self.execution_mode == "client":
            wrapped = textwrap.wrap(
                self.flux_support.reason,
                width=max(width, 40),
                initial_indent="reason   ",
                subsequent_indent="         ",
            )
            lines.extend(wrapped)
        return "\n".join(lines)
