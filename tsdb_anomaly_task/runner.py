"""Client-side execution: an asyncio runner with exponential-backoff writes.

Detectors that Flux cannot express still need to run on a schedule.  This
module provides that: an asyncio loop that reads each task's window,
evaluates the detector in Python and writes the flags back, concurrently
across tasks and with a retry policy in front of every write.

Two ideas from the InfluxDB client-orchestration playbook are load-bearing here
and are worth reading up on if you are building something similar:
`using asyncio with the InfluxDB client v2
<https://taskautomation.org/automated-task-scheduling-orchestration/python-client-orchestration-patterns/using-python-asyncio-with-influxdb-client-v2-for-batch-tasks/>`_
and `exponential backoff and retry for client writes
<https://taskautomation.org/automated-task-scheduling-orchestration/python-client-orchestration-patterns/exponential-backoff-and-retry-for-influxdb-client-writes/>`_.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from .client import AsyncInfluxProtocol, WriteError
from .duration import parse_duration
from .task import AnomalyTask, RunReport

__all__ = ["AsyncAnomalyRunner", "RetryPolicy", "RunnerStats"]

log = logging.getLogger("tsdb_anomaly_task.runner")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Delay before attempt *n* (1-based) is drawn uniformly from
    ``[0, min(base * factor**(n-1), max_delay)]``.  The jitter is not
    decoration: a fleet of runners that all back off by exactly the same
    doubling sequence will retry in lockstep and re-create the very write
    storm that knocked the server over.  Randomising the wait spreads them out.

    Args:
        attempts: Total attempts, including the first.
        base_delay: Delay ceiling for the first retry, in seconds.
        factor: Multiplier applied per attempt.
        max_delay: Upper bound on any single delay, in seconds.
        jitter: Apply full jitter.  Disable only in tests that assert timing.
    """

    attempts: int = 5
    base_delay: float = 0.5
    factor: float = 2.0
    max_delay: float = 30.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.factor < 1:
            raise ValueError("factor must be >= 1")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")

    def delay_for(self, attempt: int, rng: random.Random | None = None) -> float:
        """Seconds to wait before ``attempt`` (1-based; attempt 1 never waits)."""
        if attempt <= 1:
            return 0.0
        ceiling = min(self.base_delay * self.factor ** (attempt - 2), self.max_delay)
        if not self.jitter:
            return ceiling
        return (rng or random).uniform(0.0, ceiling)


@dataclass
class RunnerStats:
    """Cumulative counters for a runner, useful as a health metric."""

    cycles: int = 0
    runs: int = 0
    flags: int = 0
    written: int = 0
    write_attempts: int = 0
    retries: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "cycles": self.cycles,
            "runs": self.runs,
            "flags": self.flags,
            "written": self.written,
            "write_attempts": self.write_attempts,
            "retries": self.retries,
            "failures": self.failures,
        }


@dataclass
class AsyncAnomalyRunner:
    """Runs client-side anomaly tasks on a schedule.

    Args:
        client: Anything implementing
            :class:`~tsdb_anomaly_task.client.AsyncInfluxProtocol`.
        tasks: The tasks to execute each cycle.
        interval: Cycle period.  Defaults to the shortest schedule among the
            tasks, so a runner holding a ``1m`` and a ``5m`` task ticks every
            minute and each task is evaluated on its own window.
        offset: Delay applied before the first cycle, mirroring an InfluxDB
            task offset.  Give late-arriving sensor data time to land before
            you read the window it belongs to, or the newest bucket will look
            like a dip on every single run.
        retry: Write retry policy.
        concurrency: Maximum tasks evaluated in parallel.
        on_report: Optional callback invoked with each :class:`RunReport`.

    Example:
        >>> import asyncio
        >>> from tsdb_anomaly_task import AsyncAnomalyRunner, FakeInfluxClient
        >>> runner = AsyncAnomalyRunner(client=FakeInfluxClient(), tasks=[])
        >>> asyncio.run(runner.run_cycle()) == []
        True
    """

    client: AsyncInfluxProtocol
    tasks: Sequence[AnomalyTask]
    interval: str | timedelta | None = None
    offset: str | timedelta | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    concurrency: int = 8
    on_report: Callable[[RunReport], None] | None = None
    stats: RunnerStats = field(default_factory=RunnerStats)
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._semaphore = asyncio.Semaphore(self.concurrency)

    # -- scheduling --------------------------------------------------------

    @property
    def cycle_seconds(self) -> float:
        """Seconds between cycles."""
        if self.interval is not None:
            return parse_duration(self.interval).total_seconds()
        intervals = [
            t.schedule.interval.total_seconds()
            for t in self.tasks
            if t.schedule.interval is not None
        ]
        if not intervals:
            raise ValueError(
                "cannot infer a cycle interval: pass interval=, or give the tasks "
                "an 'every' schedule (cron schedules are server-side only)"
            )
        return min(intervals)

    # -- execution ---------------------------------------------------------

    async def run_cycle(self, *, now: datetime | None = None) -> list[RunReport]:
        """Evaluate every task once, concurrently, and write the flags found."""
        moment = now or datetime.now(UTC)
        self.stats.cycles += 1
        reports = await asyncio.gather(
            *(self._run_task(task, moment) for task in self.tasks),
            return_exceptions=True,
        )
        out: list[RunReport] = []
        for task, report in zip(self.tasks, reports, strict=True):
            if isinstance(report, BaseException):
                self.stats.failures += 1
                log.error("task %s failed: %s", task.name, report)
                continue
            out.append(report)
        return out

    async def run_forever(
        self, *, stop: asyncio.Event | None = None, max_cycles: int | None = None
    ) -> RunnerStats:
        """Loop until ``stop`` is set or ``max_cycles`` cycles have run."""
        stop = stop or asyncio.Event()
        if self.offset is not None:
            await self.sleeper(parse_duration(self.offset).total_seconds())
        period = self.cycle_seconds
        cycles = 0
        while not stop.is_set():
            await self.run_cycle()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            await self.sleeper(period)
        return self.stats

    async def _run_task(self, task: AnomalyTask, now: datetime) -> RunReport:
        async with self._semaphore:
            series = await self.client.aread(task.query, now=now)
            results = task.detector.evaluate_all(series, now=now)
            flags = tuple(
                sorted(
                    (flag for result in results for flag in result.flags),
                    key=lambda f: (f.time, f.series_key),
                )
            )
            notes = tuple(dict.fromkeys(n for r in results for n in r.notes))
            self.stats.runs += 1
            self.stats.flags += len(flags)

            written = 0
            if flags:
                written = await self._write_with_retry(task, flags)
                self.stats.written += written

            report = RunReport(
                task=task.name,
                series_read=len(series),
                flags=flags,
                written=written,
                notes=notes,
                evaluated_at=now,
            )
            if self.on_report is not None:
                self.on_report(report)
            return report

    async def _write_with_retry(self, task: AnomalyTask, flags: Sequence) -> int:
        """Write flags, retrying transient failures with exponential backoff."""
        last: Exception | None = None
        for attempt in range(1, self.retry.attempts + 1):
            delay = self.retry.delay_for(attempt, self.rng)
            if delay:
                self.stats.retries += 1
                log.warning(
                    "retrying write for task %s (attempt %d/%d) in %.2fs",
                    task.name,
                    attempt,
                    self.retry.attempts,
                    delay,
                )
                await self.sleeper(delay)
            self.stats.write_attempts += 1
            try:
                return await self.client.awrite_flags(
                    task.output.bucket,
                    task.output.flag_measurement,
                    flags,
                    task.output.extra_tags,
                )
            except WriteError as exc:
                last = exc
            except Exception as exc:
                last = WriteError(str(exc))
        assert last is not None
        raise last
