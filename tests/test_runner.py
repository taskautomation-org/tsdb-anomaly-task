from __future__ import annotations

import asyncio
import random
from datetime import timedelta

import pytest

from tsdb_anomaly_task import (
    AnomalyTask,
    AsyncAnomalyRunner,
    FakeInfluxClient,
    MetricQuery,
    ResultsBucket,
    RetryPolicy,
    Schedule,
    SeasonalDetector,
    ThresholdDetector,
    WriteError,
)


def build(detector=None, *, every: str = "5m", name: str = "t") -> AnomalyTask:
    return AnomalyTask(
        name=name,
        query=MetricQuery(
            bucket="telemetry",
            measurement="sensor",
            field="temperature",
            filters={"host": "*"},
            group_by=["host"],
            range_start="-48h",
        ),
        detector=detector or ThresholdDetector(upper=25.0, lower=17.0),
        schedule=Schedule(every=every),
        output=ResultsBucket("anomalies"),
    )


class FakeClock:
    """Records every sleep instead of performing it."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)
        await asyncio.sleep(0)


# -- retry policy -----------------------------------------------------------


def test_retry_policy_validation() -> None:
    with pytest.raises(ValueError, match="attempts"):
        RetryPolicy(attempts=0)
    with pytest.raises(ValueError, match="base_delay"):
        RetryPolicy(base_delay=0)
    with pytest.raises(ValueError, match="factor"):
        RetryPolicy(factor=0.5)
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(base_delay=10, max_delay=1)


def test_backoff_doubles_and_is_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, factor=2.0, max_delay=8.0, jitter=False)
    assert [policy.delay_for(n) for n in range(1, 7)] == [0.0, 1.0, 2.0, 4.0, 8.0, 8.0]


def test_full_jitter_stays_within_the_envelope() -> None:
    policy = RetryPolicy(base_delay=1.0, factor=2.0, max_delay=8.0, jitter=True)
    rng = random.Random(0)
    for attempt in range(2, 8):
        ceiling = min(1.0 * 2 ** (attempt - 2), 8.0)
        for _ in range(50):
            assert 0.0 <= policy.delay_for(attempt, rng) <= ceiling


def test_jitter_actually_varies() -> None:
    policy = RetryPolicy(base_delay=4.0, jitter=True)
    rng = random.Random(1)
    draws = {policy.delay_for(3, rng) for _ in range(20)}
    assert len(draws) > 1


# -- cycles -----------------------------------------------------------------


async def test_run_cycle_evaluates_and_writes(client, now) -> None:
    runner = AsyncAnomalyRunner(client=client, tasks=[build()])
    reports = await runner.run_cycle(now=now)
    assert len(reports) == 1
    assert reports[0].written == len(reports[0].flags) > 0
    assert runner.stats.runs == 1
    assert runner.stats.cycles == 1
    assert runner.stats.write_attempts == 1
    assert runner.stats.retries == 0
    assert runner.stats.as_dict()["flags"] == len(reports[0].flags)


async def test_run_cycle_with_no_tasks() -> None:
    assert await AsyncAnomalyRunner(client=FakeInfluxClient(), tasks=[]).run_cycle() == []


async def test_client_side_detector_runs_here(client, now) -> None:
    task = build(SeasonalDetector(period="hour-of-day", k=4.0, training="2d"))
    assert task.execution_mode == "client"
    reports = await AsyncAnomalyRunner(client=client, tasks=[task]).run_cycle(now=now)
    assert reports[0].series_read == 1


async def test_reports_are_passed_to_the_callback(client, now) -> None:
    seen = []
    runner = AsyncAnomalyRunner(client=client, tasks=[build()], on_report=seen.append)
    await runner.run_cycle(now=now)
    assert len(seen) == 1


async def test_tasks_run_concurrently_under_the_semaphore(client, now) -> None:
    tasks = [build(name=f"t{i}") for i in range(5)]
    runner = AsyncAnomalyRunner(client=client, tasks=tasks, concurrency=2)
    reports = await runner.run_cycle(now=now)
    assert len(reports) == 5
    assert runner.stats.runs == 5


def test_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        AsyncAnomalyRunner(client=FakeInfluxClient(), tasks=[], concurrency=0)


async def test_a_failing_task_does_not_abort_the_cycle(client, now) -> None:
    client.fail_writes = 99
    runner = AsyncAnomalyRunner(
        client=client,
        tasks=[build(name="a"), build(name="b")],
        retry=RetryPolicy(attempts=1),
        sleeper=FakeClock(),
    )
    reports = await runner.run_cycle(now=now)
    assert reports == []
    assert runner.stats.failures == 2


# -- retry behaviour --------------------------------------------------------


async def test_write_retries_until_it_succeeds(client, now) -> None:
    client.fail_writes = 2
    clock = FakeClock()
    runner = AsyncAnomalyRunner(
        client=client,
        tasks=[build()],
        retry=RetryPolicy(attempts=5, base_delay=1.0, factor=2.0, jitter=False),
        sleeper=clock,
    )
    reports = await runner.run_cycle(now=now)
    assert reports[0].written > 0
    assert runner.stats.write_attempts == 3
    assert runner.stats.retries == 2
    assert clock.slept == [1.0, 2.0]


async def test_write_gives_up_after_the_configured_attempts(client, now) -> None:
    client.fail_writes = 99
    clock = FakeClock()
    runner = AsyncAnomalyRunner(
        client=client,
        tasks=[build()],
        retry=RetryPolicy(attempts=3, base_delay=0.5, jitter=False),
        sleeper=clock,
    )
    with pytest.raises(WriteError, match="injected failure"):
        await runner._run_task(runner.tasks[0], now)
    assert runner.stats.write_attempts == 3
    assert len(clock.slept) == 2


async def test_unexpected_exceptions_are_normalised_to_write_errors(client, now) -> None:
    class Exploding(FakeInfluxClient):
        async def awrite_flags(self, *args, **kwargs):
            raise RuntimeError("connection reset")

    exploding = Exploding(series=client.series)
    runner = AsyncAnomalyRunner(
        client=exploding,
        tasks=[build()],
        retry=RetryPolicy(attempts=2, base_delay=0.1, jitter=False),
        sleeper=FakeClock(),
    )
    with pytest.raises(WriteError, match="connection reset"):
        await runner._run_task(runner.tasks[0], now)


async def test_no_flags_means_no_write_attempt(client, now) -> None:
    runner = AsyncAnomalyRunner(client=client, tasks=[build(ThresholdDetector(upper=500.0))])
    await runner.run_cycle(now=now)
    assert runner.stats.write_attempts == 0


# -- scheduling -------------------------------------------------------------


def test_cycle_interval_defaults_to_the_shortest_task_schedule() -> None:
    runner = AsyncAnomalyRunner(
        client=FakeInfluxClient(),
        tasks=[build(every="5m", name="a"), build(every="1m", name="b")],
    )
    assert runner.cycle_seconds == 60.0


def test_explicit_interval_wins() -> None:
    runner = AsyncAnomalyRunner(
        client=FakeInfluxClient(), tasks=[build(every="5m")], interval="90s"
    )
    assert runner.cycle_seconds == 90.0


def test_cron_only_tasks_need_an_explicit_interval() -> None:
    task = AnomalyTask(
        name="t",
        query=MetricQuery(bucket="telemetry", measurement="s", field="f"),
        detector=ThresholdDetector(upper=1.0),
        schedule=Schedule(cron="0 * * * *"),
        output=ResultsBucket("anomalies"),
    )
    runner = AsyncAnomalyRunner(client=FakeInfluxClient(), tasks=[task])
    with pytest.raises(ValueError, match="cannot infer a cycle interval"):
        _ = runner.cycle_seconds


async def test_run_forever_honours_offset_and_max_cycles(client) -> None:
    clock = FakeClock()
    runner = AsyncAnomalyRunner(
        client=client, tasks=[build(every="5m")], offset="30s", sleeper=clock
    )
    stats = await runner.run_forever(max_cycles=3)
    assert stats.cycles == 3
    # One offset sleep, then a period sleep between cycles (not after the last).
    assert clock.slept == [30.0, 300.0, 300.0]


async def test_run_forever_stops_when_the_event_is_set(client) -> None:
    clock = FakeClock()
    stop = asyncio.Event()
    runner = AsyncAnomalyRunner(client=client, tasks=[build()], sleeper=clock)

    def halt(_report) -> None:
        stop.set()

    runner.on_report = halt
    stats = await runner.run_forever(stop=stop)
    assert stats.cycles == 1


async def test_write_delay_is_awaited(client, now) -> None:
    client.write_delay = 0.001
    runner = AsyncAnomalyRunner(client=client, tasks=[build()])
    reports = await runner.run_cycle(now=now)
    assert reports[0].written > 0


async def test_reads_use_one_reference_instant_for_the_whole_cycle(client, spiky) -> None:
    """Deadman decisions must not drift between tasks in the same cycle."""
    from tsdb_anomaly_task import DeadmanDetector

    quiet = spiky.series.end + timedelta(hours=5)
    runner = AsyncAnomalyRunner(
        client=client,
        tasks=[build(DeadmanDetector(tolerance="30m"), name=f"d{i}") for i in range(3)],
    )
    reports = await runner.run_cycle(now=quiet)
    assert {len(r.flags) for r in reports} == {1}
