"""``python -m tsdb_anomaly_task`` — a runnable tour of the library.

Everything the CLI shows runs against :class:`~tsdb_anomaly_task.client.FakeInfluxClient`
loaded with deterministic synthetic sensor data, so the output is reproducible
and no server is needed.  The README's demo is literally this command's output.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta

from .client import FakeInfluxClient
from .detectors import (
    DeadmanDetector,
    Detector,
    MADDetector,
    RateOfChangeDetector,
    SeasonalDetector,
    ThresholdDetector,
)
from .models import MetricQuery, ResultsBucket, Schedule
from .runner import AsyncAnomalyRunner, RetryPolicy
from .synthetic import DEFAULT_START, Anomaly, make_series
from .task import AnomalyTask
from .tuning import sweep_parameter

__all__ = ["build_scenario", "main"]

BUCKET = "telemetry"
RESULTS = "anomalies"
MEASUREMENT = "sensor"
FIELD = "temperature"

#: The demo evaluates as-of this instant so every run prints identical output.
DEMO_NOW = DEFAULT_START + timedelta(days=2)


def build_scenario() -> tuple[FakeInfluxClient, dict[str, AnomalyTask], datetime]:
    """Build the demo fleet: three sensors, each with a different fault."""
    client = FakeInfluxClient()

    # A well-behaved sensor with two brief spikes and one sustained excursion.
    client.add_synthetic(
        BUCKET,
        make_series(
            measurement=MEASUREMENT,
            field_name=FIELD,
            tags={"host": "edge-01"},
            count=576,
            interval="5m",
            base=21.0,
            noise=0.35,
            daily_amplitude=2.5,
            anomalies=[
                Anomaly(index=140, kind="spike", magnitude=14.0),
                Anomaly(index=305, kind="spike", magnitude=18.0, length=4),
                Anomaly(index=470, kind="dip", magnitude=15.0, length=3),
            ],
            seed=11,
        ),
    )
    # A sensor that drifts, then glitches hard for one sample.
    client.add_synthetic(
        BUCKET,
        make_series(
            measurement=MEASUREMENT,
            field_name=FIELD,
            tags={"host": "edge-02"},
            count=576,
            interval="5m",
            base=19.5,
            noise=0.5,
            daily_amplitude=2.5,
            anomalies=[
                Anomaly(index=210, kind="drift", magnitude=16.0, length=20),
                Anomaly(index=400, kind="spike", magnitude=40.0),
            ],
            seed=22,
        ),
    )
    # A sensor whose uplink drops out for four hours and never comes back.
    client.add_synthetic(
        BUCKET,
        make_series(
            measurement=MEASUREMENT,
            field_name=FIELD,
            tags={"host": "edge-03"},
            count=528,
            interval="5m",
            base=20.2,
            noise=0.3,
            daily_amplitude=2.5,
            anomalies=[Anomaly(index=300, kind="gap", length=48)],
            seed=33,
        ),
    )

    query = MetricQuery(
        bucket=BUCKET,
        measurement=MEASUREMENT,
        field=FIELD,
        filters={"host": "*"},
        group_by=["host"],
        range_start="-48h",
    )
    output = ResultsBucket(RESULTS, flag_measurement="anomaly", extra_tags={"env": "demo"})

    detectors: dict[str, Detector] = {
        "threshold": ThresholdDetector(upper=26.0, lower=14.0, consecutive_points=2),
        "mad": MADDetector(k=3.5, window="6h", min_points=24, consecutive_points=2),
        "seasonal": SeasonalDetector(period="hour-of-day", k=4.0, training="2d"),
        "deadman": DeadmanDetector(tolerance="30m"),
        "rate": RateOfChangeDetector(max_rate=0.05, per="1s"),
    }

    tasks = {
        key: AnomalyTask(
            name=f"{FIELD}-{key}",
            query=query,
            detector=detector,
            schedule=Schedule(every="5m", offset="30s"),
            output=output,
            description=f"Fleet {FIELD} watch ({key}).",
        )
        for key, detector in detectors.items()
    }
    return client, tasks, DEMO_NOW


# ---------------------------------------------------------------------------


def _cmd_detectors(_: argparse.Namespace) -> int:
    _, tasks, _now = build_scenario()
    print("detector      mode    notes")
    print("-" * 78)
    for key, task in tasks.items():
        support = task.flux_support
        reason = "compiles to Flux" if support else support.reason
        if len(reason) > 58:
            reason = reason[:57] + "…"
        print(f"{key:<14}{task.execution_mode:<8}{reason}")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    client, tasks, now = build_scenario()
    task = _pick(tasks, args.detector)
    print(task.summary())
    print()
    print(task.preview(client, now=now).render(limit=args.limit))
    return 0


def _cmd_flux(args: argparse.Namespace) -> int:
    _client, tasks, _now = build_scenario()
    task = _pick(tasks, args.detector)
    support = task.flux_support
    if not support:
        print(f"{task.detector.name} does not compile to Flux:")
        print(f"  {support.reason}")
        print("\nRun it client-side with AsyncAnomalyRunner instead.")
        return 1
    print(task.to_flux())
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    client, tasks, now = build_scenario()
    task = _pick(tasks, args.detector)
    values = [float(v) for v in args.values] if args.values else [2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    result = sweep_parameter(
        task,
        client,
        parameter=args.parameter,
        values=values,
        now=now,
        target_rate=args.target_rate,
    )
    print(result.render())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    client, tasks, now = build_scenario()
    task = _pick(tasks, args.detector)
    runner = AsyncAnomalyRunner(
        client=client,
        tasks=[task],
        retry=RetryPolicy(attempts=4, base_delay=0.1),
    )
    reports = asyncio.run(runner.run_cycle(now=now))
    for report in reports:
        print(
            f"{report.task}: read {report.series_read} series, "
            f"{len(report.flags)} flag(s), wrote {report.written} record(s)"
        )
    if client.written:
        print("\nfirst records written:")
        for record in client.written[:3]:
            print(f"  {record}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    client, tasks, now = build_scenario()

    print("=" * 78)
    print("1. What can run where")
    print("=" * 78)
    _cmd_detectors(args)

    print()
    print("=" * 78)
    print("2. Dry-run the MAD detector against real history")
    print("=" * 78)
    mad = tasks["mad"]
    print(mad.summary())
    print()
    print(mad.preview(client, now=now).render(limit=8))

    print()
    print("=" * 78)
    print("3. Tune k before anyone gets paged")
    print("=" * 78)
    print(
        sweep_parameter(
            mad, client, parameter="k", values=[2.5, 3.0, 3.5, 4.0, 5.0], now=now
        ).render()
    )

    print()
    print("=" * 78)
    print("4. Deadman: the sensor that went quiet")
    print("=" * 78)
    print(tasks["deadman"].preview(client, now=now).render(limit=5))

    print()
    print("=" * 78)
    print("5. Deploy the threshold detector as a native InfluxDB task")
    print("=" * 78)
    ref = tasks["threshold"].deploy(client)
    print(f"task {ref.name} -> id {ref.id} ({'created' if ref.created else 'updated'})")
    print()
    print(tasks["threshold"].to_flux())
    return 0


def _pick(tasks: dict[str, AnomalyTask], key: str) -> AnomalyTask:
    if key not in tasks:
        raise SystemExit(f"unknown detector {key!r}; choose from {', '.join(tasks)}")
    return tasks[key]


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m tsdb_anomaly_task``."""
    parser = argparse.ArgumentParser(
        prog="python -m tsdb_anomaly_task",
        description="Explore the library against deterministic synthetic sensor data.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="the full walkthrough (this is what the README shows)")
    sub.add_parser("detectors", help="list detectors and their execution mode")

    choices = ["threshold", "mad", "seasonal", "deadman", "rate"]

    p_preview = sub.add_parser("preview", help="dry-run a detector, writing nothing")
    p_preview.add_argument("detector", choices=choices)
    p_preview.add_argument("--limit", type=int, default=12)

    p_flux = sub.add_parser("flux", help="print the generated Flux script")
    p_flux.add_argument("detector", choices=choices)

    p_sweep = sub.add_parser("sweep", help="sweep a parameter and report flag rates")
    p_sweep.add_argument("detector", choices=choices)
    p_sweep.add_argument("--parameter", default="k")
    p_sweep.add_argument("--values", nargs="*")
    p_sweep.add_argument("--target-rate", type=float, default=None)

    p_run = sub.add_parser("run", help="execute one client-side cycle against the fake client")
    p_run.add_argument("detector", choices=choices)

    args = parser.parse_args(argv)
    handlers = {
        None: _cmd_demo,
        "demo": _cmd_demo,
        "detectors": _cmd_detectors,
        "preview": _cmd_preview,
        "flux": _cmd_flux,
        "sweep": _cmd_sweep,
        "run": _cmd_run,
    }
    return handlers[args.command](args)
