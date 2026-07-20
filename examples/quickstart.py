#!/usr/bin/env python
"""End-to-end walkthrough against the in-memory fake client.

Swap ``FakeInfluxClient`` for ``InfluxClient(InfluxDBClient(...))`` and this
script works unchanged against a real server.

Run from a clone::

    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsdb_anomaly_task import (
    AnomalyTask,
    AsyncAnomalyRunner,
    FakeInfluxClient,
    MADDetector,
    MetricQuery,
    ResultsBucket,
    RetryPolicy,
    Schedule,
    ThresholdDetector,
    sweep_parameter,
)
from tsdb_anomaly_task.synthetic import DEFAULT_START, Anomaly, make_series


def seeded_client() -> FakeInfluxClient:
    """Two hosts of synthetic telemetry with known faults planted in each."""
    client = FakeInfluxClient()
    for host, seed, faults in (
        ("edge-01", 1, [Anomaly(index=200, kind="spike", magnitude=18.0, length=3)]),
        ("edge-02", 2, [Anomaly(index=310, kind="dip", magnitude=16.0, length=2)]),
    ):
        client.add_synthetic(
            "telemetry",
            make_series(
                measurement="cpu",
                field_name="usage",
                tags={"host": host},
                count=576,
                interval="5m",
                base=42.0,
                noise=1.4,
                daily_amplitude=6.0,
                anomalies=faults,
                seed=seed,
            ),
        )
    return client


QUERY = MetricQuery(
    bucket="telemetry",
    measurement="cpu",
    field="usage",
    filters={"host": "*"},
    group_by=["host"],
    range_start="-48h",
)
OUTPUT = ResultsBucket("anomalies", flag_measurement="anomaly", extra_tags={"env": "demo"})
SCHEDULE = Schedule(every="5m", offset="30s")
NOW = DEFAULT_START + timedelta(days=2)


def main() -> int:
    client = seeded_client()

    # 1. A server-side task: compiles to Flux, runs on the database.
    ceiling = AnomalyTask(
        name="cpu-ceiling",
        query=QUERY,
        detector=ThresholdDetector(upper=70.0, hysteresis=0.0, consecutive_points=2),
        schedule=SCHEDULE,
        output=OUTPUT,
    )
    print(ceiling.summary(), "\n")

    # 2. Tune before deploying. One query, replayed at every candidate value.
    robust = AnomalyTask(
        name="cpu-anomalies",
        query=QUERY,
        detector=MADDetector(k=3.5, window="6h", min_points=24, consecutive_points=2),
        schedule=SCHEDULE,
        output=OUTPUT,
    )
    sweep = sweep_parameter(
        robust, client, parameter="k", values=[2.5, 3.0, 3.5, 4.0, 5.0], now=NOW
    )
    print(sweep.render(), "\n")

    # 3. Dry-run the chosen configuration against real history.
    preview = robust.preview(client, now=NOW)
    print(preview.render(limit=5), "\n")

    # 4. Deploy what the server can run...
    ref = ceiling.deploy(client)
    print(f"deployed {ref.name} (id {ref.id})\n")

    # 5. ...and run the rest client-side, with retrying writes.
    runner = AsyncAnomalyRunner(
        client=client,
        tasks=[robust],
        retry=RetryPolicy(attempts=5, base_delay=0.2),
    )
    reports = asyncio.run(runner.run_cycle(now=NOW))
    for report in reports:
        print(f"{report.task}: {len(report.flags)} flag(s), {report.written} written")
    print("\nfirst record written:")
    print(f"  {client.written[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
