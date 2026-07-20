"""Golden-file assertions on generated Flux.

The scripts in ``tests/golden/`` are the contract for what gets deployed to a
server.  Any change to generation shows up here as a diff, which is exactly
what you want for code you cannot unit-test by executing it.

Regenerate deliberately with ``UPDATE_GOLDEN=1 pytest tests/test_flux_generation.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tsdb_anomaly_task import (
    AnomalyTask,
    DeadmanDetector,
    MADDetector,
    MetricQuery,
    RateOfChangeDetector,
    ResultsBucket,
    Schedule,
    SeasonalDetector,
    ThresholdDetector,
)
from tsdb_anomaly_task.flux import FLUX_DOC_LINK, flux_float, flux_header, severity_expr
from tsdb_anomaly_task.task import FluxUnsupportedError

GOLDEN = Path(__file__).parent / "golden"


def assert_golden(name: str, actual: str) -> None:
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, f"generated Flux drifted from {path.name}"


def make_task(detector, **overrides) -> AnomalyTask:
    query = overrides.pop(
        "query",
        MetricQuery(
            bucket="telemetry",
            measurement="sensor",
            field="temperature",
            filters={"host": "*"},
            group_by=["host"],
            range_start="-1h",
        ),
    )
    return AnomalyTask(
        name=overrides.pop("name", "temperature-watch"),
        query=query,
        detector=detector,
        schedule=overrides.pop("schedule", Schedule(every="5m", offset="30s")),
        output=overrides.pop("output", ResultsBucket("anomalies", flag_measurement="anomaly")),
        **overrides,
    )


# -- golden files -----------------------------------------------------------


def test_threshold_script() -> None:
    task = make_task(ThresholdDetector(upper=26.0, lower=14.0, consecutive_points=2))
    assert_golden("threshold.flux", task.to_flux())


def test_threshold_upper_only_script() -> None:
    task = make_task(
        ThresholdDetector(upper=90.0),
        name="cpu-hot",
        query=MetricQuery(bucket="telemetry", measurement="cpu", field="usage"),
        schedule=Schedule(cron="0 * * * *"),
        output=ResultsBucket("anomalies", org="acme", extra_tags={"env": "prod"}),
        description="Hourly CPU ceiling check.",
    )
    assert_golden("threshold_upper_only.flux", task.to_flux())


def test_deadman_script() -> None:
    task = make_task(DeadmanDetector(tolerance="15m"), name="sensor-deadman")
    assert_golden("deadman.flux", task.to_flux())


def test_rate_of_change_script() -> None:
    task = make_task(
        RateOfChangeDetector(max_increase=2.0, max_decrease=0.5, per="1m"),
        name="sensor-rate",
    )
    assert_golden("rate_of_change.flux", task.to_flux())


def test_mad_script() -> None:
    task = make_task(
        MADDetector(k=3.5),
        name="cpu-mad",
        query=MetricQuery(bucket="telemetry", measurement="cpu", field="usage"),
    )
    assert_golden("mad.flux", task.to_flux())


# -- structural properties of every generated script ------------------------


ALL_SERVER_SIDE = [
    ThresholdDetector(upper=26.0, lower=14.0, consecutive_points=2),
    ThresholdDetector(upper=90.0),
    DeadmanDetector(tolerance="15m"),
    RateOfChangeDetector(max_rate=2.0, per="1m"),
]


@pytest.mark.parametrize("detector", ALL_SERVER_SIDE, ids=lambda d: str(d))
def test_every_script_has_the_required_scaffolding(detector) -> None:
    flux = make_task(detector).to_flux()
    assert flux.startswith("// Anomaly task: temperature-watch")
    assert 'import "math"' in flux
    assert flux.count("option task = {") == 1
    assert 'option task = {name: "temperature-watch", every: 5m, offset: 30s}' in flux
    assert '|> to(bucket: "anomalies"' in flux
    assert "_score" in flux and "_threshold" in flux and "_reason" in flux
    assert flux.endswith("\n")


@pytest.mark.parametrize("detector", ALL_SERVER_SIDE, ids=lambda d: str(d))
def test_exactly_one_reference_link_per_script(detector) -> None:
    flux = make_task(detector).to_flux()
    assert flux.count("https://") == 1
    assert FLUX_DOC_LINK in flux


@pytest.mark.parametrize("detector", ALL_SERVER_SIDE, ids=lambda d: str(d))
def test_group_key_carries_the_tag_columns(detector) -> None:
    flux = make_task(detector).to_flux()
    assert '|> group(columns: ["host", "detector", "severity", "_measurement", "_field"])' in flux


def test_extra_tags_are_stamped_and_grouped() -> None:
    task = make_task(
        ThresholdDetector(upper=1.0),
        output=ResultsBucket("anomalies", extra_tags={"env": "prod", "team": "iot"}),
    )
    flux = task.to_flux()
    assert 'env: "prod",' in flux
    assert 'team: "iot",' in flux
    assert '"env", "team"' in flux


def test_severity_mapping_matches_the_python_thresholds() -> None:
    expression = severity_expr("r._score", "r._threshold")
    assert '>= 2.0 then "critical"' in expression
    assert '>= 1.25 then "warning"' in expression
    assert 'else "info"' in expression


# -- unsupported detectors --------------------------------------------------


def test_seasonal_task_refuses_to_compile() -> None:
    task = make_task(SeasonalDetector())
    assert task.execution_mode == "client"
    with pytest.raises(FluxUnsupportedError, match="seasonal cannot be compiled"):
        task.to_flux()


def test_grouped_mad_task_refuses_to_compile() -> None:
    task = make_task(MADDetector(k=3.0))  # default query groups by host
    assert task.execution_mode == "client"
    with pytest.raises(FluxUnsupportedError, match="findRecord"):
        task.to_flux()


# -- helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "1.0"),
        (0.5, "0.5"),
        (-0.5, "-0.5"),
        (100.0, "100.0"),
        (float("inf"), 'float(v: "+Inf")'),
        (float("-inf"), 'float(v: "-Inf")'),
    ],
)
def test_flux_float(value: float, expected: str) -> None:
    assert flux_float(value) == expected


def test_flux_header_without_notes_still_carries_the_link() -> None:
    header = flux_header("t", "threshold(...)")
    assert header.count("https://") == 1
    assert "// Detector:     threshold(...)" in header
