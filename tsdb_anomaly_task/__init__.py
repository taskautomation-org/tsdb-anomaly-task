"""Declare a metric and a detector; get a scheduled InfluxDB anomaly task.

.. code-block:: python

    from tsdb_anomaly_task import (
        AnomalyTask, MADDetector, MetricQuery, ResultsBucket, Schedule,
    )

    task = AnomalyTask(
        name="cpu-anomalies",
        query=MetricQuery(bucket="telemetry", measurement="cpu", field="usage",
                          filters={"host": "*"}, group_by=["host"]),
        detector=MADDetector(k=3.5, window="1h", consecutive_points=2),
        schedule=Schedule(every="5m", offset="30s"),
        output=ResultsBucket("anomalies", flag_measurement="anomaly"),
    )

    task.preview(client)    # dry-run against real history
    print(task.to_flux())   # inspect the generated script
    task.deploy(client)     # register it as a native InfluxDB task
"""

from __future__ import annotations

from .client import (
    AsyncInfluxProtocol,
    FakeInfluxClient,
    InfluxClient,
    InfluxProtocol,
    RecordingClient,
    TaskRef,
    WriteError,
)
from .detectors import (
    NORMAL_CONSISTENCY,
    BucketBaseline,
    DeadmanDetector,
    Detector,
    FluxContext,
    FluxSupport,
    FluxUnsupportedError,
    MADDetector,
    RateOfChangeDetector,
    SeasonalDetector,
    SeasonalPeriod,
    ThresholdDetector,
    robust_scale,
    small_sample_correction,
)
from .duration import format_duration, parse_duration
from .models import (
    DetectionResult,
    Flag,
    MetricQuery,
    Point,
    ResultsBucket,
    Schedule,
    Series,
    Severity,
)
from .runner import AsyncAnomalyRunner, RetryPolicy, RunnerStats
from .synthetic import Anomaly, SyntheticSeries, make_series
from .task import AnomalyTask, PreviewResult, RunReport
from .tuning import SweepResult, SweepRow, sweep_parameter

__version__ = "1.0.0"

__all__ = [
    "NORMAL_CONSISTENCY",
    "Anomaly",
    "AnomalyTask",
    "AsyncAnomalyRunner",
    "AsyncInfluxProtocol",
    "BucketBaseline",
    "DeadmanDetector",
    "DetectionResult",
    "Detector",
    "FakeInfluxClient",
    "Flag",
    "FluxContext",
    "FluxSupport",
    "FluxUnsupportedError",
    "InfluxClient",
    "InfluxProtocol",
    "MADDetector",
    "MetricQuery",
    "Point",
    "PreviewResult",
    "RateOfChangeDetector",
    "RecordingClient",
    "ResultsBucket",
    "RetryPolicy",
    "RunReport",
    "RunnerStats",
    "Schedule",
    "SeasonalDetector",
    "SeasonalPeriod",
    "Series",
    "Severity",
    "SweepResult",
    "SweepRow",
    "SyntheticSeries",
    "TaskRef",
    "ThresholdDetector",
    "WriteError",
    "__version__",
    "format_duration",
    "make_series",
    "parse_duration",
    "robust_scale",
    "small_sample_correction",
    "sweep_parameter",
]
