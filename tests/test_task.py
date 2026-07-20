from __future__ import annotations

from datetime import timedelta

import pytest

from tsdb_anomaly_task import (
    AnomalyTask,
    DeadmanDetector,
    FakeInfluxClient,
    MADDetector,
    MetricQuery,
    RecordingClient,
    ResultsBucket,
    Schedule,
    SeasonalDetector,
    ThresholdDetector,
)


def build(detector, **kwargs) -> AnomalyTask:
    return AnomalyTask(
        name=kwargs.pop("name", "t"),
        query=kwargs.pop(
            "query",
            MetricQuery(
                bucket="telemetry",
                measurement="sensor",
                field="temperature",
                filters={"host": "*"},
                group_by=["host"],
                range_start="-48h",
            ),
        ),
        detector=detector,
        schedule=kwargs.pop("schedule", Schedule(every="5m")),
        output=kwargs.pop("output", ResultsBucket("anomalies")),
        **kwargs,
    )


# -- validation -------------------------------------------------------------


def test_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build(ThresholdDetector(upper=1.0), name="  ")


def test_rejects_writing_flags_into_the_source_bucket() -> None:
    with pytest.raises(ValueError, match="consume its own output"):
        build(ThresholdDetector(upper=1.0), output=ResultsBucket("telemetry"))


# -- mode reporting ---------------------------------------------------------


def test_execution_mode_server() -> None:
    task = build(ThresholdDetector(upper=1.0))
    assert task.execution_mode == "server"
    assert task.flux_support


def test_execution_mode_client_explains_itself() -> None:
    task = build(SeasonalDetector())
    assert task.execution_mode == "client"
    assert "client-side" in task.flux_support.reason


def test_summary_covers_both_modes() -> None:
    server = build(ThresholdDetector(upper=1.0), schedule=Schedule(every="5m", offset="30s"))
    assert "mode     server" in server.summary()
    assert "every 5m offset 30s" in server.summary()

    client = build(SeasonalDetector(), schedule=Schedule(cron="0 * * * *"))
    summary = client.summary()
    assert "mode     client" in summary
    assert "reason" in summary
    assert "cron 0 * * * *" in summary


# -- preview ----------------------------------------------------------------


def test_preview_finds_the_injected_anomalies(client, now, spiky) -> None:
    task = build(ThresholdDetector(upper=25.0, lower=17.0))
    preview = task.preview(client, now=now)
    assert len(preview.series) == 1
    assert preview.points_evaluated == len(spiky.series)
    assert 0 < preview.flag_rate < 0.05
    assert spiky.caught([f.time for f in preview.flags]) == len(spiky.anomalies)
    assert preview.usable


def test_preview_writes_nothing(client, now) -> None:
    build(ThresholdDetector(upper=25.0)).preview(client, now=now)
    assert client.written == []
    assert client.writes == []


def test_preview_flags_are_time_ordered(client, now) -> None:
    flags = build(ThresholdDetector(upper=25.0, lower=17.0)).preview(client, now=now).flags
    assert list(flags) == sorted(flags, key=lambda f: f.time)


def test_preview_render_includes_the_summary_line(client, now) -> None:
    text = build(ThresholdDetector(upper=25.0, lower=17.0)).preview(client, now=now).render(limit=2)
    assert text.startswith("preview: t  [threshold(")
    assert "series 1" in text
    assert "and " in text  # the truncation line
    assert "severity" in text


def test_preview_render_when_nothing_fired(client, now) -> None:
    text = build(ThresholdDetector(upper=500.0)).preview(client, now=now).render()
    assert "no anomalies in the previewed window" in text


def test_preview_render_shows_notes(client, now) -> None:
    text = build(MADDetector(k=3.0, window="6h", min_points=24)).preview(client, now=now).render()
    assert "note:" in text


def test_preview_of_an_empty_bucket(now) -> None:
    empty = FakeInfluxClient()
    preview = build(ThresholdDetector(upper=1.0)).preview(empty, now=now)
    assert len(preview) == 0
    assert preview.points_evaluated == 0
    assert preview.flag_rate == 0.0
    assert not preview.usable


def test_preview_notes_are_deduplicated(now, spiky) -> None:
    fake = FakeInfluxClient()
    fake.add_synthetic("telemetry", spiky)
    fake.add_series("telemetry", spiky.series)  # same shape twice
    preview = build(MADDetector(k=3.0, window="6h", min_points=24)).preview(fake, now=now)
    assert len(preview.series) == 2
    assert len(preview.notes) == len(set(preview.notes))


# -- run_once ---------------------------------------------------------------


def test_run_once_writes_line_protocol(client, now) -> None:
    report = build(ThresholdDetector(upper=25.0, lower=17.0)).run_once(client, now=now)
    assert report.written == len(report.flags) > 0
    assert client.writes == [("anomalies", "anomaly", report.written)]
    record = client.written[0]
    assert record.startswith("anomaly,detector=threshold,host=edge-01,severity=")


def test_run_once_dry_run_writes_nothing(client, now) -> None:
    report = build(ThresholdDetector(upper=25.0, lower=17.0)).run_once(
        client, now=now, dry_run=True
    )
    assert len(report.flags) > 0
    assert report.written == 0
    assert client.written == []


def test_run_once_with_no_flags_skips_the_write(client, now) -> None:
    report = build(ThresholdDetector(upper=500.0)).run_once(client, now=now)
    assert report.written == 0
    assert client.writes == []


def test_run_once_applies_extra_tags(client, now) -> None:
    task = build(
        ThresholdDetector(upper=25.0),
        output=ResultsBucket("anomalies", extra_tags={"env": "prod"}),
    )
    task.run_once(client, now=now)
    assert all("env=prod" in record for record in client.written)


def test_run_once_deadman_uses_the_reference_instant(client, spiky) -> None:
    task = build(DeadmanDetector(tolerance="30m"))
    quiet = spiky.series.end + timedelta(hours=5)
    assert len(task.run_once(client, now=quiet).flags) == 1
    assert len(task.run_once(client, now=spiky.series.end).flags) == 0


# -- deploy -----------------------------------------------------------------


def test_deploy_creates_then_updates(client) -> None:
    task = build(ThresholdDetector(upper=25.0), name="temperature-threshold")
    first = task.deploy(client)
    assert first.created and not first.updated
    assert task.deployed(client) is not None

    second = task.deploy(client)
    assert second.updated
    assert second.id == first.id
    assert second.flux == task.to_flux()


def test_deploy_refuses_client_side_detectors(client) -> None:
    from tsdb_anomaly_task import FluxUnsupportedError

    with pytest.raises(FluxUnsupportedError, match="seasonal cannot be compiled"):
        build(SeasonalDetector()).deploy(client)
    assert client.tasks == {}


def test_undeploy(client) -> None:
    task = build(ThresholdDetector(upper=25.0))
    task.deploy(client)
    assert task.undeploy(client) is True
    assert task.undeploy(client) is False
    assert task.deployed(client) is None


def test_recording_client_logs_the_calls(client, now) -> None:
    recorder = RecordingClient(inner=client)
    task = build(ThresholdDetector(upper=25.0, lower=17.0))
    task.deploy(recorder)
    task.run_once(recorder, now=now)
    task.deployed(recorder)
    task.undeploy(recorder)
    names = [call[0] for call in recorder.calls]
    assert names == ["upsert_task", "read", "write_flags", "find_task", "delete_task"]
