from __future__ import annotations

import pytest

from tsdb_anomaly_task.cli import DEMO_NOW, build_scenario, main


def test_scenario_is_wired_up() -> None:
    client, tasks, now = build_scenario()
    assert now == DEMO_NOW
    assert set(tasks) == {"threshold", "mad", "seasonal", "deadman", "rate"}
    assert len(client.series["telemetry"]) == 3
    assert {t.execution_mode for t in tasks.values()} == {"server", "client"}


def test_scenario_is_deterministic() -> None:
    first, _, _ = build_scenario()
    second, _, _ = build_scenario()
    assert first.series["telemetry"][0].values == second.series["telemetry"][0].values


def test_demo_runs_end_to_end(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "1. What can run where" in out
    assert "preview: temperature-mad" in out
    assert "sweep: k" in out
    assert "option task = {" in out
    assert out.count("https://") == 1


def test_detectors_command(capsys) -> None:
    assert main(["detectors"]) == 0
    out = capsys.readouterr().out
    assert "compiles to Flux" in out
    assert out.count("client") >= 2


@pytest.mark.parametrize("detector", ["threshold", "mad", "seasonal", "deadman", "rate"])
def test_preview_command_for_every_detector(detector: str, capsys) -> None:
    assert main(["preview", detector, "--limit", "3"]) == 0
    assert "preview: temperature-" in capsys.readouterr().out


def test_flux_command_prints_a_script(capsys) -> None:
    assert main(["flux", "deadman"]) == 0
    out = capsys.readouterr().out
    assert 'option task = {name: "temperature-deadman"' in out
    assert "|> last()" in out


def test_flux_command_explains_a_client_side_detector(capsys) -> None:
    assert main(["flux", "seasonal"]) == 1
    out = capsys.readouterr().out
    assert "does not compile to Flux" in out
    assert "AsyncAnomalyRunner" in out


def test_sweep_command(capsys) -> None:
    assert main(["sweep", "mad", "--values", "3", "4", "5"]) == 0
    out = capsys.readouterr().out
    assert "sweep: k" in out
    assert out.count("<-") == 1


def test_sweep_command_with_defaults_and_a_target(capsys) -> None:
    assert main(["sweep", "mad", "--target-rate", "0.001"]) == 0
    assert "sweep: k" in capsys.readouterr().out


def test_run_command_writes_records(capsys) -> None:
    assert main(["run", "threshold"]) == 0
    out = capsys.readouterr().out
    assert "wrote" in out
    assert "anomaly,detector=threshold" in out


def test_unknown_detector_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["preview", "kalman"])
