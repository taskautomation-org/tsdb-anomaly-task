# Contributing

Thanks for taking a look. Small, focused pull requests are the easiest to merge.

## Getting set up

```bash
git clone https://github.com/taskautomation-org/tsdb-anomaly-task.git
cd tsdb-anomaly-task
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Before you open a PR

```bash
ruff check .
ruff format --check .
pytest --cov=tsdb_anomaly_task
```

All three must be clean. Coverage should not drop.

## House rules

- **No network in tests.** Use `FakeInfluxClient` and the synthetic series
  generator in `tsdb_anomaly_task/synthetic.py`. If you need a new fixture
  shape, add it to the generator rather than hard-coding numbers.
- **New detectors** subclass `Detector` and implement `_judge`. They must also
  answer `flux_support()` honestly: if the statistic cannot be expressed in
  Flux, say exactly which part is the obstacle so the task can fall back to the
  client-side runner with a useful message.
- **Detector tests must prove both halves**: that injected anomalies are caught,
  and that a clean control series produces zero flags.
- **Changing generated Flux** changes the golden files in `tests/golden/`.
  Review the diff, then regenerate with
  `UPDATE_GOLDEN=1 pytest tests/test_flux_generation.py`.
- **Changing terminal output** means regenerating the demo assets:
  `python examples/make_chart.py && python examples/make_demo_svg.py`.

## Reporting a bug

A failing case is worth ten paragraphs. Include the detector configuration and
a short synthetic series (`make_series(...)` call) that reproduces it.
