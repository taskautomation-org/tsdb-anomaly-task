# tsdb-anomaly-task

Declare a metric and a detector; get a scheduled InfluxDB anomaly-detection task — compiled to Flux and deployed to the server, or run in Python where Flux cannot express the statistics.

Anomaly detection on time-series data usually arrives as one of two disappointments. Either it is a `value > 90` check written directly in Flux, which pages at 3 a.m. because a sensor bounced across the line four times in a minute and has no idea that 90 is normal at 14:00 and alarming at 04:00 — or it is a Python notebook full of good statistics that nobody ever scheduled, because turning it into a production task means writing the query layer, the retry loop, the flag schema and the deployment plumbing by hand.

This library covers the gap. You write the metric, the detector and the schedule as one object. Where the detector's maths fits in Flux, it is compiled to a real Flux script and registered as a native InfluxDB task, so it runs on the database with no Python process alive. Where it does not fit — a rolling median, a per-hour seasonal profile — the library says so explicitly, names the exact obstacle, and the same detector runs unchanged in an asyncio runner that queries, evaluates and writes flags back with retrying writes. Before either happens, `preview()` replays the detector over real history so you can see what it *would* have fired on, and a parameter sweep tells you what each candidate threshold would have cost you in alerts.

## Demo

Five detectors run over one week of synthetic sensor data with four deliberately planted faults — a slow drift, a single-sample spike, a five-hour dropout and a short dip. Shaded spans are the ground truth; red rings are what each detector actually flagged. Every marker in this image was produced by running the code (`python examples/make_chart.py`).

![Five stacked time-series panels, one per detector, each showing a week of sensor temperature data as a blue line with grey shaded bands marking four injected faults. Red circles mark the samples each detector flagged: the threshold detector catches only the excursions that leave its band, MAD catches the drift and the spike, the seasonal detector flags deviations from each hour's learned normal, the deadman detector flags the single dropout, and the rate-of-change detector flags the four instantaneous jumps.](docs/detectors.png)

Notice how differently they behave on the same data. The threshold detector never sees the spike at all, because a single sample cannot satisfy `consecutive_points=2`. The deadman check is the only one that notices the sensor stopped talking. The rate-of-change detector ignores the slow drift entirely and fires only on the physically impossible jumps. That is the point: these are complementary questions, not competing algorithms.

And the tuning loop, captured from a real terminal session:

![Terminal session showing two commands. The first, python -m tsdb_anomaly_task preview mad, prints the task summary, reports that the MAD detector must run client-side and explains exactly why, then lists six flagged samples with their values, robust z-scores and severities. The second, python -m tsdb_anomaly_task sweep mad, prints a table of candidate k values from 2.5 to 5.0 with the flag count, flag rate per thousand points and number of series affected for each, marking k=3.0 as the recommended knee of the curve.](docs/demo.svg)

<details>
<summary>The same session as text</summary>

```console
$ python -m tsdb_anomaly_task preview mad --limit 6
task     temperature-mad
metric   telemetry/sensor.temperature
detector mad(k=3.5, window=6h)
schedule every 5m offset 30s
output   anomalies/anomaly
mode     client
reason   a rolling window needs a per-row median, which Flux has no windowed-
         quantile primitive for; grouped series need one median scalar per
         group, and findRecord extracts a single scalar from a single table;
         consecutive-point de-bouncing of a computed score cannot be expressed
         with stateCount over a mapped column

preview: temperature-mad  [mad(k=3.5, window=6h)]
  series 3   points 1560   flags 19   rate 1.22%

  time (UTC)           series                 value     score  severity
  ------------------------------------------------------------------------
  2024-03-04 18:25:00  host=edge-02           21.94      4.01  info
  2024-03-04 18:30:00  host=edge-02           22.27      4.33  info
  2024-03-04 18:35:00  host=edge-02           22.54      4.60  warning
  2024-03-04 18:40:00  host=edge-02           23.05      5.10  warning
  2024-03-04 18:45:00  host=edge-02           24.41      6.43  warning
  2024-03-04 18:50:00  host=edge-02           23.73      5.76  warning
  … and 13 more
  note: 24 point(s) skipped: fewer than min_points=24 baseline samples available

$ python -m tsdb_anomaly_task sweep mad --values 2.5 3 3.5 4 5
sweep: k
       value   flags   points   per 1k  series  max score
  -------------------------------------------------------
         2.5      66     1560     42.3       3       7.12
         3.0      27     1560     17.3       3       7.12 <-
         3.5      19     1560     12.2       3       7.12
         4.0      18     1560     11.5       3       7.12
         5.0      10     1560      6.4       2       7.12
```

</details>

## Install and run from a clone

This is not published to PyPI. Clone it and run it in place.

```bash
git clone https://github.com/taskautomation-org/tsdb-anomaly-task.git
cd tsdb-anomaly-task
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Everything then runs against a deterministic in-memory fake, so you can explore
without a server:

```bash
python -m tsdb_anomaly_task demo             # the full walkthrough
python -m tsdb_anomaly_task detectors        # which detectors run where
python -m tsdb_anomaly_task preview mad      # dry-run, writing nothing
python -m tsdb_anomaly_task flux threshold   # the generated Flux script
python -m tsdb_anomaly_task sweep mad        # tune k against real history
python -m tsdb_anomaly_task run threshold    # one client-side cycle
python examples/quickstart.py                # the library used as a library
```

Requires Python 3.11+. Runtime dependencies are `numpy` and `influxdb-client`;
`matplotlib` is only needed to regenerate the chart.

## Usage

### The shape of a task

A task is four declarations: what to watch, how to judge it, when to run, and where the flags go.

```python
from tsdb_anomaly_task import (
    AnomalyTask, MADDetector, MetricQuery, ResultsBucket, Schedule,
)

task = AnomalyTask(
    name="cpu-anomalies",
    query=MetricQuery(
        bucket="telemetry",
        measurement="cpu",
        field="usage",
        filters={"host": "*"},     # the tag must exist, any value
        group_by=["host"],         # judge each host independently
        range_start="-6h",
    ),
    detector=MADDetector(k=3.5, window="1h", consecutive_points=2),
    schedule=Schedule(every="5m", offset="30s"),
    output=ResultsBucket("anomalies", flag_measurement="anomaly"),
)
```

`group_by` matters more than it looks. Without it, every host's data lands in one series and a busy host drowns out a quiet one's anomalies. With it, each host gets its own baseline, and the host tag is copied onto every flag so alert routing keeps working downstream.

The output bucket must differ from the source bucket — the constructor enforces it. Writing flags into the bucket the task reads makes it consume its own output on the next run.

### Connecting to a real server

```python
from influxdb_client import InfluxDBClient
from tsdb_anomaly_task import InfluxClient

client = InfluxClient(InfluxDBClient(url="http://localhost:8086", token=TOKEN, org="acme"))
```

`InfluxClient` is a thin adapter over the official client — it is the only place in the library that touches `influxdb_client`, which is why the whole test suite runs without a socket. Anywhere a client is accepted, `FakeInfluxClient` works identically.

### Preview before you deploy

This is the feature worth the clone. `preview()` reads real history and runs the detector over it without writing anything, answering the only question that matters at tuning time: *how many alerts would this configuration have produced yesterday?*

```python
preview = task.preview(client)
print(preview.render())

print(len(preview.flags), "flags")
print(f"{preview.flag_rate:.2%} of evaluated points")
for flag in preview.flags[:3]:
    print(flag.time, flag.label, flag.severity, flag.reason)
```

Real output from the quickstart example:

```
preview: cpu-anomalies  [mad(k=3.5, window=6h)]
  series 2   points 1104   flags 5   rate 0.45%

  time (UTC)           series                 value     score  severity
  ------------------------------------------------------------------------
  2024-03-04 16:40:00  host=edge-01           64.12      7.73  critical
  2024-03-04 16:45:00  host=edge-01           64.35      7.80  critical
  2024-03-04 16:50:00  host=edge-01           59.99      6.43  warning
  2024-03-05 01:50:00  host=edge-02           22.94     -4.63  warning
  2024-03-05 01:55:00  host=edge-02           22.35     -4.80  warning
  note: 24 point(s) skipped: fewer than min_points=24 baseline samples available
```

Note the last line. Detectors never raise on thin data; they skip what they cannot judge and record why. `preview.usable` is `False` when nothing at all could be evaluated, and `preview.notes` carries the explanation.

### Tune the parameter instead of guessing it

`sweep_parameter` replays the same history at a range of values. The data is read **once** and replayed for every candidate, so a sweep costs one query however many values you try, and every row is comparable because they all saw exactly the same points.

```python
from tsdb_anomaly_task import sweep_parameter

sweep = sweep_parameter(task, client, parameter="k", values=[2.5, 3.0, 3.5, 4.0, 5.0])
print(sweep.render())
print("pick", sweep.recommended.value)
```

```
sweep: k
       value   flags   points   per 1k  series  max score
  -------------------------------------------------------
         2.5      13     1104     11.8       2       7.80
         3.0       5     1104      4.5       2       7.80 <-
         3.5       5     1104      4.5       2       7.80
         4.0       5     1104      4.5       2       7.80
         5.0       3     1104      2.7       1       7.80
```

Read the knee, not the number. Between `k=3.0` and `k=4.0` nothing changes — those five flags are the real events, and tightening further only starts discarding them. `k=2.5` more than doubles the alert load for no additional signal. Pass `target_rate=` if you would rather express the budget directly ("no more than one flag per thousand points") and let `recommended` pick for you.

Any constructor parameter can be swept, not just `k`: `upper`, `consecutive_points`, `tolerance`, `window`.

### Deploy, or run client-side

```python
print(task.execution_mode)          # "server" or "client"
print(task.flux_support.reason)     # why, when it is "client"
```

For a server-side task, compile and deploy:

```python
print(task.to_flux())               # inspect before you trust it
ref = task.deploy(client)           # create, or replace an existing script
print(ref.id, "created" if ref.created else "updated")
task.undeploy(client)
```

For a client-side task, `deploy()` raises with the reason and points you at the runner:

```python
import asyncio
from tsdb_anomaly_task import AsyncAnomalyRunner, RetryPolicy

runner = AsyncAnomalyRunner(
    client=client,
    tasks=[task, other_task],
    offset="30s",
    retry=RetryPolicy(attempts=5, base_delay=0.5, factor=2.0, max_delay=30.0),
    concurrency=8,
)

asyncio.run(runner.run_forever())      # or run_cycle() for a single pass
```

The runner evaluates tasks concurrently under a semaphore, shares one reference instant across the whole cycle (so a deadman check cannot disagree with itself between tasks), and puts an exponential-backoff retry with full jitter in front of every write. A task that fails is logged and skipped; it does not abort the cycle.

### What gets written

Each flag becomes one line-protocol record in the results bucket:

```
anomaly,detector=mad,env=demo,host=edge-01,severity=critical value=64.12,score=7.73,threshold=3.5,reason="robust z-score +7.73 is above the median 39.5314 by more than k=3.5 scaled MADs (scaled MAD 3.181)" 1709570400000000000
```

`detector`, `severity` and the query's `group_by` tags become tags, so you can route on them. The score, the threshold that fired and a human sentence explaining the decision become fields — because the most expensive part of an alert is the ten minutes someone spends working out why it fired.

## Reference

### Detectors

| Detector | Question it asks | Flux? |
|---|---|---|
| `ThresholdDetector` | Is the value outside a fixed band? | yes, unless `hysteresis > 0` |
| `MADDetector` | Is the value far from its own recent history? | only ungrouped, whole-window, `consecutive_points=1` |
| `SeasonalDetector` | Is the value far from this time-of-day's normal? | no — client-side only |
| `DeadmanDetector` | Has the series stopped reporting? | yes, unless `flag_gaps=True` |
| `RateOfChangeDetector` | Did the value move faster than physics allows? | yes, unless `consecutive_points > 1` |

#### `ThresholdDetector`

| Parameter | Default | Meaning |
|---|---|---|
| `upper` | `None` | Upper limit; `None` disables the upper check |
| `lower` | `None` | Lower limit; `None` disables the lower check |
| `hysteresis` | `0.0` | Width of the reset band, in the metric's units |
| `consecutive_points` | `1` | Adjacent breaches required before flagging |

A naive `value > limit` check is the most common alert in monitoring and the most common source of alert fatigue: a signal hovering around the limit crosses it dozens of times an hour, and each crossing is a page. Hysteresis splits the single limit into a trip point and a *reset* point — once `value > upper` the series stays in alarm until it falls back below `upper - hysteresis` — so an oscillating signal produces one continuous excursion instead of a burst of separate ones. `consecutive_points` handles the other half: a lone offending sample on a real sensor feed is far more often a dropped packet than an event.

Hysteresis is the one part Flux cannot express, because it needs alarm state carried across rows. Set it and the task moves to client-side execution; leave it at zero and the whole detector compiles, with `consecutive_points` implemented via `stateCount`.

#### `MADDetector`

| Parameter | Default | Meaning |
|---|---|---|
| `k` | `3.0` | Robust z-score above which a point is flagged |
| `window` | `None` | Trailing baseline window; `None` uses the whole series |
| `min_points` | `8` | Baseline samples required before a point can be judged |
| `consecutive_points` | `1` | Adjacent breaches required before flagging |

The textbook outlier test is `|x - mean| / stddev > k`, and it has a fatal flaw on machine data: both estimators are computed from the data being tested, and both are unbounded. A sensor reporting `-9999` on a read error drags the mean toward itself and inflates the standard deviation, raising the very threshold that was supposed to catch it. The breakdown point of the mean/stddev pair is 0% — one bad value in a million moves them arbitrarily far.

The median and the MAD have a breakdown point of 50%: up to half the window can be garbage before the estimate moves at all. Multiplying the MAD by 1.4826 (`1/Φ⁻¹(0.75)`) rescales it onto the standard deviation's scale for Gaussian data, so `k=3` keeps its familiar meaning while the estimator stops being hijacked. A finite-sample correction of `n/(n-0.8)` is applied on top, because the 1.4826 constant is asymptotic and a short baseline otherwise biases the scale low and inflates every score computed against it.

The baseline is computed **leave-one-out**: the point under test is excluded from its own median and MAD, so a large excursion cannot dilute the statistic that judges it. Where a window is perfectly flat, the detector falls back to the whole-series scale; where the entire series is constant, it skips rather than inventing an alert, and says so in the notes.

This is the same reasoning that makes the median the better aggregate for noisy sensor rollups, rather than a quirk of anomaly detection.

#### `SeasonalDetector`

| Parameter | Default | Meaning |
|---|---|---|
| `period` | `"hour-of-day"` | Also `"day-of-week"`, `"hour-of-week"` |
| `k` | `3.0` | Robust z-score against the bucket baseline |
| `training` | `"14d"` | Trailing window each baseline is drawn from |
| `min_samples_per_bucket` | `5` | Training samples a bucket needs before it is used |
| `consecutive_points` | `1` | Adjacent breaches required before flagging |

A flat threshold is wrong for anything with a daily rhythm. Traffic that is perfectly normal at 14:00 is an incident at 04:00, and a single limit either misses the night-time anomaly or pages all afternoon. This detector decomposes time into repeating buckets and judges each point only against other points from the same bucket, using the same robust median/MAD baseline as `MADDetector`.

Insufficient history is handled explicitly and never guessed at. A bucket with too few training samples cannot support a baseline, so points landing in it are skipped and the reason appears in `notes`; a result whose every point was skipped comes back with `usable=False`. Silently falling back to a global baseline would report confident nonsense for the first week of a new deployment.

`profile(series)` returns the learned per-bucket baseline (count, median, scaled MAD) and `coverage(series)` returns the fraction of buckets that have enough history — worth checking before you trust the detector in production.

#### `DeadmanDetector`

| Parameter | Default | Meaning |
|---|---|---|
| `tolerance` | `"10m"` | Maximum acceptable silence |
| `flag_gaps` | `False` | Also flag interior gaps, not just trailing silence |

Value-based detectors are blind to the most common IoT failure mode: the sensor does not report a wrong number, it reports nothing. A battery dies, a gateway loses its uplink, a watchdog reboots into a loop — and every threshold and z-score check stays perfectly quiet, because silence never crosses a limit.

Set `tolerance` from the reporting interval, not from how long you are willing to wait: a sensor writing every 60s needs several intervals of slack so one dropped write is not an incident. Enabling `flag_gaps` additionally catches intermittent uplinks that a trailing check misses entirely — by the time the task runs, the sensor is talking again — at the cost of moving the task to client-side execution.

One caveat applies to both modes and is worth internalising: a deadman check can only see series that returned at least one row in the task's window. A sensor silent for longer than the window disappears from the query result entirely and must be caught by comparing against a registry of expected series.

#### `RateOfChangeDetector`

| Parameter | Default | Meaning |
|---|---|---|
| `max_rate` | `None` | Symmetric bound on `abs(rate)` |
| `max_increase` | `max_rate` | Bound on positive rate |
| `max_decrease` | `max_rate` | Bound on the magnitude of negative rate |
| `per` | `"1s"` | Time unit the rates are expressed in |
| `min_interval` | `None` | Ignore pairs closer together than this |
| `consecutive_points` | `1` | Adjacent breaches required before flagging |

Physical quantities have inertia. A room does not warm by 30 °C in one second, a tank does not lose half its level between two reads. When the numbers say otherwise, the sensor is usually the thing that broke — a corrupted I²C read, an ADC glitch, a counter reset, a unit change in new firmware. Those glitches often stay comfortably *inside* the configured min/max limits, so a threshold check never sees them, and they are frequently too brief to move a MAD baseline.

Asymmetric bounds are useful for counters that may legitimately jump up but should never fall, and for tank levels that drain fast and fill slowly. `min_interval` stops a duplicated timestamp from producing a near-infinite rate.

### Configuration objects

| Object | Key parameters |
|---|---|
| `MetricQuery` | `bucket`, `measurement`, `field`, `filters`, `group_by`, `range_start`, `range_stop`, `aggregate_window`, `aggregate_fn` |
| `Schedule` | exactly one of `every` or `cron`, plus `offset` |
| `ResultsBucket` | `bucket`, `flag_measurement`, `org`, `extra_tags` |
| `RetryPolicy` | `attempts`, `base_delay`, `factor`, `max_delay`, `jitter` |

`filters` values are flexible: a plain string is an equality match, `"*"` compiles to `exists r.<tag>` ("the tag must be present, any value"), and a list compiles to an OR-group.

`Schedule.offset` deserves a moment. It delays each run relative to its schedule so late-arriving sensor data has landed before the window is read — without it, an interval task reliably evaluates a window the slowest writers have not finished filling, and the newest bucket looks like a dip on every single run. Getting that number right is a per-measurement question: see [per-measurement offset tuning for late IoT data](https://taskautomation.org/automated-task-scheduling-orchestration/cron-interval-scheduling-logic/per-measurement-offset-tuning-for-late-iot-data/). If you need timezone-aware scheduling rather than a fixed interval, use `cron=` and read [configuring cron expressions for timezone-aware InfluxDB tasks](https://taskautomation.org/automated-task-scheduling-orchestration/cron-interval-scheduling-logic/configuring-cron-expressions-for-timezone-aware-influxdb-tasks/).

### Flag output

| Field | Meaning |
|---|---|
| `time`, `value` | The flagged sample |
| `score` | The detector's own measure (robust z-score, raw value, seconds of silence, rate) |
| `threshold` | The bound that fired |
| `ratio` | `abs(score) / abs(threshold)` — comparable across detectors |
| `severity` | `info` (< 1.25×), `warning` (< 2×), `critical` (≥ 2×) |
| `detector`, `reason` | Which check fired, and a sentence explaining why |
| `tags`, `series_key`, `label` | Series identity, copied from the source |

### CLI

| Command | What it does |
|---|---|
| `demo` | The full walkthrough (the default with no arguments) |
| `detectors` | Lists every detector and its execution mode |
| `preview <detector> [--limit N]` | Dry-run against the synthetic fleet |
| `flux <detector>` | Print the generated Flux, or explain why there isn't any |
| `sweep <detector> [--parameter P] [--values …] [--target-rate R]` | Parameter sweep |
| `run <detector>` | One client-side cycle, writing to the fake client |

## How it works

```
MetricQuery ─┐
Detector ────┼─► AnomalyTask ─┬─► to_flux() ──► deploy()  ──► native InfluxDB task
Schedule ────┤                │
ResultsBucket┘                ├─► preview()  ──► PreviewResult (writes nothing)
                              └─► run_once() ──► AsyncAnomalyRunner ──► write_flags()
                                        │
                                   InfluxProtocol
                                   ├── InfluxClient      (real, over influxdb-client)
                                   └── FakeInfluxClient  (in-memory, used by every test)
```

Four ideas hold it together.

**Detectors are pure functions over a series.** `Detector._judge(series, now)` returns one optional verdict per point, and the base class does the rest: run-length de-bouncing, severity assignment, result assembly. A new detector is one method plus an honest answer to `flux_support()`.

**Flux support is a first-class answer, not an exception.** Every detector reports whether it compiles *as configured* and, if not, exactly which part is the obstacle — a rolling quantile Flux has no primitive for, a per-group scalar `findRecord` cannot produce, alarm state a row-wise pipeline cannot carry. `AnomalyTask.execution_mode` turns that into a one-word answer, and the message you get from `deploy()` tells you what to change or where to run it instead.

**The client surface is five methods.** `read`, `write_flags`, `upsert_task`, `find_task`, `delete_task`, plus async variants of the first two. That is small enough that `FakeInfluxClient` is a complete stand-in rather than a partial mock — which is why the entire test suite runs without a network, and why swapping the fake for the real adapter changes exactly one line of your code.

**Generated Flux is golden-file tested.** The scripts under `tests/golden/` are the contract for what gets deployed; any change to generation shows up as a diff. Each script carries a header naming the task, the detector configuration and any semantic caveat — for instance, the server-side MAD form evaluates over the whole task window while the client-side runner uses a leave-one-out rolling baseline, so the header says so rather than letting you discover it from a disagreeing alert.

### Tests

347 tests, 99% line coverage, no network.

```bash
pytest --cov=tsdb_anomaly_task --cov-report=term-missing
```

Every detector is tested against synthetic series with *known* injected anomalies, and the generator records exactly which samples it corrupted. That lets each detector assert both halves of the quality question — that it catches what was planted, and that a clean control series produces **zero** flags. The rest covers hysteresis state transitions, consecutive-point run logic, seasonal insufficient-history handling, the async runner's backoff schedule and give-up behaviour, `preview()`, the sweep helper, the real client adapter against a stub, and golden-file assertions on every generated script.

## Further reading

Background on the concepts this library implements, on taskautomation.org:

- [Threshold-based alerting with Flux and Python hooks](https://taskautomation.org/automated-task-scheduling-orchestration/anomaly-detection-and-alerting/threshold-based-alerting-with-flux-and-python-hooks/) — the server-side/client-side split this library formalises.
- [Choosing mean vs median for noisy sensor rollups](https://taskautomation.org/downsampling-aggregation-pipeline-design/threshold-tuning-for-aggregation/choosing-mean-vs-median-for-noisy-sensor-rollups/) — why `MADDetector` exists at all.
- [Deadman checks for detecting silent IoT sensors](https://taskautomation.org/automated-task-scheduling-orchestration/anomaly-detection-and-alerting/deadman-checks-for-detecting-silent-iot-sensors/) — alerting on absence, and why it needs a registry of expected series.
- [Using Python asyncio with the InfluxDB client v2 for batch tasks](https://taskautomation.org/automated-task-scheduling-orchestration/python-client-orchestration-patterns/using-python-asyncio-with-influxdb-client-v2-for-batch-tasks/) — the concurrency model behind `AsyncAnomalyRunner`.
- [Exponential backoff and retry for InfluxDB client writes](https://taskautomation.org/automated-task-scheduling-orchestration/python-client-orchestration-patterns/exponential-backoff-and-retry-for-influxdb-client-writes/) — why the retry policy uses full jitter.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports are most useful with a
`make_series(...)` call that reproduces the behaviour.

## License

MIT — see [LICENSE](LICENSE).
