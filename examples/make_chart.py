#!/usr/bin/env python
"""Render ``docs/detectors.png`` by actually running every detector.

One week of synthetic sensor data with four known faults is generated, each
detector is run over it, and the flagged samples are marked.  Nothing in the
figure is drawn by hand — the markers are the detectors' real output.

Run from a clone::

    python examples/make_chart.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tsdb_anomaly_task import (
    DeadmanDetector,
    MADDetector,
    RateOfChangeDetector,
    SeasonalDetector,
    ThresholdDetector,
)
from tsdb_anomaly_task.synthetic import DEFAULT_START, Anomaly, make_series

# -- style ------------------------------------------------------------------
# A solid light surface, so the PNG reads identically in GitHub's light and
# dark themes.  Palette validated for CVD separation and contrast.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SIGNAL = "#2a78d6"
FLAG = "#e34948"
GUIDE = "#b8b7b1"
GROUND_TRUTH = "#e8e8e3"

INTERVAL_MINUTES = 10
DAYS = 7
COUNT = DAYS * 24 * 60 // INTERVAL_MINUTES

ANOMALIES = [
    Anomaly(index=int(3.4 * 144), kind="drift", magnitude=11.0, length=30),
    Anomaly(index=int(4.3 * 144), kind="spike", magnitude=16.0),
    Anomaly(index=int(5.1 * 144), kind="gap", length=30),
    Anomaly(index=int(6.2 * 144), kind="dip", magnitude=14.0, length=4),
]

DETECTORS = [
    (
        "ThresholdDetector(upper=23.5, lower=16.5, consecutive_points=2)",
        "static band, de-bounced",
        ThresholdDetector(upper=23.5, lower=16.5, consecutive_points=2),
        (16.5, 23.5),
    ),
    (
        "MADDetector(k=4, window='12h')",
        "robust z-score against recent history",
        MADDetector(k=4.0, window="12h", min_points=36),
        None,
    ),
    (
        "SeasonalDetector(period='hour-of-day', k=5.5)",
        "deviation from this hour's learned normal",
        SeasonalDetector(period="hour-of-day", k=5.5, training="7d", min_samples_per_bucket=4),
        None,
    ),
    (
        "DeadmanDetector(tolerance='1h', flag_gaps=True)",
        "the sensor stopped reporting",
        DeadmanDetector(tolerance="1h", flag_gaps=True),
        None,
    ),
    (
        "RateOfChangeDetector(max_rate=0.5, per='1m')",
        "moved faster than physics allows",
        RateOfChangeDetector(max_rate=0.5, per="1m"),
        None,
    ),
]


def main() -> int:
    generated = make_series(
        measurement="sensor",
        field_name="temperature",
        tags={"host": "edge-01"},
        count=COUNT,
        interval=f"{INTERVAL_MINUTES}m",
        base=20.0,
        noise=0.45,
        daily_amplitude=2.2,
        anomalies=ANOMALIES,
        seed=2024,
    )
    series = generated.series
    now = series.end + timedelta(minutes=INTERVAL_MINUTES)

    fig, axes = plt.subplots(
        len(DETECTORS),
        1,
        figsize=(11.0, 12.6),
        sharex=True,
        facecolor=SURFACE,
    )
    fig.suptitle(
        "Five detectors, one week of sensor data, four planted faults",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.958,
        "Shaded spans are the injected faults. Red rings are what each detector actually flagged.",
        ha="left",
        fontsize=10.5,
        color=INK_MUTED,
    )

    times = series.times
    for axis, (title, subtitle, detector, band) in zip(axes, DETECTORS, strict=True):
        result = detector.evaluate(series, now=now)
        flagged = {flag.time: flag.value for flag in result.flags}

        axis.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(GUIDE)
        axis.tick_params(colors=INK_MUTED, labelsize=9, length=3)
        axis.grid(axis="y", color=GUIDE, alpha=0.35, linewidth=0.6)
        axis.set_axisbelow(True)

        # Ground truth: shade the spans that were deliberately corrupted.
        for anomaly in ANOMALIES:
            start = DEFAULT_START + anomaly.index * timedelta(minutes=INTERVAL_MINUTES)
            end = start + max(anomaly.length, 2) * timedelta(minutes=INTERVAL_MINUTES)
            axis.axvspan(start, end, color=GROUND_TRUTH, zorder=0)

        if band is not None:
            for limit in band:
                axis.axhline(limit, color=GUIDE, linestyle=(0, (5, 4)), linewidth=1.1, zorder=1)

        axis.plot(times, series.values, color=SIGNAL, linewidth=1.1, zorder=2)

        if flagged:
            axis.scatter(
                list(flagged),
                list(flagged.values()),
                s=46,
                facecolors="none",
                edgecolors=FLAG,
                linewidths=1.7,
                zorder=3,
            )

        axis.set_ylabel("°C", color=INK_MUTED, fontsize=9)
        axis.text(
            0.0,
            1.13,
            title,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=INK,
            family="DejaVu Sans Mono",
        )
        axis.text(
            0.0,
            1.005,
            f"{subtitle}  ·  {len(result.flags)} flagged of {result.points_evaluated} evaluated",
            transform=axis.transAxes,
            fontsize=9.5,
            color=INK_MUTED,
        )

    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    axes[-1].set_xlabel("time (UTC)", color=INK_MUTED, fontsize=9)

    fig.subplots_adjust(left=0.065, right=0.975, top=0.925, bottom=0.05, hspace=0.62)

    out = Path(__file__).resolve().parent.parent / "docs" / "detectors.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"wrote {out.relative_to(out.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
