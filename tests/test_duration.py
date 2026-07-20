from __future__ import annotations

from datetime import timedelta

import pytest

from tsdb_anomaly_task.duration import format_duration, parse_duration


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30s", 30),
        ("5m", 300),
        ("1h", 3600),
        ("1h30m", 5400),
        ("2d", 172800),
        ("1w", 604800),
        ("250ms", 0.25),
        ("1h30m15s", 5415),
    ],
)
def test_parse_duration(text: str, seconds: float) -> None:
    assert parse_duration(text).total_seconds() == pytest.approx(seconds)


def test_parse_duration_accepts_timedelta() -> None:
    delta = timedelta(minutes=7)
    assert parse_duration(delta) is delta


def test_parse_duration_negative() -> None:
    assert parse_duration("-1h").total_seconds() == -3600


@pytest.mark.parametrize("bad", ["", "   ", "5", "5x", "m5", "1h30", "abc"])
def test_parse_duration_rejects_junk(bad: str) -> None:
    with pytest.raises(ValueError, match="duration"):
        parse_duration(bad)


def test_parse_duration_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        parse_duration(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(minutes=90), "1h30m"),
        ("300s", "5m"),
        (timedelta(0), "0s"),
        (timedelta(days=8), "1w1d"),
        (timedelta(seconds=-60), "-1m"),
        (timedelta(milliseconds=250), "250ms"),
    ],
)
def test_format_duration(value, expected: str) -> None:
    assert format_duration(value) == expected


def test_format_round_trips() -> None:
    for text in ("30s", "5m", "1h30m", "2d", "1w"):
        assert format_duration(parse_duration(text)) == format_duration(text)
