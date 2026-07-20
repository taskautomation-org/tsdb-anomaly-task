"""Parsing and formatting of InfluxDB-style duration literals.

InfluxDB and Flux both speak durations as compact literals such as ``5m``,
``1h30m`` or ``2w``.  The whole library accepts those strings (and
:class:`datetime.timedelta` objects) interchangeably, so this module is the
single place where the conversion lives.
"""

from __future__ import annotations

import re
from datetime import timedelta

__all__ = ["Duration", "format_duration", "parse_duration"]

#: A duration as accepted by the public API: a Flux literal or a timedelta.
Duration = "str | timedelta"

_UNITS: dict[str, float] = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
}

# Longest units first so that "ms" wins over "m".
_TERM = re.compile(r"(\d+(?:\.\d+)?)(ns|us|ms|s|m|h|d|w)")
_FULL = re.compile(r"^(?:\d+(?:\.\d+)?(?:ns|us|ms|s|m|h|d|w))+$")


def parse_duration(value: str | timedelta) -> timedelta:
    """Convert a Flux duration literal into a :class:`~datetime.timedelta`.

    Compound literals are supported, so ``"1h30m"`` parses as 90 minutes.

    >>> parse_duration("5m")
    datetime.timedelta(seconds=300)
    >>> parse_duration("1h30m")
    datetime.timedelta(seconds=5400)

    Raises:
        ValueError: if the literal is empty, malformed, or has no unit.
    """
    if isinstance(value, timedelta):
        return value
    if not isinstance(value, str):  # pragma: no cover - defensive
        raise TypeError(f"duration must be a str or timedelta, got {type(value)!r}")

    text = value.strip()
    if not text:
        raise ValueError("duration must not be empty")
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    if not _FULL.match(text):
        raise ValueError(
            f"invalid duration {value!r}: expected a Flux literal such as "
            f"'30s', '5m', '1h30m' or '2w'"
        )

    seconds = sum(float(n) * _UNITS[u] for n, u in _TERM.findall(text))
    return timedelta(seconds=-seconds if negative else seconds)


def format_duration(value: str | timedelta) -> str:
    """Render a duration as the shortest exact Flux literal.

    >>> format_duration(timedelta(minutes=90))
    '1h30m'
    >>> format_duration("300s")
    '5m'
    """
    delta = parse_duration(value)
    # Work in whole nanoseconds: seconds-as-float would render 250ms as
    # "249ms999us999ns".
    total = round(delta.total_seconds() * 1_000_000_000)
    if total == 0:
        return "0s"
    sign = "-" if total < 0 else ""
    total = abs(total)

    parts: list[str] = []
    for unit, size in (
        ("w", 604_800_000_000_000),
        ("d", 86_400_000_000_000),
        ("h", 3_600_000_000_000),
        ("m", 60_000_000_000),
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("us", 1_000),
        ("ns", 1),
    ):
        if total >= size:
            count, total = divmod(total, size)
            parts.append(f"{count}{unit}")
        if total == 0:
            break
    return sign + "".join(parts)
