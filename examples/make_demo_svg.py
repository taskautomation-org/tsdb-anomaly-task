#!/usr/bin/env python
"""Render ``docs/demo.svg`` from a real captured terminal session.

The commands below are executed for real; their stdout is what ends up in the
SVG.  Re-run this after changing any output formatting so the README's terminal
render never drifts from the tool.

Run from a clone::

    python examples/make_demo_svg.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent

COMMANDS = [
    ["preview", "mad", "--limit", "6"],
    ["sweep", "mad", "--values", "2.5", "3", "3.5", "4", "5"],
]

# Terminal chrome.  A solid dark background keeps the SVG legible in GitHub's
# light and dark themes alike, without relying on prefers-color-scheme.
BG = "#161b22"
CHROME = "#21262d"
FG = "#d6d9de"
PROMPT = "#7ee787"
COMMAND = "#e6edf3"
DIM = "#8b949e"
ACCENT = "#79c0ff"
WARN = "#f0a35e"
CRIT = "#ff7b72"

CHAR_W = 8.05
LINE_H = 19.0
PAD_X = 22.0
PAD_TOP = 52.0
PAD_BOTTOM = 22.0
FONT = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def capture() -> list[tuple[str, str]]:
    """Run each command and return ``(kind, text)`` lines."""
    lines: list[tuple[str, str]] = []
    for index, argv in enumerate(COMMANDS):
        if index:
            lines.append(("blank", ""))
        lines.append(("prompt", "python -m tsdb_anomaly_task " + " ".join(argv)))
        completed = subprocess.run(
            [sys.executable, "-m", "tsdb_anomaly_task", *argv],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        for raw in completed.stdout.rstrip("\n").split("\n"):
            lines.append(("out", raw))
    return lines


def colour_for(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return FG
    if text.startswith("         ") and not stripped[0].isdigit():
        return DIM  # wrapped continuation of the "reason" block
    if stripped.startswith(("task ", "metric ", "detector ", "schedule ", "output ", "mode ")):
        return DIM
    if stripped.startswith(("preview:", "sweep:")):
        return ACCENT
    if stripped.startswith(("note:", "reason ")):
        return DIM
    if stripped.startswith(("-", "time (UTC)", "value")):
        return DIM
    if stripped.endswith("critical"):
        return CRIT
    if stripped.endswith("warning"):
        return WARN
    if stripped.endswith("<-"):
        return PROMPT
    return FG


def render(lines: list[tuple[str, str]]) -> str:
    width = max(len(text) for _, text in lines) + 4
    px_w = round(PAD_X * 2 + width * CHAR_W)
    px_h = round(PAD_TOP + len(lines) * LINE_H + PAD_BOTTOM)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{px_w}" height="{px_h}" '
        f'viewBox="0 0 {px_w} {px_h}" font-family="{FONT}" font-size="13">',
        f'<rect width="{px_w}" height="{px_h}" rx="10" fill="{BG}"/>',
        f'<rect width="{px_w}" height="34" rx="10" fill="{CHROME}"/>',
        f'<rect y="24" width="{px_w}" height="10" fill="{CHROME}"/>',
        '<circle cx="21" cy="17" r="6" fill="#ff5f57"/>',
        '<circle cx="41" cy="17" r="6" fill="#febc2e"/>',
        '<circle cx="61" cy="17" r="6" fill="#28c840"/>',
        f'<text x="{px_w / 2}" y="21" fill="{DIM}" font-size="12" '
        f'text-anchor="middle">tsdb-anomaly-task</text>',
    ]

    y = PAD_TOP
    for kind, text in lines:
        if kind == "blank":
            y += LINE_H
            continue
        if kind == "prompt":
            out.append(
                f'<text x="{PAD_X}" y="{y}" fill="{PROMPT}" xml:space="preserve">$ '
                f'<tspan fill="{COMMAND}">{escape(text)}</tspan></text>'
            )
        else:
            out.append(
                f'<text x="{PAD_X}" y="{y}" fill="{colour_for(text)}" '
                f'xml:space="preserve">{escape(text)}</text>'
            )
        y += LINE_H

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    lines = capture()
    target = ROOT / "docs" / "demo.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(lines), encoding="utf-8")
    print(f"wrote docs/demo.svg ({len(lines)} lines)")

    transcript = ROOT / "docs" / "demo.txt"
    transcript.write_text(
        "\n".join(("$ " + text if kind == "prompt" else text) for kind, text in lines) + "\n",
        encoding="utf-8",
    )
    print("wrote docs/demo.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
