#!/usr/bin/env python3
"""Render a GitHub-style LexiPilot study intensity heatmap."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from console_theme import should_enable_color, strip_ansi


LEVEL_CHARS = ["  ", "..", "::", "++", "##"]
LEVEL_COLORS = [
    "48;5;236;37",
    "48;5;22;97",
    "48;5;28;97",
    "48;5;34;30",
    "48;5;40;30",
]


@dataclass(frozen=True)
class DayActivity:
    day: date
    words: int
    correct: int
    missed: int

    @property
    def intensity(self) -> int:
        if self.words <= 0:
            return 0
        if self.words < 5:
            return 1
        if self.words < 12:
            return 2
        if self.words < 25:
            return 3
        return 4


def color_enabled(no_color: bool) -> bool:
    return False if no_color else should_enable_color()


def style(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def generate_random_activity(days: int, seed: int, today: date | None = None) -> list[DayActivity]:
    rng = random.Random(seed)
    end = today or date.today()
    start = end - timedelta(days=max(1, days) - 1)
    data: list[DayActivity] = []
    streak_bias = 0.0
    for offset in range(days):
        day = start + timedelta(days=offset)
        weekday = day.weekday()
        weekend_penalty = 0.18 if weekday >= 5 else 0.0
        study_probability = max(0.18, min(0.88, 0.58 + streak_bias - weekend_penalty))
        if rng.random() < study_probability:
            words = max(1, int(rng.expovariate(1 / 11)) + rng.randint(0, 8))
            if rng.random() < 0.08:
                words += rng.randint(18, 35)
            streak_bias = min(0.22, streak_bias + 0.025)
        else:
            words = 0
            streak_bias = max(-0.18, streak_bias - 0.06)
        correct = int(words * rng.uniform(0.62, 0.93)) if words else 0
        missed = max(0, words - correct)
        data.append(DayActivity(day, words, correct, missed))
    return data


def activity_from_daily_stats(
    daily_stats: dict[str, Any],
    days: int = 28,
    *,
    today: date | None = None,
) -> list[DayActivity]:
    end = today or date.today()
    count = max(1, min(365, int(days)))
    start = end - timedelta(days=count - 1)
    data: list[DayActivity] = []
    for offset in range(count):
        day = start + timedelta(days=offset)
        raw = daily_stats.get(day.isoformat(), {})
        stats = raw if isinstance(raw, dict) else {}
        words = max(0, int(stats.get("studied", 0) or 0))
        missed = max(0, int(stats.get("missed", 0) or 0))
        remembered = max(0, int(stats.get("remembered", max(0, words - missed)) or 0))
        data.append(DayActivity(day, words, remembered, missed))
    return data


def month_labels(weeks: list[list[DayActivity | None]]) -> str:
    labels = ["   "]
    previous = None
    for week in weeks:
        first = next((item for item in week if item is not None), None)
        if first is None:
            labels.append("  ")
            continue
        month = first.day.strftime("%b")
        if month != previous and first.day.day <= 7:
            labels.append(month[:2])
            previous = month
        else:
            labels.append("  ")
    return " ".join(labels).rstrip()


def week_grid(data: list[DayActivity]) -> list[list[DayActivity | None]]:
    if not data:
        return []
    first = data[0].day
    leading = (first.weekday() + 1) % 7
    cells: list[DayActivity | None] = [None] * leading + data
    while len(cells) % 7:
        cells.append(None)
    return [cells[index : index + 7] for index in range(0, len(cells), 7)]


def render_heatmap(
    data: list[DayActivity],
    *,
    no_color: bool = False,
    source: str = "deterministic random demo data",
) -> str:
    enabled = color_enabled(no_color)
    weeks = week_grid(data)
    heatmap_lines = [month_labels(weeks)]
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    for row in range(7):
        cells = []
        for week in weeks:
            activity = week[row]
            if activity is None:
                cells.append("  ")
            else:
                level = activity.intensity
                cells.append(style(LEVEL_CHARS[level], LEVEL_COLORS[level], enabled))
        label = day_names[row] if row in {1, 3, 5} else "   "
        heatmap_lines.append(f"{label} " + " ".join(cells).rstrip())

    total_words = sum(item.words for item in data)
    active_days = sum(1 for item in data if item.words > 0)
    missed = sum(item.missed for item in data)
    best = max(data, key=lambda item: item.words) if data else None

    def compact_progress(value: int, total: int, width: int = 10) -> str:
        ratio = (value / total) if total > 0 else 0.0
        filled = max(0, min(width, int(round(ratio * width))))
        return f"[{'█' * filled}{'░' * (width - filled)}] {ratio * 100:.1f}%"

    heatmap_lines.append("")
    heatmap_lines.append(
        f"ActiveDays: {active_days}|{len(data)}  "
        f"{compact_progress(active_days, len(data))}"
    )
    heatmap_lines.append(
        f"Missed: {missed}|{total_words}  "
        f"{compact_progress(missed, total_words)}"
    )
    if best:
        heatmap_lines.append(f"Peak day: {best.day.isoformat()} ({best.words} words)")
    heatmap_lines.append(source if source.startswith("profile [") else f"Source: {source}")

    legend_lines = [
        "Level    Mark  Words",
        "None     [  ]  0",
        f"Light    [{style('..', LEVEL_COLORS[1], enabled)}]  1-4",
        f"Medium   [{style('::', LEVEL_COLORS[2], enabled)}]  5-11",
        f"Strong   [{style('++', LEVEL_COLORS[3], enabled)}]  12-24",
        f"Intense  [{style('##', LEVEL_COLORS[4], enabled)}]  25+",
    ]
    return render_activity_panel(heatmap_lines, legend_lines)


def _visible_width(text: str) -> int:
    return len(strip_ansi(text))


def _visible_ljust(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_width(text))


def _titled_border(label: str, width: int) -> str:
    title = f"─ {label} "
    return title + "─" * max(0, width - len(title))


def render_activity_panel(left_lines: list[str], right_lines: list[str]) -> str:
    left_width = max(34, *(_visible_width(line) for line in left_lines))
    right_width = max(24, *(_visible_width(line) for line in right_lines))
    height = max(len(left_lines), len(right_lines))
    lines = [
        "╭"
        + _titled_border("LexiPilot Study Activity", left_width + 2)
        + "┬"
        + _titled_border("Intensity Legend", right_width + 2)
        + "╮"
    ]
    for index in range(height):
        left = left_lines[index] if index < len(left_lines) else ""
        right = right_lines[index] if index < len(right_lines) else ""
        lines.append(
            f"│ {_visible_ljust(left, left_width)} │ "
            f"{_visible_ljust(right, right_width)} │"
        )
    lines.append("╰" + "─" * (left_width + 2) + "┴" + "─" * (right_width + 2) + "╯")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show a LexiPilot study-intensity heatmap with demo data.")
    parser.add_argument("--days", type=int, default=365, help="Number of days to render")
    parser.add_argument("--seed", type=int, default=20260805, help="Random seed for deterministic demo data")
    parser.add_argument("--no-color", action="store_true", help="Disable terminal color")
    args = parser.parse_args()
    data = generate_random_activity(max(1, args.days), args.seed)
    print(render_heatmap(data, no_color=args.no_color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
