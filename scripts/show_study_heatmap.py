#!/usr/bin/env python3
"""Render a GitHub-style LexiPilot study intensity heatmap."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from console_theme import should_enable_color


LEVEL_CHARS = ["  ", "..", "::", "++", "##"]
LEVEL_COLORS = ["2", "32", "92", "1;32", "42;30"]


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


def render_heatmap(data: list[DayActivity], *, no_color: bool = False) -> str:
    enabled = color_enabled(no_color)
    weeks = week_grid(data)
    lines = [style("LexiPilot Study Intensity", "1", enabled)]
    lines.append(month_labels(weeks))
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
        lines.append(f"{label} " + " ".join(cells).rstrip())

    total_words = sum(item.words for item in data)
    active_days = sum(1 for item in data if item.words > 0)
    missed = sum(item.missed for item in data)
    best = max(data, key=lambda item: item.words) if data else None
    lines.append("")
    lines.append("Legend: none='  ' light='..' medium='::' strong='++' intense='##'")
    lines.append(f"Days: {len(data)} | Active days: {active_days} | Words reviewed: {total_words} | Missed: {missed}")
    if best:
        lines.append(f"Peak day: {best.day.isoformat()} ({best.words} words)")
    lines.append("Source: deterministic random demo data")
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
