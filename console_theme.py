#!/usr/bin/env python3
"""Small semantic terminal renderer for LexiPilot."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


@dataclass
class ConsoleTheme:
    enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = should_enable_color()

    def style(self, text: str, code: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def title(self, text: str) -> str:
        return self.style(text, "1")

    def label(self, name: str) -> str:
        palette = {
            "AGENT": "1",
            "MODEL TOOL": "2",
            "MODEL PLAN": "1",
            "CONTROLLER": "1",
            "PLAN": "1",
            "TOOL": "2",
            "SELECTED": "1",
            "ANSWER": "1;37",
            "ADAPT": "1",
            "GENERATE": "1",
            "SAVED": "1;32",
            "STATUS": "2",
            "WARNING": "1",
            "ERROR": "1;31",
            "FAIL": "1;31",
            "PASS": "1;32",
        }
        return self.style(f"[{name}]", palette.get(name, "1"))

    def event_text(self, name: str, text: str) -> str:
        palette = {
            "AGENT": "2",
            "MODEL PLAN": "2",
            "CONTROLLER": "2",
            "PLAN": "2",
            "SELECTED": "2",
            "ADAPT": "2",
            "GENERATE": "2",
            "SAVED": "32",
            "STATUS": "2",
            "WARNING": "2",
            "ERROR": "31",
            "FAIL": "31",
            "PASS": "32",
        }
        code = palette.get(name)
        return self.style(text, code) if code else text

    def word(self, text: str) -> str:
        return self.style(text, "1;93")

    def phonetic(self, text: str) -> str:
        return self.style(text, "94")

    def pos(self, text: str) -> str:
        return self.style(text, "2")

    def definition(self, text: str) -> str:
        return self.style(text, "1;95")

    def chinese_target(self, text: str) -> str:
        return self.style(text, "1;95")

    def correct(self, text: str) -> str:
        return self.style(text, "1;32")

    def incorrect(self, text: str) -> str:
        return self.style(text, "1;31")

    def skipped(self, text: str) -> str:
        return self.style(text, "2")

    def dim(self, text: str) -> str:
        return self.style(text, "2")

    def cyan(self, text: str) -> str:
        return self.style(text, "2")


def should_enable_color() -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("FORCE_COLOR", "").strip() not in {"", "0", "false", "False"}:
        return True
    return sys.stdout.isatty()


def _english_pattern(word: str) -> re.Pattern[str]:
    base = re.escape(word)
    suffix = r"(?:s|es|ed|d|ing|red|ring)?" if word.endswith("r") else r"(?:s|es|ed|d|ing)?"
    return re.compile(rf"\b({base}{suffix})\b", re.IGNORECASE)


def highlight_english_terms(text: str, target_words: list[str], theme: ConsoleTheme) -> str:
    result = text
    for word in sorted({w for w in target_words if w}, key=len, reverse=True):
        pattern = _english_pattern(word)
        result = pattern.sub(lambda match: theme.word(match.group(1)), result)
    return result


def highlight_chinese_terms(text: str, target_translations: dict[str, list[str]], theme: ConsoleTheme) -> str:
    phrases = sorted({p for values in target_translations.values() for p in values if p}, key=len, reverse=True)
    result = text
    for phrase in phrases:
        result = result.replace(phrase, theme.chinese_target(phrase))
    return result


class Console:
    def __init__(self, theme: ConsoleTheme | None = None) -> None:
        self.theme = theme or ConsoleTheme()

    def line(self, text: str = "") -> None:
        print(text)

    def event(self, label: str, message: str) -> None:
        print(f"{self.theme.label(label)} {self.theme.event_text(label, message)}")

    def plan(self, message: str) -> None:
        self.event("PLAN", message)

    def agent(self, message: str) -> None:
        self.event("AGENT", message)

    def model_tool(self, name: str) -> None:
        print(self.theme.dim(f"[MODEL TOOL] {name}"))

    def model_plan(self, message: str) -> None:
        self.event("MODEL PLAN", message)

    def controller(self, message: str) -> None:
        self.event("CONTROLLER", message)

    def tool(self, name: str) -> None:
        print(self.theme.dim(f"[TOOL] {name}"))

    def selected(self, message: str) -> None:
        self.event("SELECTED", message)

    def adapt(self, message: str) -> None:
        self.event("ADAPT", message)

    def generate(self, message: str) -> None:
        self.event("GENERATE", message)

    def saved(self, message: str) -> None:
        self.event("SAVED", message)

    def error(self, message: str) -> None:
        self.event("ERROR", message)

    def status(self, message: str) -> None:
        self.event("STATUS", message)

    def warning(self, message: str) -> None:
        self.event("WARNING", message)

    def profile_status(self, summary: dict[str, object], runtime_summary: dict[str, str] | None = None) -> None:
        total = int(summary.get("total_vocabulary_count", 0) or 0)
        started = int(summary.get("started_word_count", 0) or 0)
        progress_ratio = (started / total) if total > 0 else 0.0
        progress_width = 20
        filled = max(0, min(progress_width, int(round(progress_ratio * progress_width))))
        bar = "█" * filled + "░" * (progress_width - filled)
        percent = progress_ratio * 100 if total else 0.0
        rows = [
            ("Profile", str(summary.get("profile", ""))),
            ("Started words", f"{started} / {total}"),
            ("Progress", f"[{bar}] {percent:.1f}%"),
            ("Due today", str(summary.get("reviews_due_today", 0))),
            ("Historical misses", str(summary.get("total_incorrect_answers", 0))),
            ("Current position", str(summary.get("current_new_word_position", 0))),
            ("Recent activity", f"{len(summary.get('recent_study_statistics', {}) or {})} days"),
        ]
        if runtime_summary:
            rows.extend(
                [
                    ("Model", runtime_summary.get("model", "")),
                    ("Endpoint", runtime_summary.get("endpoint", "")),
                ]
            )
        self.box("LexiPilot Status", rows)

    def box(self, title: str, rows: list[tuple[str, str]], width: int = 78) -> None:
        inner = max(40, width - 2)
        top = "╭" + "─" * inner + "╮"
        bottom = "╰" + "─" * inner + "╯"
        print(self.theme.dim(top))
        print(self.theme.dim("│ " + title.ljust(inner - 2) + " │"))
        print(self.theme.dim("│" + " " * inner + "│"))
        label_width = max((len(label) for label, _ in rows), default=0)
        for label, value in rows:
            text = f"  {label + ':':<{label_width + 1}}  {value}"
            if len(strip_ansi(text)) > inner:
                text = text[: max(0, inner - 1)] + "…"
            print(self.theme.dim("│") + self.theme.dim(text.ljust(inner)) + self.theme.dim("│"))
        print(self.theme.dim(bottom))

    def answer(self, word: str, outcome: str) -> None:
        if outcome == "correct":
            rendered = self.theme.correct("✓")
        elif outcome == "incorrect":
            rendered = self.theme.incorrect("✗")
        else:
            rendered = self.theme.skipped("skipped")
        self.event("ANSWER", rendered)
