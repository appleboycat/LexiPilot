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
        return self.style(text, "1;96")

    def label(self, name: str) -> str:
        palette = {
            "PLAN": "1;36",
            "TOOL": "94",
            "SELECTED": "1;35",
            "ANSWER": "1;37",
            "ADAPT": "1;33",
            "GENERATE": "1;35",
            "SAVED": "1;32",
            "STATUS": "36",
            "WARNING": "33",
            "ERROR": "1;31",
            "FAIL": "1;31",
            "PASS": "1;32",
        }
        return self.style(f"[{name}]", palette.get(name, "1"))

    def word(self, text: str) -> str:
        return self.style(text, "1;93")

    def phonetic(self, text: str) -> str:
        return self.style(text, "36")

    def pos(self, text: str) -> str:
        return self.style(text, "35")

    def chinese_target(self, text: str) -> str:
        return self.style(text, "1;95")

    def correct(self, text: str) -> str:
        return self.style(text, "1;32")

    def incorrect(self, text: str) -> str:
        return self.style(text, "1;31")

    def skipped(self, text: str) -> str:
        return self.style(text, "33")

    def dim(self, text: str) -> str:
        return self.style(text, "2")

    def cyan(self, text: str) -> str:
        return self.style(text, "36")


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
        print(f"{self.theme.label(label)} {message}")

    def plan(self, message: str) -> None:
        self.event("PLAN", message)

    def tool(self, name: str) -> None:
        self.event("TOOL", self.theme.style(name, "1"))

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

    def answer(self, word: str, outcome: str) -> None:
        if outcome == "correct":
            rendered = self.theme.correct("correct")
        elif outcome == "incorrect":
            rendered = self.theme.incorrect("incorrect")
        else:
            rendered = self.theme.skipped("skipped")
        self.event("ANSWER", f"{word}: {rendered}")
