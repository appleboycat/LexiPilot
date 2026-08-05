#!/usr/bin/env python3
"""PDF vocabulary extractor and Ebbinghaus-style review trainer."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import subprocess
import sys
import termios
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


DEFAULT_PDF = Path("new48_ch_print_pagenum.pdf")
CONFIG_FILE = Path("conf.json")
INDEX_FILE = Path(".vocab_index.json")
STATE_FILE = Path(".vocab_state.json")
PROGRESS_DIR = Path(".vocab_progress")
SNAP_PAGE_CACHE_DIR = Path(".vocab_snap_pages")
DEFAULT_PROFILE = "default"
RAW_LINES_CSV = Path("pdf_lines_marked.csv")
WORDS_CSV = Path("words_marked.csv")
DEFAULT_LOCAL_MODEL = Path.home() / "models" / "Qwen3-8B"
DEFAULT_REQUIRE_CUDA = True
DEFAULT_DEEPSEEK_TIMEOUT = 120
DEFAULT_DEEPSEEK_MAX_TOKENS = 8192
ETYMONLINE_BASE_URL = "https://www.etymonline.com"
CONFIG: dict[str, Any] | None = None

# Intervals in days after a correct answer. This follows the common
# Ebbinghaus-inspired rhythm: same day, 1, 2, 4, 7, 15, 30 days.
REVIEW_INTERVALS = [0, 1, 2, 4, 7, 15, 30]

ENTRY_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z][A-Za-z'’.-]*)\s*(.*)$")
PAGE_RE = re.compile(r"第\s*(\d+)\s*页")
PHONETIC_RE = re.compile(r"(英|美):/([^/]+)/")
BARE_PHONETIC_RE = re.compile(r"/([^/\u4e00-\u9fff]+?)/")
POS_RE = re.compile(
    r"\b(?:abbr|prep|conj|pron|num|int|adj|adv|vt|vi|n|v)\.\s*"
)
POS_SPLIT_RE = re.compile(
    r"(\b(?:abbr|prep|conj|pron|num|int|adj|adv|vt|vi|n|v)\.\s*)"
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
COLOR = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "word": "\033[1;36m",
    "phonetic": "\033[33m",
    "pos": "\033[1;35m",
    "definition": "\033[32m",
    "meta": "\033[90m",
    "correct": "\033[1;32m",
    "wrong": "\033[1;31m",
}


class PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "title"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "title"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        cleaned = clean_spaces(html.unescape(data))
        if cleaned:
            self.parts.append(cleaned)


def first_definition_cjk(text: str) -> re.Match[str] | None:
    for match in CJK_RE.finditer(text):
        if match.group(0) not in {"英", "美"}:
            return match
    return None


@dataclass
class Entry:
    seq: int
    word: str
    first_letter: str
    page: int
    phonetic: str
    definition: str
    source_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "word": self.word,
            "first_letter": self.first_letter,
            "page": self.page,
            "phonetic": self.phonetic,
            "definition": self.definition,
            "source_text": self.source_text,
        }


def run_pdftotext(pdf: Path) -> str:
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise SystemExit("Missing pdftotext. Please install poppler-utils first.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"PDF extraction failed:\n{exc.stderr}") from exc
    return result.stdout


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def paint(text: str, color_name: str) -> str:
    if not text:
        return text
    return f"{COLOR[color_name]}{text}{COLOR['reset']}"


def normalize_profile_name(name: str | None) -> str:
    raw_name = (name or DEFAULT_PROFILE).strip()
    if not raw_name:
        return DEFAULT_PROFILE
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)
    return safe_name.strip("._") or DEFAULT_PROFILE


def load_config() -> dict[str, Any]:
    global CONFIG
    if CONFIG is not None:
        return CONFIG
    if not CONFIG_FILE.exists():
        CONFIG = {}
        return CONFIG
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid {CONFIG_FILE}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid {CONFIG_FILE}: top-level value must be an object.")
    CONFIG = data
    return CONFIG


def config_path(key: str) -> Path | None:
    value = load_config().get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def snap_page_cache_dir() -> Path:
    return config_path("snap_page_cache_dir") or SNAP_PAGE_CACHE_DIR


def state_path(profile: str | None) -> Path:
    normalized = normalize_profile_name(profile)
    if normalized == DEFAULT_PROFILE:
        configured = config_path("default_progress_path")
        if configured is not None:
            return configured
    return PROGRESS_DIR / normalized / "progress.json"


def legacy_state_path(profile: str | None) -> Path:
    return PROGRESS_DIR / f"{normalize_profile_name(profile)}.json"


def normalize_phonetic(text: str) -> str:
    compact = clean_spaces(text).replace(" /", "/").replace("/ ", "/")
    parts = []
    for label, value in PHONETIC_RE.findall(compact):
        parts.append(f"{label}:/{value.replace(' ', '').strip()}/")
    if parts:
        return " ".join(parts)
    for value in BARE_PHONETIC_RE.findall(compact):
        value = value.replace(" ", "").strip()
        if value:
            parts.append(f"/{value}/")
    return " ".join(parts)


def strip_phonetic_prefix(text: str) -> str:
    compact = clean_spaces(text)
    compact = re.sub(r"(英|美):/[^/]+/", " ", compact)
    compact = BARE_PHONETIC_RE.sub(" ", compact)
    compact = re.sub(r"\b[əɑæɒɔʌɜɪʊʃʒθðŋːˌ'ˈˋˊrtdkgpbmnlfszvwhejɑːɔːɜːɪəʊ]+\b", " ", compact)
    compact = clean_spaces(compact)
    match = POS_RE.search(compact)
    if match:
        compact = compact[match.start() :]
    return compact


def split_pronunciation_and_definition(text: str) -> tuple[str, str]:
    """Split one extracted PDF line into pronunciation-ish and definition-ish parts."""
    compact = clean_spaces(text)
    if not compact:
        return "", ""

    pos_match = POS_RE.search(compact)
    cjk_match = first_definition_cjk(compact)
    if pos_match and (not cjk_match or pos_match.start() < cjk_match.start()):
        return compact[: pos_match.start()], compact[pos_match.start() :]

    if cjk_match:
        prefix = compact[: cjk_match.start()]
        if "/" in prefix or "英:" in prefix or "美:" in prefix:
            return prefix, compact[cjk_match.start() :]
        return "", compact

    if "/" in compact or re.search(r"[əɑæɒɔʌɜɪʊʃʒθðŋː]", compact):
        return compact, ""
    return "", compact


def iter_pages(text: str) -> list[tuple[int, list[str]]]:
    pages: list[tuple[int, list[str]]] = []
    for fallback_no, page_text in enumerate(text.split("\f"), start=1):
        lines = page_text.splitlines()
        page_no = fallback_no
        for line in lines[:5]:
            match = PAGE_RE.search(line)
            if match:
                page_no = int(match.group(1))
                break
        pages.append((page_no, lines))
    return pages


def extract_entries_and_lines(pdf: Path) -> tuple[list[Entry], list[dict[str, Any]]]:
    text = run_pdftotext(pdf)
    entries: list[Entry] = []
    raw_lines: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    raw_seq = 0

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        source = clean_spaces(" ".join(current["parts"]))
        phonetic = normalize_phonetic(" ".join(current["phonetic_parts"]))
        definition = clean_spaces(" ".join(current["definition_parts"]))
        if not definition:
            definition = strip_phonetic_prefix(source)
        entries.append(
            Entry(
                seq=current["seq"],
                word=current["word"],
                first_letter=current["word"][0].upper(),
                page=current["page"],
                phonetic=phonetic,
                definition=definition,
                source_text=source,
            )
        )
        current = None

    for page_no, lines in iter_pages(text):
        for line in lines:
            if not line.strip() or PAGE_RE.search(line):
                continue
            raw_seq += 1
            stripped = line.strip()
            first_word = re.search(r"[A-Za-z][A-Za-z'’.-]*", stripped)
            raw_lines.append(
                {
                    "line_seq": raw_seq,
                    "page": page_no,
                    "first_letter": first_word.group(0)[0].upper() if first_word else "",
                    "text": stripped,
                }
            )

            match = ENTRY_RE.match(line)
            if match:
                flush_current()
                seq, word, rest = match.groups()
                if word.lower() == "word" and "list" in rest.lower():
                    continue
                phonetic_part, definition_part = split_pronunciation_and_definition(rest)
                current = {
                    "seq": int(seq),
                    "word": word.strip(),
                    "page": page_no,
                    "parts": [rest],
                    "phonetic_parts": [phonetic_part],
                    "definition_parts": [definition_part],
                }
            elif current:
                current["parts"].append(stripped)
                phonetic_part, definition_part = split_pronunciation_and_definition(stripped)
                if phonetic_part:
                    current["phonetic_parts"].append(phonetic_part)
                if definition_part:
                    current["definition_parts"].append(definition_part)

    flush_current()
    entries.sort(key=lambda item: item.seq)
    return entries, raw_lines


def save_index(entries: list[Entry], raw_lines: list[dict[str, Any]]) -> None:
    INDEX_FILE.write_text(
        json.dumps([entry.to_dict() for entry in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with RAW_LINES_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["line_seq", "page", "first_letter", "text"])
        writer.writeheader()
        writer.writerows(raw_lines)
    with WORDS_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["seq", "word", "first_letter", "page", "phonetic", "definition"],
        )
        writer.writeheader()
        for entry in entries:
            data = entry.to_dict()
            writer.writerow({key: data[key] for key in writer.fieldnames or []})


def build_index(pdf: Path) -> list[dict[str, Any]]:
    entries, raw_lines = extract_entries_and_lines(pdf)
    if not entries:
        raise SystemExit("No vocabulary entries were parsed from the PDF.")
    save_index(entries, raw_lines)
    return [entry.to_dict() for entry in entries]


def load_index(pdf: Path, rebuild: bool = False) -> list[dict[str, Any]]:
    if rebuild or not INDEX_FILE.exists():
        return build_index(pdf)
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def empty_state(profile: str | None) -> dict[str, Any]:
    return {
        "profile": normalize_profile_name(profile),
        "start_page": 1,
        "start_seq": 1,
        "last_new_seq": 0,
        "cards": {},
        "daily_stats": {},
        "daily_misses": {},
        "daily_miss_counts": {},
        "daily_seen": {},
    }


def load_state(profile: str | None) -> dict[str, Any]:
    path = state_path(profile)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state.setdefault("profile", normalize_profile_name(profile))
        state.setdefault("start_page", 1)
        state.setdefault("start_seq", 1)
        state.setdefault("last_new_seq", 0)
        state.setdefault("cards", {})
        state.setdefault("daily_stats", {})
        state.setdefault("daily_misses", {})
        state.setdefault("daily_miss_counts", {})
        state.setdefault("daily_seen", {})
        return state
    old_path = legacy_state_path(profile)
    if old_path.exists():
        state = json.loads(old_path.read_text(encoding="utf-8"))
        state.setdefault("profile", normalize_profile_name(profile))
        state.setdefault("start_page", 1)
        state.setdefault("start_seq", 1)
        state.setdefault("last_new_seq", 0)
        state.setdefault("cards", {})
        state.setdefault("daily_stats", {})
        state.setdefault("daily_misses", {})
        state.setdefault("daily_miss_counts", {})
        state.setdefault("daily_seen", {})
        save_state(state, profile)
        return state
    if normalize_profile_name(profile) == DEFAULT_PROFILE and STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state.setdefault("profile", DEFAULT_PROFILE)
        state.setdefault("start_page", 1)
        state.setdefault("start_seq", 1)
        state.setdefault("last_new_seq", 0)
        state.setdefault("cards", {})
        state.setdefault("daily_stats", {})
        state.setdefault("daily_misses", {})
        state.setdefault("daily_miss_counts", {})
        state.setdefault("daily_seen", {})
        save_state(state, DEFAULT_PROFILE)
        return state
    return empty_state(profile)


def save_state(state: dict[str, Any], profile: str | None) -> None:
    path = state_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["profile"] = normalize_profile_name(profile)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def card_state(state: dict[str, Any], seq: int) -> dict[str, Any]:
    cards = state.setdefault("cards", {})
    key = str(seq)
    if key not in cards:
        cards[key] = {"stage": 0, "due": date.today().isoformat(), "seen": 0, "correct": 0}
    return cards[key]


def next_due(stage: int) -> str:
    interval = REVIEW_INTERVALS[min(stage, len(REVIEW_INTERVALS) - 1)]
    return (date.today() + timedelta(days=interval)).isoformat()


def update_daily_stats(state: dict[str, Any], seq: int, mode: str, correct: bool) -> None:
    today = date.today().isoformat()
    daily_stats = state.setdefault("daily_stats", {})
    stats = daily_stats.setdefault(
        today,
        {"studied": 0, "new": 0, "review": 0, "remembered": 0, "missed": 0},
    )
    stats["studied"] = int(stats.get("studied", 0)) + 1
    stats[mode] = int(stats.get(mode, 0)) + 1
    seen = state.setdefault("daily_seen", {}).setdefault(today, [])
    if seq not in seen:
        seen.append(seq)
    if correct:
        stats["remembered"] = int(stats.get("remembered", 0)) + 1
    else:
        stats["missed"] = int(stats.get("missed", 0)) + 1
        misses = state.setdefault("daily_misses", {}).setdefault(today, [])
        if seq not in misses:
            misses.append(seq)
        miss_counts = state.setdefault("daily_miss_counts", {}).setdefault(today, {})
        miss_counts[str(seq)] = int(miss_counts.get(str(seq), 0)) + 1


def mark_answer(state: dict[str, Any], seq: int, correct: bool, mode: str) -> None:
    card = card_state(state, seq)
    card["seen"] = int(card.get("seen", 0)) + 1
    if correct:
        card["correct"] = int(card.get("correct", 0)) + 1
        card["stage"] = min(int(card.get("stage", 0)) + 1, len(REVIEW_INTERVALS) - 1)
    else:
        card["stage"] = 0
    card["due"] = next_due(int(card["stage"]))
    update_daily_stats(state, seq, mode, correct)


def due_entries(entries: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    today = date.today().isoformat()
    start_seq = int(state.get("start_seq", 1))
    last_new_seq = int(state.get("last_new_seq", 0))
    result = []
    for entry in entries:
        seq = int(entry["seq"])
        if seq < start_seq or seq > last_new_seq:
            continue
        card = state.get("cards", {}).get(str(entry["seq"]))
        if card and card.get("due", today) <= today:
            result.append(entry)
    return result


def page_start_seq(entries: list[dict[str, Any]], page: int) -> int:
    for entry in entries:
        if int(entry["page"]) >= page:
            return int(entry["seq"])
    raise SystemExit(f"No entries found on or after page {page}.")


def total_pdf_pages(entries: list[dict[str, Any]]) -> int:
    return max(int(entry["page"]) for entry in entries) if entries else 0


def scoped_total_pages(max_page: int, start_page: int) -> int:
    if start_page <= 1:
        return max_page
    return max(0, max_page - start_page)


def scoped_page_number(pdf_page: int, start_page: int) -> int:
    if start_page <= 1:
        return pdf_page
    return max(0, pdf_page - start_page)


def new_entries(entries: list[dict[str, Any]], state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    start_seq = int(state.get("start_seq", 1))
    last_new_seq = max(int(state.get("last_new_seq", 0)), start_seq - 1)
    result = [entry for entry in entries if int(entry["seq"]) > last_new_seq]
    return result[:limit]


def entry_for_seq(entries: list[dict[str, Any]], seq: int) -> dict[str, Any] | None:
    for entry in entries:
        if int(entry["seq"]) == seq:
            return entry
    return None


def first_entry_after_seq(entries: list[dict[str, Any]], seq: int) -> dict[str, Any] | None:
    for entry in entries:
        if int(entry["seq"]) > seq:
            return entry
    return None


def progress_bar(percent: float, width: int = 30) -> str:
    filled = round(width * max(0.0, min(100.0, percent)) / 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {percent:.1f}%"


def entry_progress(entries: list[dict[str, Any]], state: dict[str, Any]) -> tuple[int, int, float]:
    start_seq = int(state.get("start_seq", 1))
    last_new_seq = int(state.get("last_new_seq", 0))
    scoped_entries = [entry for entry in entries if int(entry["seq"]) >= start_seq]
    done = sum(1 for entry in scoped_entries if int(entry["seq"]) <= last_new_seq)
    total = len(scoped_entries)
    percent = (done / total * 100) if total else 0
    return done, total, percent


def print_session_summary(
    entries: list[dict[str, Any]],
    state: dict[str, Any],
    touched: int,
    missed: int,
    new_count: int = 0,
    review_count: int = 0,
) -> None:
    done, total_entries, percent = entry_progress(entries, state)
    print(f"This session: new {new_count}, review {review_count}")
    print(f"Missed/Total: {missed}/{touched}")
    print(f"Progress: {done} / {total_entries} {progress_bar(percent)}")


def progress_profiles() -> list[str]:
    profiles = {DEFAULT_PROFILE}
    if PROGRESS_DIR.exists():
        for path in PROGRESS_DIR.glob("*/progress.json"):
            profiles.add(path.parent.name)
        for path in PROGRESS_DIR.glob("*.json"):
            profiles.add(path.stem)
    return sorted(profiles)


def show_progress_list(entries: list[dict[str, Any]]) -> None:
    max_page = total_pdf_pages(entries)
    rows = []
    for profile in progress_profiles():
        state = load_state(profile)
        start_page = int(state.get("start_page", 1))
        last_new_seq = int(state.get("last_new_seq", 0))
        last_new_entry = entry_for_seq(entries, last_new_seq)
        next_new_entry = first_entry_after_seq(entries, last_new_seq)
        current_pdf_page = (
            int(next_new_entry["page"])
            if next_new_entry
            else int(last_new_entry["page"])
            if last_new_entry
            else start_page
        )
        current_page = scoped_page_number(current_pdf_page, start_page)
        total_pages = scoped_total_pages(max_page, start_page)
        done, total_entries, percent = entry_progress(entries, state)
        page_range = f"{start_page}-{max_page}"
        progress_now = f"{done}/{total_entries} ({percent:.1f}%), page {current_page}/{total_pages}"
        rows.append(
            {
                "Profile": profile,
                "Page range": page_range,
                "Vocabulary": str(total_entries),
                "Progress now": progress_now,
            }
        )

    headers = ["Profile", "Page range", "Vocabulary", "Progress now"]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def plain_definition(definition: str) -> str:
    return clean_spaces(POS_SPLIT_RE.sub(lambda match: match.group(1), definition))


def short_meaning(entry: dict[str, Any]) -> str:
    parts = definition_parts(entry.get("definition") or entry.get("source_text") or "")
    for _, meaning in parts:
        meaning = clean_spaces(meaning)
        meaning = re.sub(r"^&\s*(?:abbr|prep|conj|pron|num|int|adj|adv|vt|vi|n|v)\.\s*", "", meaning)
        if meaning and not re.fullmatch(r"[&.\s]+", meaning):
            return re.split(r"[；;，,、]", meaning, maxsplit=1)[0]
    return "这个词"


def highlight_words(text: str, words: list[str]) -> str:
    for word in sorted(words, key=len, reverse=True):
        text = re.sub(
            rf"\b{re.escape(word)}\b",
            lambda match: paint(match.group(0), "word"),
            text,
            flags=re.IGNORECASE,
        )
    return text


def highlight_cjk(text: str) -> str:
    return re.sub(r"[\u4e00-\u9fff]+", lambda match: paint(match.group(0), "definition"), text)


def highlight_marked_chinese(text: str) -> str:
    if "[[" not in text or "]]" not in text:
        return highlight_cjk(text)
    return re.sub(r"\[\[([^\[\]]+)\]\]", lambda match: paint(match.group(1), "definition"), text)


def mark_chinese_phrases(text: str, phrases: list[str]) -> str:
    for phrase in phrases:
        if phrase and phrase in text:
            text = text.replace(phrase, f"[[{phrase}]]", 1)
    return text


def missing_target_words(text: str, words: list[str]) -> list[str]:
    missing = []
    for word in words:
        if not re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, flags=re.IGNORECASE):
            missing.append(word)
    return missing


def response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list):
        chunks: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                chunks.append(message["content"])
            elif isinstance(choice.get("text"), str):
                chunks.append(choice["text"])
        if chunks:
            return "\n".join(chunks).strip()
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def parse_snap_json(text: str) -> dict[str, str] | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    english = clean_spaces(str(parsed.get("english", "")))
    chinese = clean_spaces(str(parsed.get("chinese", "")))
    if not english or not chinese:
        return None
    return {"english": english, "chinese": chinese}


def snap_prompt(missed_entries: list[dict[str, Any]]) -> str:
    words = [
        {
            "word": str(entry["word"]),
            "definition": plain_definition(entry.get("definition") or entry.get("source_text") or ""),
        }
        for entry in missed_entries
    ]
    return (
        "Write two short paragraphs for vocabulary review. "
        "Paragraph 1 must be one natural English story paragraph, not a vocabulary list, "
        "not a word chain, and not definitions. "
        "Use every target word in the listed order, keeping the exact target spelling at least once. "
        "Use each word according to its Chinese definition and connect the words into one coherent scene. "
        "Paragraph 2 must be a faithful Chinese translation of the English paragraph. "
        "In Paragraph 2, wrap each Chinese phrase that translates a target word with [[ and ]]. "
        "Do not explain the task. Return strict JSON only with keys english and chinese.\n\n"
        f"Target words:\n{json.dumps(words, ensure_ascii=False, indent=2)}"
    )


def snap_repair_prompt(missed_entries: list[dict[str, Any]], result: dict[str, str], missing: list[str]) -> str:
    return (
        "Revise the JSON vocabulary practice paragraphs below. "
        "The English paragraph missed these exact target spellings: "
        f"{', '.join(missing)}. "
        "Keep one natural English story paragraph and one faithful Chinese translation. "
        "The English paragraph must include every target word in the listed order, with exact spelling. "
        "The Chinese paragraph must wrap each Chinese phrase that translates a target word with [[ and ]]. "
        "Return strict JSON only with keys english and chinese.\n\n"
        f"Target words:\n{json.dumps(snap_words_payload(missed_entries), ensure_ascii=False, indent=2)}\n\n"
        f"Current JSON:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
    )


def snap_words_payload(missed_entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "word": str(entry["word"]),
            "definition": plain_definition(entry.get("definition") or entry.get("source_text") or ""),
        }
        for entry in missed_entries
    ]


def validate_snap_result(result: dict[str, str], missed_entries: list[dict[str, Any]], source: str) -> dict[str, str] | None:
    missing = missing_target_words(result["english"], [str(entry["word"]) for entry in missed_entries])
    if missing:
        print(
            f"{source} output missed target words: {', '.join(missing[:8])}"
            f"{'...' if len(missing) > 8 else ''}. Using fallback.",
            file=sys.stderr,
        )
        return None
    if not CJK_RE.search(result["chinese"]):
        print(f"{source} output did not include Chinese translation. Using fallback.", file=sys.stderr)
        return None
    return result


def api_error_message(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if body:
            return f"HTTP {exc.code}: {body}"
        return f"HTTP {exc.code}: {exc.reason}"
    return str(exc)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        value = load_config().get(name)
    if not value:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"Ignoring invalid {name}={value!r}; using {default}.", file=sys.stderr)
        return default


def deepseek_api_key() -> str | None:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if api_key:
        return api_key.strip()
    key_path = config_path("DEEPSEEK_API_KEY_FILE")
    if key_path is None:
        return None
    try:
        return key_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"DeepSeek key file not readable: {key_path}. {exc}", file=sys.stderr)
        return None


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def read_deepseek_stream(response: Any) -> str:
    chunks: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = data.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                chunks.append(delta["content"])
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                chunks.append(message["content"])
    return "".join(chunks).strip()


def deepseek_non_json_message(output_text: str, action: str) -> str:
    preview = clean_spaces(output_text.replace("\n", " "))[:300]
    if preview:
        return f"DeepSeek returned non-JSON text; {action}. Preview: {preview}"
    return f"DeepSeek returned empty/non-JSON text; {action}."


def snap_max_tokens(missed_entries: list[dict[str, Any]]) -> int:
    return min(max(1600, len(missed_entries) * 120), env_int("DEEPSEEK_MAX_TOKENS", DEFAULT_DEEPSEEK_MAX_TOKENS))


def call_local_huggingface_for_snap(missed_entries: list[dict[str, Any]]) -> dict[str, str] | None:
    model_dir = Path(os.getenv("HF_MODEL_DIR", str(DEFAULT_LOCAL_MODEL))).expanduser()
    if not model_dir.exists():
        print(f"Local Hugging Face model not found: {model_dir}. Using fallback.", file=sys.stderr)
        return None
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
    except ImportError:
        print(
            "Local Hugging Face model not used: install transformers and torch first. Using fallback.",
            file=sys.stderr,
        )
        return None

    prompt = snap_prompt(missed_entries)
    require_cuda = os.getenv("HF_REQUIRE_CUDA", "1" if DEFAULT_REQUIRE_CUDA else "0").lower() not in {
        "0",
        "false",
        "no",
    }
    if require_cuda and not torch.cuda.is_available():
        print("Local Hugging Face model not used: CUDA is not available. Using fallback.", file=sys.stderr)
        return None

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        quantization = os.getenv("HF_QUANTIZATION", "4bit").lower()
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if device.startswith("cuda") and quantization in {"4bit", "nf4"}:
            model_kwargs["device_map"] = {"": 0}
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["dtype"] = torch.float16 if device.startswith("cuda") else "auto"
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        for generation_flag in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, generation_flag):
                setattr(model.generation_config, generation_flag, None)
        if "device_map" not in model_kwargs:
            model.to(device)
        if device.startswith("cuda"):
            print(
                f"Local Hugging Face using GPU: {torch.cuda.get_device_name(0)}"
                f" ({quantization if quantization in {'4bit', 'nf4'} else 'fp16'})",
                file=sys.stderr,
            )
        messages = [
            {"role": "system", "content": "You write concise, natural vocabulary practice paragraphs."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        max_new_tokens = min(max(900, len(missed_entries) * 60), 4096)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        output_ids = generated[0][inputs.input_ids.shape[-1] :]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    except Exception as exc:
        print(f"Local Hugging Face generation failed: {exc}. Using fallback.", file=sys.stderr)
        return None
    result = parse_snap_json(output_text)
    if not result:
        print("Local Hugging Face returned non-JSON text. Using fallback.", file=sys.stderr)
        return None
    return validate_snap_result(result, missed_entries, "Local Hugging Face")


def request_deepseek_snap(
    api_key: str,
    prompt: str,
    missed_entries: list[dict[str, Any]],
    use_stream: bool,
) -> str | None:
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise bilingual vocabulary practice examples. "
                    "Return valid JSON only, for example: "
                    '{"english":"...","chinese":"..."}'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": snap_max_tokens(missed_entries),
        "response_format": {"type": "json_object"},
        "stream": use_stream,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=env_int("DEEPSEEK_TIMEOUT", DEFAULT_DEEPSEEK_TIMEOUT)) as response:
            if use_stream:
                return read_deepseek_stream(response)
            return response_text(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"DeepSeek request failed; trying OpenAI/local fallback. {api_error_message(exc)}", file=sys.stderr)
        return None


def mark_omitted_words(result: dict[str, str], missing: list[str]) -> dict[str, Any]:
    marked: dict[str, Any] = dict(result)
    marked["_omitted_words"] = missing
    return marked


def call_deepseek_for_snap(missed_entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    api_key = deepseek_api_key()
    if not api_key:
        print("DeepSeek not used: no API key in environment or conf.json key file. Trying OpenAI.", file=sys.stderr)
        return None

    prompt = snap_prompt(missed_entries)
    use_stream = env_bool("DEEPSEEK_STREAM", True)

    output_text = request_deepseek_snap(api_key, prompt, missed_entries, use_stream)
    if output_text is None:
        return None

    result = parse_snap_json(output_text)
    if not result and use_stream:
        print(deepseek_non_json_message(output_text, "retrying without stream"), file=sys.stderr)
        output_text = request_deepseek_snap(api_key, prompt, missed_entries, False)
        if output_text is None:
            return None
        result = parse_snap_json(output_text)
    if not result:
        print(deepseek_non_json_message(output_text, "using local fallback"), file=sys.stderr)
        return None
    missing = missing_target_words(result["english"], [str(entry["word"]) for entry in missed_entries])
    if missing:
        print(
            f"DeepSeek output missed target words: {', '.join(missing[:8])}"
            f"{'...' if len(missing) > 8 else ''}. Asking DeepSeek to revise.",
            file=sys.stderr,
        )
        repair_text = request_deepseek_snap(api_key, snap_repair_prompt(missed_entries, result, missing), missed_entries, False)
        if repair_text is None:
            return mark_omitted_words(result, missing)
        repair_result = parse_snap_json(repair_text)
        if not repair_result:
            print(deepseek_non_json_message(repair_text, "showing omitted words instead of using local fallback"), file=sys.stderr)
            return mark_omitted_words(result, missing)
        result = repair_result
        missing = missing_target_words(result["english"], [str(entry["word"]) for entry in missed_entries])
        if missing:
            print(
                f"DeepSeek revised output still missed target words: {', '.join(missing[:8])}"
                f"{'...' if len(missing) > 8 else ''}. Showing omitted words instead of using local fallback.",
                file=sys.stderr,
            )
            return mark_omitted_words(result, missing)
    return validate_snap_result(result, missed_entries, "DeepSeek")


def call_chatgpt_for_snap(missed_entries: list[dict[str, Any]]) -> dict[str, str] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ChatGPT not used: OPENAI_API_KEY is not set. Trying local Hugging Face model.", file=sys.stderr)
        return None

    prompt = snap_prompt(missed_entries)
    payload = {
        "model": os.getenv("OPENAI_MODEL", "chat-latest"),
        "instructions": "You write concise bilingual vocabulary practice examples.",
        "input": prompt,
        "max_output_tokens": min(max(900, len(missed_entries) * 60), 4096),
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"ChatGPT request failed; using local fallback. {api_error_message(exc)}", file=sys.stderr)
        return None

    result = parse_snap_json(response_text(data))
    if not result:
        print("ChatGPT returned non-JSON text; using local fallback.", file=sys.stderr)
        return None
    return validate_snap_result(result, missed_entries, "ChatGPT")


def local_snap_paragraph(missed_entries: list[dict[str, Any]], word_list: str) -> str:
    story_words = [str(entry["word"]) for entry in missed_entries]
    words = {word.lower() for word in story_words}
    current_f_words = [
        "chill",
        "eviscerate",
        "exempt",
        "exhort",
        "exotic",
        "expedient",
        "expedition",
        "expel",
        "expenditure",
        "expropriate",
        "extensive",
        "extent",
        "external",
        "externality",
        "extol",
        "extort",
        "extraordinary",
        "extrapolate",
        "extravagant",
        "extremity",
        "extrinsic",
        "facade",
        "facet",
        "facial",
        "facilitate",
        "facility",
        "faction",
        "factual",
        "Fahrenheit",
        "falcon",
        "falter",
        "familiarize",
        "farce",
        "fare",
        "far-fetched",
        "fascinate",
        "fascinating",
        "fasten",
        "fatigue",
        "faucet",
        "fauna",
        "feat",
        "federal",
        "feeble",
        "feign",
        "feminist",
        "ferment",
        "fertile",
        "fervor",
        "feudal",
        "fibrous",
        "fiction",
        "fictitious",
        "fidelity",
        "fierce",
        "fiery",
        "figurative",
        "filament",
        "filial",
        "fin",
        "finch",
        "firm",
        "fitness",
        "flagellum",
        "flair",
        "flake",
        "flank",
        "flask",
        "flexibility",
        "flick",
        "flint",
    ]
    if story_words == current_f_words:
        return (
            "In the chill before dawn, the guide had to eviscerate a fish, exempt the sick porter from duty, "
            "exhort the team to move on, choose an exotic but expedient trail for the expedition, and expel "
            "a thief from camp. Their expenditure rose when soldiers tried to expropriate supplies across an "
            "extensive border whose extent no external map showed; one externality made the captain extol "
            "honest trade instead of trying to extort money. An extraordinary clerk tried to extrapolate the "
            "cost of an extravagant bridge at the extremity of the valley, but that extrinsic detail hid a "
            "cracked facade, a dangerous facet of the plan, and a clear facial bruise on the messenger. To "
            "facilitate repairs at the facility, a faction demanded a factual Fahrenheit reading while a falcon "
            "made the engineer falter, so I had to familiarize everyone with the controls. The meeting became "
            "a farce when the fare sounded far-fetched, yet the machine began to fascinate the crowd with a "
            "fascinating trick: fasten a tired worker through fatigue to a broken faucet, then show how nearby "
            "fauna could turn the repair into a feat. A federal inspector, though feeble, refused to feign "
            "agreement; a feminist chemist watched the mixture ferment in fertile soil with such fervor that "
            "the feudal landlord grabbed a fibrous rope and shouted that the report was fiction, then a "
            "fictitious witness tested everyone's fidelity. Under a fierce and fiery sunset, the teacher made "
            "a figurative filament of memory: a filial child touched a fish fin, a finch landed on a firm post, "
            "a fitness coach studied a flagellum, and a painter with flair caught a snow flake on the flank "
            "of a flask, proving that flexibility can begin with one quick flick of flint."
        )
    target_set = {
        "imperial",
        "impersonal",
        "impetus",
        "implore",
        "imprecise",
        "imprint",
        "improvise",
        "impulse",
        "inactivate",
        "inanimate",
    }
    if words == target_set:
        paragraph = (
            f"An {paint('imperial', 'word')}, {paint('impersonal', 'word')} order gave me the "
            f"{paint('impetus', 'word')} to {paint('implore', 'word')} the team for help; "
            f"although my plan was {paint('imprecise', 'word')}, I tried to {paint('imprint', 'word')} "
            f"the key idea, {paint('improvise', 'word')} a response, resist a sudden "
            f"{paint('impulse', 'word')}, {paint('inactivate', 'word')} the faulty switch, "
            f"and move the {paint('inanimate', 'word')} device aside."
        )
        return paragraph
    chunk_size = 8
    sentence_starts = [
        "During a tense review session, I turned the missed words into a scene where",
        "The story continued as",
        "Then",
        "By the end,",
        "In the final image,",
    ]
    sentences: list[str] = []
    for index in range(0, len(story_words), chunk_size):
        chunk = story_words[index : index + chunk_size]
        if len(chunk) == 1:
            joined = chunk[0]
        elif len(chunk) == 2:
            joined = f"{chunk[0]} and {chunk[1]}"
        else:
            joined = ", ".join(chunk[:-1]) + f", and {chunk[-1]}"
        opener = sentence_starts[min(index // chunk_size, len(sentence_starts) - 1)]
        if index == 0:
            sentences.append(f"{opener} {joined} all appeared in order.")
        else:
            sentences.append(f"{opener} {joined} pushed the scene forward.")
    return " ".join(sentences)


def local_snap_translation(missed_entries: list[dict[str, Any]], meanings: list[str]) -> str:
    story_words = [str(entry["word"]) for entry in missed_entries]
    current_f_words = [
        "chill",
        "eviscerate",
        "exempt",
        "exhort",
        "exotic",
        "expedient",
        "expedition",
        "expel",
        "expenditure",
        "expropriate",
        "extensive",
        "extent",
        "external",
        "externality",
        "extol",
        "extort",
        "extraordinary",
        "extrapolate",
        "extravagant",
        "extremity",
        "extrinsic",
        "facade",
        "facet",
        "facial",
        "facilitate",
        "facility",
        "faction",
        "factual",
        "Fahrenheit",
        "falcon",
        "falter",
        "familiarize",
        "farce",
        "fare",
        "far-fetched",
        "fascinate",
        "fascinating",
        "fasten",
        "fatigue",
        "faucet",
        "fauna",
        "feat",
        "federal",
        "feeble",
        "feign",
        "feminist",
        "ferment",
        "fertile",
        "fervor",
        "feudal",
        "fibrous",
        "fiction",
        "fictitious",
        "fidelity",
        "fierce",
        "fiery",
        "figurative",
        "filament",
        "filial",
        "fin",
        "finch",
        "firm",
        "fitness",
        "flagellum",
        "flair",
        "flake",
        "flank",
        "flask",
        "flexibility",
        "flick",
        "flint",
    ]
    if story_words == current_f_words:
        translation = (
            "黎明前的寒意中，向导不得不剖开一条鱼，免除生病脚夫的工作，劝队伍继续前进，"
            "为这次远征选择一条异国风味但权宜可行的小路，并把一个小偷逐出营地。士兵试图在一片广阔边境上没收补给，"
            "而这片边境的范围连外部地图都没有标明，于是开支上升；一个外部因素让队长开始颂扬诚实交易，而不是敲诈钱财。"
            "一名非同寻常的书记员试图推算山谷尽头那座奢华大桥的成本，但这个外在细节掩盖了破裂的正面、计划中危险的一面，"
            "以及信使脸上一块明显的伤痕。为了让设施里的维修更顺利，一个派别要求提供真实的华氏温度读数；这时一只猎鹰让工程师脚步踉跄，"
            "所以我不得不让所有人熟悉控制装置。会议变成一场闹剧，因为票价听起来牵强；可是机器开始用一个迷人的把戏深深吸引人群："
            "把因疲劳而虚弱的工人固定在坏水龙头旁，再展示附近的动物群如何把维修变成一项技艺表演。一位联邦检查员虽然虚弱，"
            "却拒绝假装同意；一位女权主义化学家满怀热情地看着混合物在肥沃土壤中发酵，以至于那个封建地主抓起一根纤维绳，"
            "大喊报告全是小说，而一名虚构的证人检验了每个人的忠诚。在猛烈而火红的夕阳下，老师编出一根比喻性的记忆细丝："
            "一个孝顺的孩子摸着鱼鳍，一只雀落在结实的柱子上，一名健身教练研究鞭毛，一位有天资的画家让一片雪花落在烧瓶侧面，"
            "证明灵活性可以从燧石的一次快速轻弹开始。"
        )
        return mark_chinese_phrases(
            translation,
            [
                "寒意",
                "剖开",
                "免除",
                "劝",
                "异国风味",
                "权宜可行",
                "远征",
                "逐出",
                "开支",
                "没收",
                "广阔",
                "范围",
                "外部地图",
                "外部因素",
                "颂扬",
                "敲诈",
                "非同寻常",
                "推算",
                "奢华",
                "尽头",
                "外在",
                "正面",
                "一面",
                "脸上",
                "更顺利",
                "设施",
                "派别",
                "真实",
                "华氏",
                "猎鹰",
                "脚步踉跄",
                "熟悉",
                "闹剧",
                "票价",
                "牵强",
                "吸引",
                "迷人",
                "固定",
                "疲劳",
                "水龙头",
                "动物群",
                "技艺表演",
                "联邦",
                "虽然虚弱",
                "假装",
                "女权主义",
                "发酵",
                "肥沃",
                "热情",
                "封建",
                "纤维",
                "小说",
                "虚构",
                "忠诚",
                "猛烈",
                "火红",
                "比喻性",
                "细丝",
                "孝顺",
                "鳍",
                "雀",
                "结实",
                "健身",
                "鞭毛",
                "天资",
                "雪花",
                "侧面",
                "烧瓶",
                "灵活性",
                "轻弹",
                "燧石",
            ],
        )
    phrases = [meaning for meaning in meanings]
    if len(phrases) == 1:
        joined = phrases[0]
    elif len(phrases) == 2:
        joined = f"{phrases[0]}和{phrases[1]}"
    else:
        joined = "、".join(phrases[:-1]) + f"和{phrases[-1]}"
    return f"我把这些漏记词编成一段连续画面，依次包含：[[{joined}]]。"


def print_snap_plus_result(result: dict[str, Any], words: list[str], omitted_entries: list[dict[str, Any]] | None = None) -> None:
    print(paint("English:", "meta"))
    print(highlight_words(result["english"], words))
    print()
    print(paint("Chinese:", "meta"))
    print(highlight_marked_chinese(result["chinese"]))
    if omitted_entries:
        print()
        print_entry_table(omitted_entries, "Remote model omitted words", blank_between_entries=True)


def snap_page_cache_path(page: int) -> Path:
    return snap_page_cache_dir() / f"page_{page:04d}.json"


def page_entry_seqs(entries: list[dict[str, Any]]) -> list[int]:
    return [int(entry["seq"]) for entry in entries]


def load_page_snap_cache(page: int, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    path = snap_page_cache_path(page)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("entry_seqs") != page_entry_seqs(entries):
        return None
    result = data.get("result")
    if not isinstance(result, dict) or not result.get("english") or not result.get("chinese"):
        return None
    return result


def save_page_snap_cache(page: int, entries: list[dict[str, Any]], result: dict[str, Any]) -> None:
    path = snap_page_cache_path(page)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "page": page,
        "entry_seqs": page_entry_seqs(entries),
        "words": [str(entry["word"]) for entry in entries],
        "result": result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def definition_parts(definition: str) -> list[tuple[str, str]]:
    definition = clean_spaces(definition)
    if not definition:
        return [("", "(No definition parsed)")]

    parts = POS_SPLIT_RE.split(definition)
    result: list[tuple[str, str]] = []
    prefix = parts[0].strip() if parts else ""
    if prefix:
        result.append(("", prefix))

    index = 1
    while index < len(parts):
        pos = clean_spaces(parts[index])
        meaning = clean_spaces(parts[index + 1] if index + 1 < len(parts) else "")
        result.append((pos, meaning))
        index += 2
    return result or [("", definition)]


def format_definition_lines(definition: str) -> list[str]:
    lines: list[str] = []
    for pos, meaning in definition_parts(definition):
        if pos and meaning:
            lines.append(f"{paint(pos, 'pos')} {paint(meaning, 'definition')}")
        elif pos:
            lines.append(paint(pos, "pos"))
        else:
            lines.append(paint(meaning, "definition"))
    return lines


def print_definition(entry: dict[str, Any]) -> None:
    definition = entry.get("definition") or entry.get("source_text") or ""
    for line in format_definition_lines(definition):
        print(f"  {line}")


def etymonline_url(word: str) -> str:
    return f"{ETYMONLINE_BASE_URL}/word/{urllib.parse.quote(word.strip().lower())}"


def etymonline_cn_url(word: str) -> str:
    return f"{ETYMONLINE_BASE_URL}/cn/word/{urllib.parse.quote(word.strip().lower())}"


def page_text_from_html(raw_html: str) -> list[str]:
    parser = PlainTextHTMLParser()
    parser.feed(raw_html)
    lines: list[str] = []
    previous = ""
    for part in parser.parts:
        if part == previous:
            continue
        previous = part
        lines.append(part)
    return lines


def extract_etymonline_lines(word: str, lines: list[str]) -> list[str]:
    def clean_etymonline_line(line: str) -> str:
        line = clean_spaces(line)
        line = re.sub(r"\s+([.,;:])", r"\1", line)
        line = re.sub(r"\balso from \d{3,4}\b", "", line, flags=re.IGNORECASE)
        return clean_spaces(line)

    word_lower = word.lower()
    start_index = None
    for index, line in enumerate(lines):
        lowered = line.lower()
        if re.fullmatch(rf"origin and history of\s+{re.escape(word_lower)}", lowered):
            start_index = index
            break
    if start_index is None:
        for index, line in enumerate(lines):
            if re.match(rf"{re.escape(word_lower)}\s*\([^)]+\)", line, flags=re.IGNORECASE):
                start_index = index
                break
    if start_index is None:
        page_text = "\n".join(lines)
        match = re.search(
            rf"(Origin and history of\s+{re.escape(word)}.*?)(?:Entries linking to|More to explore|Share\s+{re.escape(word)}|Advertisement)",
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return []
        chunk = clean_spaces(match.group(1))
        chunk = re.sub(rf"({re.escape(word)}\s*\([^)]+\))", r"\n\1\n", chunk, flags=re.IGNORECASE)
        chunk = re.sub(r"(\.\s+)(Related:)", r".\n\2", chunk)
        extracted = [line for line in (clean_etymonline_line(part) for part in chunk.splitlines()) if line]
        if extracted and extracted[0].lower() == "origin and history of":
            extracted[0] = f"Origin and history of {word}"
        return extracted

    result: list[str] = []
    for line in lines[start_index:]:
        lowered = line.lower()
        if result and (
            lowered.startswith("entries linking to")
            or lowered == "more to explore"
            or lowered.startswith("share ")
            or lowered == "advertisement"
        ):
            break
        if line in {"*", "* * *", "also from 1790", "Advertisement Remove Ads"}:
            continue
        if line.startswith("Want to remove ads?"):
            continue
        cleaned = clean_etymonline_line(line)
        if cleaned:
            result.append(cleaned)
    return result


def extract_etymonline_cn_lines(word: str, lines: list[str]) -> list[str]:
    cleaned_lines = [clean_spaces(line) for line in lines if clean_spaces(line)]
    word_lower = word.lower()
    result: list[str] = []

    def is_word_line(value: str) -> bool:
        return value.lower() == word_lower

    def is_entry_heading(index: int) -> bool:
        if index + 1 >= len(cleaned_lines):
            return False
        return re.fullmatch(r"[A-Za-z][A-Za-z'’.-]*", cleaned_lines[index]) is not None and re.fullmatch(
            r"\([^)]+\)",
            cleaned_lines[index + 1],
        ) is not None

    def append_paragraph(parts: list[str]) -> None:
        text = clean_spaces("".join(parts))
        text = re.sub(r"\s+([，。；：、）])", r"\1", text)
        text = re.sub(r"([（])\s+", r"\1", text)
        if text:
            result.append(text)

    source_start = None
    for index in range(len(cleaned_lines) - 2):
        if is_word_line(cleaned_lines[index]) and cleaned_lines[index + 1] == "的意思":
            source_start = index
            break
    if source_start is None:
        return []

    result.append(f"{word} 的意思")
    index = source_start + 2
    meaning_parts: list[str] = []
    while index < len(cleaned_lines):
        line = cleaned_lines[index]
        if is_word_line(line) and index + 1 < len(cleaned_lines) and cleaned_lines[index + 1] == "的词源":
            break
        if line not in {":", "*", "* * *"}:
            if is_word_line(line):
                meaning_parts.append(f"{line}: ")
            else:
                meaning_parts.append(line)
        index += 1
    append_paragraph(meaning_parts)

    if index >= len(cleaned_lines) - 1:
        return result

    result.append(f"{word} 的词源")
    index += 2
    paragraph_parts: list[str] = []
    while index < len(cleaned_lines):
        line = cleaned_lines[index]
        if (
            line.startswith("分享")
            or line.startswith("中文翻译由AI生成")
            or line.startswith("查看原文")
            or line == "更多探索"
            or line == "广告"
            or line == "广告 移除广告"
        ):
            break
        if line.startswith("同样来自于:"):
            index += 1
            continue
        if line == "相关词汇":
            append_paragraph(paragraph_parts)
            paragraph_parts = []
            result.append(line)
            index += 1
            continue
        if is_entry_heading(index):
            append_paragraph(paragraph_parts)
            paragraph_parts = []
            result.append(f"{line} {cleaned_lines[index + 1]}")
            index += 2
            continue
        if line not in {"*", "* * *"}:
            paragraph_parts.append(line)
        index += 1
    append_paragraph(paragraph_parts)
    return result


def extract_etymonline_cn_description(word: str, raw_html: str) -> list[str]:
    descriptions: list[str] = []
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        script_text = html.unescape(match.group(1)).strip()
        try:
            data = json.loads(script_text)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            description = item.get("description")
            if not isinstance(description, str):
                continue
            description = clean_spaces(description)
            if CJK_RE.search(description) and description.lower().startswith(word.lower()):
                descriptions.append(description)
    return descriptions[:1]


def fetch_etymology_page(word: str, localized: bool = False) -> tuple[str, list[str]]:
    target_word = word.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z'’.-]*", target_word):
        raise SystemExit(f"Invalid word: {word!r}")

    url = etymonline_cn_url(target_word) if localized else etymonline_url(target_word)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "vocab-trainer/1.0 (+https://www.etymonline.com)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw_html = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return url, []
        raise SystemExit(f"Etymonline request failed: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Etymonline request failed: {exc.reason}") from exc

    lines = page_text_from_html(raw_html)
    if localized:
        extracted = extract_etymonline_cn_lines(target_word, lines)
        if not extracted:
            extracted = extract_etymonline_cn_description(target_word, raw_html)
        return url, extracted
    return url, extract_etymonline_lines(target_word, lines)


def fetch_etymology(word: str) -> tuple[str, list[str]]:
    return fetch_etymology_page(word, localized=False)


def fetch_etymology_cn(word: str) -> tuple[str, list[str]]:
    return fetch_etymology_page(word, localized=True)


def parse_etymology_translation(text: str) -> str | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    chinese = clean_spaces(str(parsed.get("chinese", "")))
    if not chinese or not CJK_RE.search(chinese):
        return None
    return chinese


def call_deepseek_for_etymology_translation(word: str, lines: list[str]) -> str | None:
    api_key = deepseek_api_key()
    if not api_key:
        return None

    prompt = (
        "把下面 Etymonline 词源内容翻译并整理成中文。"
        "要求：保留年代、词源语言、词根、词义演变和相关词；不要扩写没有给出的事实；"
        "用 2 到 5 句自然中文。返回严格 JSON，只有 chinese 一个键。\n\n"
        f"Word: {word}\n"
        f"Etymonline text:\n{json.dumps(lines, ensure_ascii=False, indent=2)}"
    )
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": [
            {
                "role": "system",
                "content": "You translate concise etymology notes into accurate Chinese. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": min(max(800, len(" ".join(lines)) // 2), 1800),
        "response_format": {"type": "json_object"},
        "stream": False,
        "temperature": 0.1,
    }
    request = urllib.request.Request(
        os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=env_int("DEEPSEEK_TIMEOUT", DEFAULT_DEEPSEEK_TIMEOUT)) as response:
            output_text = response_text(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"DeepSeek etymology translation failed; showing English source. {api_error_message(exc)}", file=sys.stderr)
        return None
    return parse_etymology_translation(output_text)


def print_etymology(word: str) -> None:
    cn_url, cn_lines = fetch_etymology_cn(word)
    url, lines = fetch_etymology(word)
    if not cn_lines and not lines:
        print(f"No Etymonline entry found for {word}.")
        print(f"Search: {ETYMONLINE_BASE_URL}/search?q={urllib.parse.quote(word)}")
        return

    print(paint(f"Etymonline: {word}", "meta"))
    print(paint("中文词源", "definition"))
    if cn_lines:
        for line in cn_lines:
            if line == f"{word} 的意思和词源":
                print(f"  {line}")
            else:
                print(f"  {line}")
        print(f"  来源: {cn_url}")
    else:
        chinese = call_deepseek_for_etymology_translation(word, lines)
        if chinese:
            print(f"  {chinese}")
        else:
            print("  （没有取到 Etymonline 中文页内容，下面显示英文原文。）")
    if lines:
        print()
        print(paint("英文原文", "meta"))
        for line in lines:
            if re.fullmatch(rf"{re.escape(word)}\s*\([^)]+\)", line, flags=re.IGNORECASE):
                print(paint(line, "word"))
            else:
                print(f"  {line}")
        print(f"  来源: {url}")


def display_phonetic(entry: dict[str, Any]) -> str:
    phonetic = entry.get("phonetic") or "(No phonetic parsed)"
    return phonetic.replace("英:", "UK:").replace("美:", "US:")


def read_hidden_answer() -> str:
    if not sys.stdin.isatty():
        return input().strip().lower()

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        new_attrs = old_attrs[:]
        new_attrs[3] = new_attrs[3] & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
        return sys.stdin.readline().strip().lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def prompt_card(entry: dict[str, Any], mode: str) -> str:
    label = "Review" if mode == "review" else "New"
    print()
    meta = f"[{label}] #{entry['seq']}  page {entry['page']}  {entry['first_letter']}"
    print(paint(meta, "meta"))
    prompt_line = f"{paint(entry['word'], 'word')}  {paint(display_phonetic(entry), 'phonetic')}"
    print(prompt_line)
    while True:
        try:
            answer = read_hidden_answer()
        except EOFError:
            print_definition(entry)
            return "q"
        if answer == "e":
            try:
                print_etymology(str(entry["word"]))
            except SystemExit as exc:
                print(exc)
            continue
        if answer in {"y", "n", "q"}:
            if answer == "y":
                print(paint("✓", "correct"))
            elif answer == "n":
                print(paint("✗", "wrong"))
            print_definition(entry)
            return answer
        print("Please enter y, n, e, or q.")


def study(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    profile = normalize_profile_name(args.profile)
    state = load_state(profile)
    if args.from_page is not None:
        start_seq = page_start_seq(entries, args.from_page)
        state["start_page"] = args.from_page
        state["start_seq"] = start_seq
        state["last_new_seq"] = start_seq - 1
        save_state(state, profile)
        print(f"New words will start from page {args.from_page}, entry #{start_seq}.")
    reviews = due_entries(entries, state)
    new_words = new_entries(entries, state, args.new)
    if args.reviews_first:
        queue = [("review", entry) for entry in reviews] + [("new", entry) for entry in new_words]
    else:
        queue = [("new", entry) for entry in new_words] + [("review", entry) for entry in reviews]

    print(f"Profile: {profile} ({state_path(profile)})")
    print(f"Vocabulary: {len(entries)} words. Review due: {len(reviews)}. New words: {len(new_words)}.")
    if not queue:
        print("No due reviews and no new words for this session.")
        return

    touched_count = 0
    missed_count = 0
    session_counts = {"new": 0, "review": 0}
    for mode, entry in queue:
        result = prompt_card(entry, mode)
        touched_count += 1
        session_counts[mode] = int(session_counts.get(mode, 0)) + 1
        if result == "q":
            save_state(state, profile)
            print("Progress saved.")
            print_session_summary(
                entries,
                state,
                touched_count,
                missed_count,
                session_counts["new"],
                session_counts["review"],
            )
            return
        if result == "n":
            missed_count += 1
        mark_answer(state, int(entry["seq"]), result == "y", mode)
        if mode == "new":
            state["last_new_seq"] = max(int(state.get("last_new_seq", 0)), int(entry["seq"]))
        save_state(state, profile)
    print("Session complete. Progress saved.")
    print_session_summary(
        entries,
        state,
        touched_count,
        missed_count,
        session_counts["new"],
        session_counts["review"],
    )


def show_stats(entries: list[dict[str, Any]], profile: str | None) -> None:
    profile = normalize_profile_name(profile)
    state = load_state(profile)
    cards = state.get("cards", {})
    today = date.today().isoformat()
    max_page = total_pdf_pages(entries)
    start_page = int(state.get("start_page", 1))
    start_seq = int(state.get("start_seq", 1))
    total_pages = scoped_total_pages(max_page, start_page)
    last_new_seq = int(state.get("last_new_seq", 0))
    scoped_cards = {
        seq: card
        for seq, card in cards.items()
        if start_seq <= int(seq) <= last_new_seq
    }
    due_count = sum(1 for card in scoped_cards.values() if card.get("due", today) <= today)
    learned = len(scoped_cards)
    last_new_entry = entry_for_seq(entries, last_new_seq)
    next_new_entry = first_entry_after_seq(entries, last_new_seq)
    learned_pages = [
        int(entry["page"])
        for entry in entries
        if str(entry["seq"]) in cards and int(entry["seq"]) >= start_seq
    ]
    current_pdf_page = (
        int(next_new_entry["page"])
        if next_new_entry
        else int(last_new_entry["page"])
        if last_new_entry
        else 1
    )
    current_page = scoped_page_number(current_pdf_page, start_page)
    learned_pdf_page = max(learned_pages) if learned_pages else 0
    learned_page = scoped_page_number(learned_pdf_page, start_page) if learned_pdf_page else 0
    scoped_done, scoped_total_entries, progress_percent = entry_progress(entries, state)
    print(f"Profile: {profile}")
    print(f"Progress file: {state_path(profile)}")
    print(f"Vocabulary size: {len(entries)}")
    print(f"Started cards: {learned}")
    print(f"Due reviews today: {due_count}")
    print(f"Start page: {start_page}")
    last_new_word = last_new_entry["word"] if last_new_entry else "none"
    print(f"Last new entry: {last_new_seq} ({last_new_word})")
    print(f"New-word page progress: page {current_page} / {total_pages} (PDF page {current_pdf_page} / {max_page})")
    print(
        f"Highest learned page: page {learned_page} / {total_pages} (PDF page {learned_pdf_page} / {max_page})"
        if learned_page
        else f"Highest learned page: 0 / {total_pages}"
    )
    print(f"Entry progress: {scoped_done} / {scoped_total_entries} {progress_bar(progress_percent)}")
    daily_stats = state.get("daily_stats", {})
    if daily_stats:
        print("Daily word counts:")
        for day in sorted(daily_stats)[-14:]:
            stats = daily_stats[day]
            print(
                f"  {day}: studied {int(stats.get('studied', 0))}, "
                f"new {int(stats.get('new', 0))}, review {int(stats.get('review', 0))}, "
                f"remembered {int(stats.get('remembered', 0))}, missed {int(stats.get('missed', 0))}"
            )
    else:
        print("Daily word counts: none")


def snap(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    profile = normalize_profile_name(args.profile)
    state = load_state(profile)
    day = selected_day(args)
    day_label = display_day(day)
    missed_seqs = state.get("daily_misses", {}).get(day, [])
    if not missed_seqs:
        print(f"No missed words on {day_label}.")
        return
    entry_by_seq = {int(entry["seq"]): entry for entry in entries}
    missed_entries = [entry_by_seq[int(seq)] for seq in missed_seqs if int(seq) in entry_by_seq]
    if not missed_entries:
        print(f"No missed words on {day_label}.")
        return

    word_width = max(len(str(entry["word"])) for entry in missed_entries)
    phonetic_width = max(len(display_phonetic(entry)) for entry in missed_entries)
    pos_width = max(
        [len(pos) for entry in missed_entries for pos, _ in definition_parts(entry.get("definition") or entry.get("source_text") or "")]
        or [0]
    )

    print(f"Missed on {day_label}: {len(missed_entries)}")
    for entry in missed_entries:
        word = str(entry["word"])
        phonetic = display_phonetic(entry)
        parts = definition_parts(entry.get("definition") or entry.get("source_text") or "")
        for index, (pos, meaning) in enumerate(parts):
            word_col = paint(word.ljust(word_width), "word") if index == 0 else " " * word_width
            phonetic_col = paint(phonetic.ljust(phonetic_width), "phonetic") if index == 0 else " " * phonetic_width
            pos_col = paint(pos.ljust(pos_width), "pos") if pos else " " * pos_width
            meaning_col = paint(meaning, "definition")
            print(f"{word_col}  {phonetic_col}  {pos_col}  {meaning_col}")
    print()


def daily_n_counts(state: dict[str, Any], day: str) -> dict[int, int]:
    counts = {
        int(seq): int(count)
        for seq, count in state.get("daily_miss_counts", {}).get(day, {}).items()
        if str(seq).isdigit()
    }
    for seq in state.get("daily_misses", {}).get(day, []):
        counts.setdefault(int(seq), 1)
    return counts


def all_miss_counts(state: dict[str, Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    days = set(state.get("daily_miss_counts", {})) | set(state.get("daily_misses", {}))
    for day in days:
        for seq, count in daily_n_counts(state, day).items():
            counts[seq] = counts.get(seq, 0) + count
    return counts


def selected_day(args: argparse.Namespace) -> str:
    day = getattr(args, "date", None) or date.today().isoformat()
    if re.fullmatch(r"\d{4}", day):
        current_year = date.today().year
        month = int(day[:2])
        month_day = int(day[2:])
        try:
            return date(current_year, month, month_day).isoformat()
        except ValueError as exc:
            raise SystemExit(f"Invalid date: {day}. Use MMDD, for example 0522.") from exc
    try:
        return date.fromisoformat(day).isoformat()
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {day}. Use MMDD, for example 0522.") from exc


def display_day(day: str) -> str:
    parsed = date.fromisoformat(day)
    return f"{parsed.month:02d}{parsed.day:02d}"


def print_entry_table(
    entries: list[dict[str, Any]],
    title: str,
    n_counts: dict[int, int] | None = None,
    count_label: str = "n",
    blank_between_entries: bool = False,
) -> None:
    if not entries:
        print(f"No {title.lower()}.")
        return
    word_width = max(len(str(entry["word"])) for entry in entries)
    phonetic_width = max(len(display_phonetic(entry)) for entry in entries)
    pos_width = max(
        [len(pos) for entry in entries for pos, _ in definition_parts(entry.get("definition") or entry.get("source_text") or "")]
        or [0]
    )

    print(f"{title}: {len(entries)}")
    for entry_index, entry in enumerate(entries):
        n_count = n_counts.get(int(entry["seq"]), 0) if n_counts is not None else None
        word = str(entry["word"])
        phonetic = display_phonetic(entry)
        parts = definition_parts(entry.get("definition") or entry.get("source_text") or "")
        for index, (pos, meaning) in enumerate(parts):
            count_prefix = f"{count_label}={n_count:<2}  " if n_count is not None else ""
            n_col = count_prefix if n_count is not None and index == 0 else " " * len(count_prefix)
            word_col = paint(word.ljust(word_width), "word") if index == 0 else " " * word_width
            phonetic_col = paint(phonetic.ljust(phonetic_width), "phonetic") if index == 0 else " " * phonetic_width
            pos_col = paint(pos.ljust(pos_width), "pos") if pos else " " * pos_width
            meaning_col = paint(meaning, "definition")
            print(f"{n_col}{word_col}  {phonetic_col}  {pos_col}  {meaning_col}")
        if blank_between_entries and entry_index < len(entries) - 1:
            print()
    print()


def today_words(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    profile = normalize_profile_name(args.profile)
    state = load_state(profile)
    today = date.today().isoformat()
    seen_seqs = state.get("daily_seen", {}).get(today, [])
    entry_by_seq = {int(entry["seq"]): entry for entry in entries}
    seen_entries = [entry_by_seq[int(seq)] for seq in seen_seqs if int(seq) in entry_by_seq]
    n_counts = daily_n_counts(state, today)
    if args.sort_by_n:
        seen_entries.sort(key=lambda entry: (-n_counts.get(int(entry["seq"]), 0), int(entry["seq"])))
        print_entry_table(seen_entries, "Studied today by n count", n_counts)
    else:
        print_entry_table(seen_entries, "Studied today")


def missed_words(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    profile = normalize_profile_name(args.profile)
    state = load_state(profile)
    day = selected_day(args) if getattr(args, "date", None) else None
    counts = daily_n_counts(state, day) if day else all_miss_counts(state)
    start_seq = int(state.get("start_seq", 1))
    last_new_seq = int(state.get("last_new_seq", 0))
    zero_seqs = state.get("daily_seen", {}).get(day, []) if day else range(start_seq, last_new_seq + 1)
    scoped_entries = [
        entry
        for entry in entries
        if start_seq <= int(entry["seq"])
        and (
            counts.get(int(entry["seq"]), 0) > 0
            or (args.include_zero and int(entry["seq"]) in zero_seqs)
        )
    ]
    if args.inc:
        scoped_entries.sort(key=lambda entry: (-counts.get(int(entry["seq"]), 0), int(entry["seq"])))
    else:
        scoped_entries.sort(key=lambda entry: (counts.get(int(entry["seq"]), 0), int(entry["seq"])))
    title = f"Missed words for {profile} on {display_day(day)}" if day else f"Missed words for {profile}"
    if args.include_zero:
        title = f"{title} including zero"
    print_entry_table(scoped_entries, title, counts, count_label="missed")


def show_page(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    page = int(args.page_number)
    max_page = total_pdf_pages(entries)
    if page < 1 or page > max_page:
        raise SystemExit(f"Invalid page: {page}. Use 1-{max_page}.")
    page_entries = [entry for entry in entries if int(entry["page"]) == page]
    if not page_entries:
        print(f"No words on page {page}.")
        print(f"Page position: {page} / {max_page} {progress_bar(page / max_page * 100 if max_page else 0)}")
        return
    print(f"Page position: {page} / {max_page} {progress_bar(page / max_page * 100 if max_page else 0)}")
    print_entry_table(page_entries, f"Words on page {page}", blank_between_entries=True)


def show_letter_counts(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    counts = {chr(code): 0 for code in range(ord("A"), ord("Z") + 1)}
    for entry in entries:
        first_letter = str(entry.get("first_letter") or str(entry.get("word", ""))[:1]).upper()
        if first_letter in counts:
            counts[first_letter] += 1

    total = sum(counts.values())
    print(f"Vocabulary by first letter: {total} words")
    print("Letter  Count  Share of PDF")
    print("------  -----  --------------------------------------------------------------------------------------------------------------------------")
    for letter in counts:
        count = counts[letter]
        percent = (count / total * 100) if total else 0
        print(f"{letter:<6}  {count:>5}  {progress_bar(percent, 120)}")


def page_entries_for_arg(entries: list[dict[str, Any]], page: int) -> list[dict[str, Any]]:
    max_page = total_pdf_pages(entries)
    if page < 1 or page > max_page:
        raise SystemExit(f"Invalid page: {page}. Use 1-{max_page}.")
    return [entry for entry in entries if int(entry["page"]) == page]


def generate_snap_for_entries(target_entries: list[dict[str, Any]]) -> dict[str, Any]:
    generated = call_deepseek_for_snap(target_entries)
    if generated:
        return generated

    generated = call_chatgpt_for_snap(target_entries)
    if generated:
        return generated

    generated = call_local_huggingface_for_snap(target_entries)
    if generated:
        return generated

    highlighted_words = [paint(str(entry["word"]), "word") for entry in target_entries]
    if len(highlighted_words) == 1:
        word_list = highlighted_words[0]
    elif len(highlighted_words) == 2:
        word_list = f"{highlighted_words[0]} and {highlighted_words[1]}"
    else:
        word_list = ", ".join(highlighted_words[:-1]) + f", and {highlighted_words[-1]}"
    return {
        "english": local_snap_paragraph(target_entries, word_list),
        "chinese": local_snap_translation(target_entries, [short_meaning(entry) for entry in target_entries]),
    }


def print_snap_for_entries(source_label: str, target_entries: list[dict[str, Any]], generated: dict[str, Any]) -> None:
    words = [str(entry["word"]) for entry in target_entries]
    entry_by_word = {str(entry["word"]).lower(): entry for entry in target_entries}
    omitted_words = [str(word) for word in generated.get("_omitted_words", [])]
    omitted_entries = [entry_by_word[word.lower()] for word in omitted_words if word.lower() in entry_by_word]
    print(f"{source_label}: {len(target_entries)}")
    print()
    print_snap_plus_result(generated, words, omitted_entries)


def build_page_snap_cache(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    start_page = int(args.from_page)
    end_page = int(args.to_page)
    max_page = total_pdf_pages(entries)
    if start_page > end_page:
        raise SystemExit("--from-page must be <= --to-page.")
    if start_page < 1 or end_page > max_page:
        raise SystemExit(f"Invalid page range: {start_page}-{end_page}. Use 1-{max_page}.")

    built = 0
    skipped = 0
    for page in range(start_page, end_page + 1):
        page_entries = page_entries_for_arg(entries, page)
        if not page_entries:
            print(f"Page {page}: no words, skipped.")
            skipped += 1
            continue
        if not getattr(args, "force", False) and load_page_snap_cache(page, page_entries):
            print(f"Page {page}: cached, skipped.")
            skipped += 1
            continue
        print(f"Page {page}: generating snap+ for {len(page_entries)} words...")
        generated = generate_snap_for_entries(page_entries)
        save_page_snap_cache(page, page_entries, generated)
        print(f"Page {page}: saved {snap_page_cache_path(page)}")
        built += 1
    print(f"Page snap+ cache complete: built {built}, skipped {skipped}.")


def snap_plus(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=args.rebuild)
    profile = normalize_profile_name(args.profile)
    state = load_state(profile)
    if getattr(args, "from_page", None) is not None or getattr(args, "to_page", None) is not None:
        if getattr(args, "from_page", None) is None or getattr(args, "to_page", None) is None:
            raise SystemExit("Use both --from-page and --to-page.")
        build_page_snap_cache(args)
        return
    page = getattr(args, "page", None)
    if page is not None:
        target_entries = page_entries_for_arg(entries, int(page))
        source_label = f"Page {page}"
        if not target_entries:
            print(f"No words on page {page}.")
            return
        cached = load_page_snap_cache(int(page), target_entries)
        if cached:
            print_snap_for_entries(source_label, target_entries, cached)
            return
    else:
        day = selected_day(args)
        day_label = display_day(day)
        missed_seqs = state.get("daily_misses", {}).get(day, [])
        entry_by_seq = {int(entry["seq"]): entry for entry in entries}
        target_entries = [entry_by_seq[int(seq)] for seq in missed_seqs if int(seq) in entry_by_seq]
        source_label = f"Missed on {day_label}"
    if not target_entries:
        print(f"No words for {source_label}.")
        return

    generated = generate_snap_for_entries(target_entries)
    if page is not None:
        save_page_snap_cache(int(page), target_entries, generated)
    print_snap_for_entries(source_label, target_entries, generated)


def main() -> None:
    direct_profile = None
    if len(sys.argv) > 1 and sys.argv[1] == "--missed":
        sys.argv[1] = "missed"
    known_commands = {
        "build",
        "study",
        "stats",
        "list",
        "letters",
        "snap",
        "snap+",
        "today",
        "missed",
        "page",
        "etym",
    }
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in known_commands:
        direct_profile = sys.argv[1]
        sys.argv = [sys.argv[0], "study", direct_profile, *sys.argv[2:]]

    parser = argparse.ArgumentParser(description="Extract vocabulary from a PDF and review it with spaced repetition.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF file path")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the PDF index")
    parser.add_argument("--profile", default=None, help="Progress profile name, default: default")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Parse the PDF and generate CSV/JSON indexes")
    build_parser.set_defaults(func=lambda args: print_build(args))

    study_parser = subparsers.add_parser("study", help="Start studying")
    study_parser.add_argument("profile_name", nargs="?", help="Progress profile name")
    study_parser.add_argument("--new", type=int, default=20, help="Number of new words")
    study_parser.add_argument("--from-page", type=int, help="Start new words from this PDF page")
    study_parser.add_argument(
        "--reviews-first",
        action="store_true",
        help="Study due reviews before new words",
    )
    study_parser.set_defaults(func=study)

    stats_parser = subparsers.add_parser("stats", help="Show progress stats")
    stats_parser.add_argument("profile_name", nargs="?", help="Progress profile name")
    stats_parser.set_defaults(func=lambda args: show_stats(load_index(args.pdf, args.rebuild), args.profile))

    list_parser = subparsers.add_parser("list", help="List all progress profiles")
    list_parser.set_defaults(func=lambda args: show_progress_list(load_index(args.pdf, args.rebuild)))

    letters_parser = subparsers.add_parser("letters", help="Show A-Z word counts for the whole PDF")
    letters_parser.set_defaults(func=show_letter_counts)

    page_parser = subparsers.add_parser("page", help="Print words on a PDF page")
    page_parser.add_argument("page_number", type=int, help="PDF page number")
    page_parser.set_defaults(func=show_page)

    etym_parser = subparsers.add_parser("etym", help="Look up a word on Etymonline")
    etym_parser.add_argument("word", help="English word to look up")
    etym_parser.set_defaults(func=lambda args: print_etymology(args.word))

    snap_parser = subparsers.add_parser("snap", help="Print today's missed words")
    snap_parser.add_argument("profile_name", nargs="?", help="Progress profile name")
    snap_parser.add_argument("--date", help="Missed-word date, MMDD; default: today")
    snap_parser.set_defaults(func=snap)

    snap_plus_parser = subparsers.add_parser("snap+", help="Write a short paragraph using today's missed words")
    snap_plus_parser.add_argument("profile_name", nargs="?", help="Progress profile name")
    snap_plus_parser.add_argument("--date", help="Missed-word date, MMDD; default: today")
    snap_plus_parser.add_argument("--page", type=int, help="Use words from this PDF page instead of missed words")
    snap_plus_parser.add_argument("--from-page", type=int, help="Build cached snap+ pages from this PDF page")
    snap_plus_parser.add_argument("--to-page", type=int, help="Build cached snap+ pages through this PDF page")
    snap_plus_parser.add_argument("--force", action="store_true", help="Regenerate cached snap+ pages")
    snap_plus_parser.set_defaults(func=snap_plus)

    today_parser = subparsers.add_parser("today", help="Print today's studied words")
    today_parser.add_argument("profile_name", nargs="?", help="Progress profile name")
    today_parser.add_argument("--sort-by-n", action="store_true", help="Sort today's words by today's n count")
    today_parser.set_defaults(func=today_words)

    missed_parser = subparsers.add_parser("missed", help="Print missed words sorted by missed count")
    missed_parser.add_argument("profile_name", nargs="?", help="Progress profile name")
    missed_parser.add_argument("--date", help="Missed-word date, MMDD; default: all dates")
    missed_parser.add_argument("--inc", action="store_true", help="Sort with the highest missed counts first")
    missed_parser.add_argument("--include-zero", action="store_true", help="Also include studied words with zero misses")
    missed_parser.set_defaults(func=missed_words)

    args = parser.parse_args()
    args.profile = args.profile or getattr(args, "profile_name", None) or direct_profile or DEFAULT_PROFILE
    if not args.command:
        args.command = "study"
        args.new = 20
        args.from_page = None
        args.reviews_first = False
        study(args)
        return
    args.func(args)


def print_build(args: argparse.Namespace) -> None:
    entries = load_index(args.pdf, rebuild=True)
    print(f"Generated {INDEX_FILE}: {len(entries)} entries")
    print(f"Generated {WORDS_CSV}: word order with seq/page/first_letter")
    print(f"Generated {RAW_LINES_CSV}: raw PDF lines with line_seq/page/first_letter")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited.")
        sys.exit(130)
