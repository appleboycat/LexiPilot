#!/usr/bin/env python3
"""Validate a LexiPilot vocabulary index without printing vocabulary content."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "seq",
    "word",
    "first_letter",
    "page",
    "phonetic",
    "definition",
}
PRIVATE_FIELDS = {
    "api_key",
    "authorization",
    "cards",
    "correct",
    "daily_misses",
    "daily_stats",
    "due",
    "missed",
    "progress",
    "seen",
    "stage",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CREDENTIAL_RE = re.compile(r"(?:Bearer\s+[A-Za-z0-9._-]+|sk-[A-Za-z0-9_-]{16,})")


class VocabularyIndexError(ValueError):
    pass


def validate_vocab_index(path: Path | str) -> dict[str, int]:
    index_path = Path(path)
    try:
        text = index_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise VocabularyIndexError("file is not valid UTF-8") from exc
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VocabularyIndexError("file is not valid JSON") from exc
    if not isinstance(entries, list):
        raise VocabularyIndexError("top-level JSON value must be a list")
    if not entries:
        raise VocabularyIndexError("index must contain at least one entry")
    if ANSI_RE.search(text):
        raise VocabularyIndexError("ANSI terminal codes are not allowed")
    if CREDENTIAL_RE.search(text):
        raise VocabularyIndexError("credential-like data is not allowed")

    words: set[str] = set()
    sequences: set[int] = set()
    pages: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise VocabularyIndexError("every entry must be a JSON object")
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise VocabularyIndexError("one or more entries are missing required fields")
        if {str(key).lower() for key in entry} & PRIVATE_FIELDS:
            raise VocabularyIndexError("learner-progress fields are not allowed")
        seq = entry["seq"]
        page = entry["page"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise VocabularyIndexError("sequence values must be positive integers")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise VocabularyIndexError("page values must be positive integers")
        word = entry["word"]
        if not isinstance(word, str) or not word.strip():
            raise VocabularyIndexError("word values must be non-empty strings")
        for field in ("first_letter", "phonetic", "definition"):
            if not isinstance(entry[field], str):
                raise VocabularyIndexError(f"{field} values must be strings")
        key = word.strip().lower()
        if key in words:
            raise VocabularyIndexError("duplicate word values are not allowed")
        if seq in sequences:
            raise VocabularyIndexError("duplicate sequence values are not allowed")
        words.add(key)
        sequences.add(seq)
        pages.add(page)

    return {
        "entries": len(entries),
        "first_sequence": min(sequences),
        "last_sequence": max(sequences),
        "unique_words": len(words),
        "pages": len(pages),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a LexiPilot vocabulary index.")
    parser.add_argument("index_file", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_vocab_index(args.index_file)
    except (OSError, VocabularyIndexError) as exc:
        print(f"Validation: FAIL - {exc}")
        return 1
    print(f"Entries: {summary['entries']}")
    print(f"First sequence: {summary['first_sequence']}")
    print(f"Last sequence: {summary['last_sequence']}")
    print(f"Unique words: {summary['unique_words']}")
    print(f"Pages: {summary['pages']}")
    print("Validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
