from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

import vocab_trainer as vt
from lexipilot import REPO_ROOT, resolve_data_paths
from lexipilot_tools import LexiPilotToolbox
from scripts.setup_demo_data import setup_demo_data
from scripts.validate_vocab_index import VocabularyIndexError, validate_vocab_index

SAMPLE_INDEX = REPO_ROOT / "examples" / "sample_vocab_index.json"


def test_sample_vocabulary_index_validation() -> None:
    summary = validate_vocab_index(SAMPLE_INDEX)
    assert summary == {
        "entries": 40,
        "first_sequence": 1,
        "last_sequence": 40,
        "unique_words": 40,
        "pages": 4,
    }


def test_sample_index_has_no_source_text_or_ansi() -> None:
    text = SAMPLE_INDEX.read_text(encoding="utf-8")
    entries = json.loads(text)
    assert all(entry.get("source_text") == "" for entry in entries)
    assert "\x1b[" not in text


def test_validator_rejects_progress_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            [
                {
                    "seq": 1,
                    "word": "safe",
                    "first_letter": "S",
                    "page": 1,
                    "phonetic": "/seɪf/",
                    "definition": "adj. 安全的",
                    "cards": {},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(VocabularyIndexError):
        validate_vocab_index(bad)


def test_setup_demo_data_creates_valid_synthetic_profile(tmp_path: Path) -> None:
    progress_root = tmp_path / "profiles"
    output = setup_demo_data(index_path=SAMPLE_INDEX, progress_root=progress_root, today=date(2026, 8, 5))
    state = json.loads(output.read_text(encoding="utf-8"))
    assert state["profile"] == "demo"
    assert state["last_new_seq"] == 12
    assert len(state["cards"]) == 12
    assert sum(card["due"] <= "2026-08-05" for card in state["cards"].values()) == 7
    assert len(state["daily_stats"]) == 35
    assert min(state["daily_stats"]) == "2026-07-02"
    assert max(state["daily_stats"]) == "2026-08-05"
    assert 20 <= sum(item["studied"] > 0 for item in state["daily_stats"].values()) < 35
    assert state["demo_data"] is True


def test_demo_activity_generation_is_reproducible(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = setup_demo_data(
        index_path=SAMPLE_INDEX,
        progress_root=first_root,
        today=date(2026, 8, 5),
    )
    second = setup_demo_data(
        index_path=SAMPLE_INDEX,
        progress_root=second_root,
        today=date(2026, 8, 5),
    )
    first_state = json.loads(first.read_text(encoding="utf-8"))
    second_state = json.loads(second.read_text(encoding="utf-8"))
    assert first_state["daily_stats"] == second_state["daily_stats"]
    assert first_state["daily_miss_counts"] == second_state["daily_miss_counts"]


def test_demo_data_setup_is_idempotent_without_force(tmp_path: Path) -> None:
    progress_root = tmp_path / "profiles"
    output = setup_demo_data(index_path=SAMPLE_INDEX, progress_root=progress_root, today=date(2026, 8, 5))
    before = output.read_bytes()
    setup_demo_data(index_path=SAMPLE_INDEX, progress_root=progress_root, today=date(2027, 1, 1))
    assert output.read_bytes() == before


def test_sample_mode_does_not_require_pdf_or_real_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_pdf = tmp_path / "missing.pdf"
    monkeypatch.setattr(vt, "DEFAULT_PDF", missing_pdf)
    progress_root = tmp_path / "profiles"
    setup_demo_data(index_path=SAMPLE_INDEX, progress_root=progress_root)
    toolbox = LexiPilotToolbox(
        index_path=SAMPLE_INDEX,
        progress_dir=progress_root,
        state_file=tmp_path / "legacy.json",
    )
    assert len(toolbox.load_entries()) == 40
    assert toolbox.get_profile_summary("demo")["started_word_count"] == 12
    assert not (progress_root / "default").exists()
    assert not missing_pdf.exists()


def test_explicit_index_and_progress_paths_resolve(tmp_path: Path) -> None:
    args = argparse.Namespace(
        profile="custom",
        demo=False,
        index_file=str(tmp_path / "index.json"),
        progress_root=str(tmp_path / "profiles"),
    )
    profile, index, progress = resolve_data_paths(args)
    assert profile == "custom"
    assert index == tmp_path / "index.json"
    assert progress == tmp_path / "profiles"


def test_demo_shortcut_uses_public_sample_paths() -> None:
    args = argparse.Namespace(profile=None, demo=True, index_file=None, progress_root=None)
    profile, index, progress = resolve_data_paths(args)
    assert profile == "demo"
    assert index == REPO_ROOT / "examples" / "sample_vocab_index.json"
    assert progress == REPO_ROOT / ".demo_data" / "profiles"


def test_fresh_clone_smoke_succeeds_without_private_data() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_fresh_clone.py"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "PASS model tools called" in result.stdout
    assert "PASS fresh-clone smoke test" in result.stdout
