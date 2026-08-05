from __future__ import annotations

import os
import subprocess
import sys

from console_theme import Console, ConsoleTheme, highlight_chinese_terms, highlight_english_terms, strip_ansi


def test_color_enabled_semantic_output() -> None:
    theme = ConsoleTheme(enabled=True)
    assert "\033[" in theme.label("PLAN")
    assert "[PLAN]" in strip_ansi(theme.label("PLAN"))


def test_color_disabled_output() -> None:
    theme = ConsoleTheme(enabled=False)
    assert theme.label("PLAN") == "[PLAN]"


def test_no_color_environment(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert ConsoleTheme().enabled is False


def test_force_color_environment(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert ConsoleTheme().enabled is True


def test_non_tty_fallback(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert ConsoleTheme().enabled is False


def test_english_target_highlighting() -> None:
    theme = ConsoleTheme(enabled=True)
    rendered = highlight_english_terms("The policy made officials abhor waste.", ["abhor"], theme)
    assert "\033[" in rendered
    assert "abhor" in strip_ansi(rendered)


def test_inflected_english_target_highlighting() -> None:
    theme = ConsoleTheme(enabled=True)
    rendered = highlight_english_terms("The committee abhorred waste while abhorring delay.", ["abhor"], theme)
    plain = strip_ansi(rendered)
    assert "abhorred" in plain
    assert "abhorring" in plain
    assert rendered.count("\033[") >= 2


def test_multiple_english_targets() -> None:
    theme = ConsoleTheme(enabled=True)
    rendered = highlight_english_terms("Granular data can redeem the model.", ["granular", "redeem"], theme)
    assert rendered.count("\033[") >= 2


def test_chinese_translation_highlighting() -> None:
    theme = ConsoleTheme(enabled=True)
    rendered = highlight_chinese_terms("他痛恨浪费，也憎恶拖延。", {"abhor": ["痛恨", "憎恶"]}, theme)
    assert rendered.count("\033[") >= 2
    assert strip_ansi(rendered) == "他痛恨浪费，也憎恶拖延。"


def test_missing_chinese_match_no_insert() -> None:
    theme = ConsoleTheme(enabled=True)
    rendered = highlight_chinese_terms("他反对浪费。", {"abhor": ["痛恨"]}, theme)
    assert rendered == "他反对浪费。"


def test_no_color_cli_disables_ansi() -> None:
    result = subprocess.run(
        [sys.executable, "lexipilot.py", "--profile", "alice", "--no-color"],
        input="/exit\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "\033[" not in result.stdout


def test_tool_line_is_dim(capsys) -> None:
    console = Console(ConsoleTheme(enabled=True))
    console.tool("save_session_summary")
    output = capsys.readouterr().out
    assert output.startswith("\033[2m[TOOL] save_session_summary\033[0m")
    assert strip_ansi(output).strip() == "[TOOL] save_session_summary"


def test_profile_status_multiline_progress(capsys) -> None:
    console = Console(ConsoleTheme(enabled=False))
    console.profile_status(
        {
            "profile": "default",
            "total_vocabulary_count": 2000,
            "started_word_count": 500,
            "reviews_due_today": 120,
            "total_incorrect_answers": 42,
            "current_new_word_position": 900,
            "recent_study_statistics": {"2026-08-01": {}, "2026-08-02": {}},
        }
    )
    output = capsys.readouterr().out
    assert "[STATUS]\n" in output
    assert "Profile: default" in output
    assert "Started words: 500 / 2000" in output
    assert "Progress: [#######---------------------] 25.0%" in output
    assert "Due today: 120" in output
    assert "Recent activity: 2 days" in output
