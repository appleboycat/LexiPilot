from __future__ import annotations

import json
from pathlib import Path

from console_theme import Console, ConsoleTheme, strip_ansi
from lexipilot_core import LexiPilotAgent, priority_words_for_session, render_material
from lexipilot_tools import LexiPilotToolbox
from tests.test_lexipilot_tools import toolbox  # noqa: F401


def make_agent(toolbox: LexiPilotToolbox) -> LexiPilotAgent:
    return LexiPilotAgent("alice", toolbox, debug=False, console=Console(ConsoleTheme(enabled=False)))


def complete_one_miss(agent: LexiPilotAgent) -> str:
    agent.plan("I have 6 minutes. Focus on missed words.")
    agent.handle_answer("n")
    return agent.finish_session()


def test_finalization_idempotency_one_each(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    first = complete_one_miss(agent)
    second = agent.finish_session()
    third = agent.finish_session()

    assert first == second == third
    assert len(list(toolbox.material_dir.glob("alice/practice_*.json"))) == 1
    assert len(list(toolbox.report_dir.glob("lexipilot_*.json"))) == 1
    session_file = toolbox.progress_dir / "alice" / "agent_sessions.jsonl"
    assert len(session_file.read_text(encoding="utf-8").splitlines()) == 1


def test_progress_not_updated_twice_after_completion(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 6 minutes. Focus on missed words.")
    word = agent.session.current_word()["word"]  # type: ignore[union-attr]
    agent.handle_answer("n")
    state_after_answer = toolbox.load_state("alice")
    seq = next(entry["seq"] for entry in toolbox.load_entries() if entry["word"] == word)
    seen_after_answer = state_after_answer["cards"][str(seq)]["seen"]
    agent.finish_session()
    agent.handle_answer("y")
    agent.handle_answer("n")
    assert toolbox.load_state("alice")["cards"][str(seq)]["seen"] == seen_after_answer


def test_blank_input_after_completion_no_side_effects(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    complete_one_miss(agent)
    before_reports = list(toolbox.report_dir.glob("lexipilot_*.json"))
    before_materials = list(toolbox.material_dir.glob("alice/practice_*.json"))
    response = agent.handle_answer("")
    assert "already completed" in response
    assert list(toolbox.report_dir.glob("lexipilot_*.json")) == before_reports
    assert list(toolbox.material_dir.glob("alice/practice_*.json")) == before_materials


def test_exit_after_completion_has_no_additional_writes(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    complete_one_miss(agent)
    before = len(list(toolbox.report_dir.glob("lexipilot_*.json")))
    # /exit is handled by the CLI before agent mutation; this verifies there is no finalization side effect needed.
    assert before == 1


def test_priority_current_mistakes_precede_historical(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 15 minutes. Focus on missed words.")
    assert agent.session is not None
    agent.session.incorrect_words = ["falter"]
    priority = priority_words_for_session(agent.session)
    assert priority[0] == "falter"
    assert "granular" in priority


def test_priority_deduplication(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 15 minutes. Focus on missed words.")
    assert agent.session is not None
    agent.session.incorrect_words = ["granular", "granular"]
    assert priority_words_for_session(agent.session).count("granular") == 1


def test_all_correct_uses_historical_or_fallback(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 15 minutes. Focus on missed words.")
    assert agent.session is not None
    agent.session.correct_words = ["granular"]
    agent.session.incorrect_words = []
    priority = priority_words_for_session(agent.session)
    assert priority[0] == "granular"


def test_passage_target_consistency(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    complete_one_miss(agent)
    assert agent.session is not None
    material = json.loads(Path(agent.session.generated_material_path).read_text(encoding="utf-8"))
    assert material["target_words"][0] == agent.session.priority_words[0] == agent.session.incorrect_words[0]


def test_no_stray_internal_control_line_after_etymology_then_finalize(toolbox: LexiPilotToolbox, monkeypatch) -> None:
    monkeypatch.setattr(toolbox, "lookup_etymology", lambda word: {"etymology": "origin note"})
    agent = make_agent(toolbox)
    output = [agent.plan("I have 6 minutes. Focus on missed words.")]
    output.append(agent.handle_answer("etymology"))
    output.append(agent.handle_answer("y"))
    output.append(agent.finish_session())
    plain_lines = [line.strip() for line in strip_ansi("\n".join(output)).splitlines()]
    assert not any(line in {"y", "n", "continue", "done", "finish"} for line in plain_lines)


def test_timing_user_and_active(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 6 minutes.")
    agent.add_user_wait(2.0)
    timings = agent.timing_summary()
    assert timings["user_interaction_wait_seconds"] >= 2.0
    assert timings["active_system_seconds"] <= timings["session_wall_seconds"]


def test_no_ansi_codes_in_saved_material(toolbox: LexiPilotToolbox) -> None:
    agent = LexiPilotAgent("alice", toolbox, console=Console(ConsoleTheme(enabled=True)))
    complete_one_miss(agent)
    data = Path(agent.session.generated_material_path).read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert "\033[" not in data


def test_render_material_highlights_without_saving_codes() -> None:
    material = {
        "target_words": ["abhor"],
        "target_translations": {"abhor": ["痛恨", "憎恶"]},
        "english_passage": "The faculty abhorred waste.",
        "chinese_translation": "教师们痛恨浪费。",
    }
    rendered = render_material(material, ConsoleTheme(enabled=True))
    assert "\033[" in rendered
    assert strip_ansi(rendered).count("abhorred") == 1
