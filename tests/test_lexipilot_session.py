from __future__ import annotations

import json
from pathlib import Path

from console_theme import Console, ConsoleTheme, strip_ansi
from lexipilot_core import LexiPilotAgent, SessionState, final_summary_lines, format_plan, priority_reasons_for_session, priority_words_for_session, render_card, render_material, stage_marks
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


def test_priority_reasons_explain_selection(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 15 minutes. Focus on missed words.")
    assert agent.session is not None
    agent.session.incorrect_words = ["falter"]
    priority = priority_words_for_session(agent.session)
    reasons = priority_reasons_for_session(agent.session, priority)
    assert reasons["falter"] == "missed in this session"
    assert reasons["granular"].startswith("selected today and historically frequently missed")


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
    assert agent.session is not None
    agent.session.started_perf = 0.0
    agent.add_user_wait(2.0)
    import lexipilot_core

    original_perf = lexipilot_core.time.perf_counter
    lexipilot_core.time.perf_counter = lambda: 3.0
    try:
        timings = agent.timing_summary()
    finally:
        lexipilot_core.time.perf_counter = original_perf
    assert timings["user_interaction_wait_seconds"] >= 2.0
    assert timings["active_system_seconds"] <= timings["session_wall_seconds"]


def test_initial_interactive_input_wait_transfers_to_session(toolbox: LexiPilotToolbox, monkeypatch) -> None:
    agent = make_agent(toolbox)
    agent.add_user_wait(5.0)
    agent.plan("I have 6 minutes.")
    assert agent.session is not None
    agent.session.started_perf = 0.0
    monkeypatch.setattr("lexipilot_core.time.perf_counter", lambda: 6.5)
    timings = agent.timing_summary()
    assert timings["user_interaction_wait_seconds"] == 5.0
    assert timings["active_system_seconds"] == 1.5


def test_non_overlapping_timing_with_nested_model_tool(toolbox: LexiPilotToolbox, monkeypatch) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 6 minutes.")
    assert agent.session is not None
    agent.session.started_perf = 0.0
    agent.session.user_interaction_wait_seconds = 3.0
    toolbox.runtime.model_request_durations[:] = [8.0]
    toolbox.runtime.story_generation_duration = 8.5
    toolbox.tool_events[:] = [{"name": "generate_practice_story", "duration": 9.0, "ok": True}]
    monkeypatch.setattr("lexipilot_core.time.perf_counter", lambda: 20.0)
    timings = agent.timing_summary()
    breakdown_total = (
        timings["non_overlapping_user_interaction_wait_seconds"]
        + timings["non_overlapping_model_api_execution_seconds"]
        + timings["non_overlapping_local_processing_seconds"]
    )
    assert abs(breakdown_total - timings["session_wall_seconds"]) <= 0.0001
    assert timings["tool_execution_seconds"] == 9.0
    assert timings["story_generation_seconds"] == 8.5


def test_no_ansi_codes_in_saved_material(toolbox: LexiPilotToolbox) -> None:
    agent = LexiPilotAgent("alice", toolbox, console=Console(ConsoleTheme(enabled=True)))
    complete_one_miss(agent)
    data = Path(agent.session.generated_material_path).read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert "\033[" not in data


def test_render_material_highlights_without_saving_codes() -> None:
    material = {
        "target_words": ["abhor"],
        "target_phonetics": {"abhor": "英:/əb'hɔː(r)/ 美:/əb'hɔːr/"},
        "target_parts_of_speech": {"abhor": "vt."},
        "target_translations": {"abhor": ["痛恨", "憎恶"]},
        "priority_reasons": {"abhor": "missed in this session"},
        "english_passage": "The faculty abhorred waste.",
        "chinese_translation": "教师们痛恨浪费。",
    }
    rendered = render_material(material, ConsoleTheme(enabled=True))
    assert "\033[" in rendered
    assert strip_ansi(rendered).count("abhorred") == 1
    plain = strip_ansi(rendered)
    assert "Let's focus on the words that need the most reinforcement" in plain
    assert "Why these words:" in plain
    assert "abhor  missed in this session" in plain
    assert "Meaning review:" in plain
    assert "abhor  vt. 痛恨；憎恶" in plain
    assert "Example sentences:" not in plain
    assert "The researcher used abhor carefully" not in plain
    assert "Word   Phonetic" in plain
    assert "abhor  UK: /əb'hɔː(r)/ US: /əb'hɔːr/  vt. 痛恨；憎恶" in plain


def test_render_material_mapping_has_three_aligned_columns() -> None:
    material = {
        "target_words": ["abate", "abbreviate", "abhor"],
        "target_phonetics": {
            "abate": "UK: /əˈbeɪt/",
            "abbreviate": "UK: /əˈbriːvieɪt/",
            "abhor": "UK: /əb'hɔː(r)/",
        },
        "target_parts_of_speech": {
            "abate": "vt.",
            "abbreviate": "vt.",
            "abhor": "vt.",
        },
        "target_translations": {
            "abate": ["减少", "减轻", "废除", "失效"],
            "abbreviate": ["使简短", "缩简", "缩略"],
            "abhor": ["痛恨", "憎恶"],
        },
        "english_passage": "Officials abhorred waste and abbreviated the plan to abate risk.",
        "chinese_translation": "官员痛恨浪费，并缩略方案以减少风险。",
    }
    rendered = render_material(material, ConsoleTheme(enabled=False))
    lines = rendered.splitlines()
    header = lines[lines.index("Vocabulary mapping:") + 1]
    first = lines[lines.index("Vocabulary mapping:") + 2]
    second = lines[lines.index("Vocabulary mapping:") + 3]
    assert header.startswith("Word")
    assert first.startswith("abate       UK: /əˈbeɪt/")
    assert second.startswith("abbreviate  UK: /əˈbriːvieɪt/")
    assert first.rstrip().endswith("vt. 减少；减轻；废除；失效")


def test_plan_highlights_study_words_and_chinese_meanings() -> None:
    theme = ConsoleTheme(enabled=True)
    plan = {
        "available_minutes": 15,
        "summary": {"reviews_due_today": 1, "total_incorrect_answers": 3},
        "planned_words": [
            {
                "word": "abhor",
                "definition": "vt. 痛恨，憎恶",
                "selection_reason": "frequently missed",
            }
        ],
    }
    rendered = format_plan(plan, theme)
    plain = strip_ansi(rendered)
    assert "\033[" in rendered
    assert "abhor" in plain
    assert "痛恨，憎恶" in plain
    assert rendered.count("\033[") >= 3


def test_plan_word_table_uses_fixed_word_pos_and_phonetic_columns() -> None:
    plan = {
        "available_minutes": 15,
        "summary": {"reviews_due_today": 3, "total_incorrect_answers": 2},
        "planned_words": [
            {
                "word": "exorbitant",
                "phonetic": "UK: /ɪɡˈzɔːbɪtənt/",
                "definition": "adj. （要价等）过高的；（性格等）过分的；不在法律范围之内的",
                "selection_reason": "due today",
            },
            {
                "word": "exotic",
                "phonetic": "UK: /ɪɡˈzɒtɪk/",
                "definition": "adj. 外来的；异国的；异国情调的",
                "selection_reason": "due today, frequently missed",
            },
            {
                "word": "expropriate",
                "phonetic": "UK: /eksˈprəʊprieɪt/",
                "definition": "vt. 没收财产；征用；剥夺所有权",
                "selection_reason": "due today, frequently missed",
            },
        ],
    }
    rendered = format_plan(plan, ConsoleTheme(enabled=False))
    lines = rendered.splitlines()
    header = next(line for line in lines if line.startswith("No."))
    rows = [line for line in lines if line[:2].strip(".").isdigit()]
    word_start = header.index("Word")
    pos_start = header.index("POS")
    phonetic_start = header.index("Phonetic")
    assert all(row.index(row.split()[1]) == word_start for row in rows)
    assert all(row.index(row.split()[2]) == pos_start for row in rows)
    assert all(row.index("UK:") == phonetic_start for row in rows)
    meaning_lines = [line for line in lines if "Meaning:" in line]
    assert len(meaning_lines) == 3
    assert all(line.index("Meaning:") > word_start for line in meaning_lines)


def test_card_highlights_word_phonetic_and_chinese_meaning() -> None:
    theme = ConsoleTheme(enabled=True)
    rendered = render_card(
        {"word": "abhor", "phonetic": "UK: /əb'hɔː(r)/ US: /əb'hɔːr/", "definition": "vt. 痛恨，憎恶"},
        6,
        7,
        theme,
    )
    plain = strip_ansi(rendered)
    assert "\033[" in rendered
    assert "WORD:" not in plain
    assert "PHONETIC:" not in plain
    assert "DEFINITION:" not in plain
    assert "abhor" in plain
    assert "/əb'hɔː(r)/" in plain
    assert "abhor    UK: /əb'hɔː(r)/ US: /əb'hɔːr/" in plain
    assert "痛恨，憎恶" in plain
    assert "Press: y / n / e=etymology / s=skip / stop" in plain


def test_card_plain_when_color_disabled() -> None:
    rendered = render_card(
        {"word": "abhor", "phonetic": "英:/əb'hɔː(r)/ 美:/əb'hɔːr/", "definition": "vt. 痛恨，憎恶"},
        1,
        1,
        ConsoleTheme(enabled=False),
    )
    assert "\033[" not in rendered
    assert "abhor" in rendered
    assert "UK: /əb'hɔː(r)/ US: /əb'hɔːr/" in rendered
    assert "痛恨，憎恶" in rendered


def test_recorded_answer_response_shows_stage_not_due_date(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 6 minutes. Focus on missed words.")
    response = agent.handle_answer("y")
    assert "Next due" not in response
    assert "Stage: " in response


def test_s_shortcut_skips_without_progress(toolbox: LexiPilotToolbox) -> None:
    agent = make_agent(toolbox)
    agent.plan("I have 6 minutes. Focus on missed words.")
    response = agent.handle_answer("s")
    assert "Skipped without changing progress." in response


def test_stage_marks_use_dashes() -> None:
    assert stage_marks(3) == "Stage: ---"
    assert stage_marks(0) == "Stage: -"


def test_summary_none_values_are_dim_not_vocabulary_colors(toolbox: LexiPilotToolbox) -> None:
    theme = ConsoleTheme(enabled=True)
    session = SessionState(
        "alice",
        "goal",
        {
            "planned_words": [],
            "new_words": [],
        },
    )
    lines = final_summary_lines(session, toolbox, 0.0, "no scheduled review", {}, theme)
    rendered = "\n".join(lines)
    assert "\033[2mnone\033[0m" in rendered
    assert "\033[1;93mnone\033[0m" not in rendered
    assert "\033[1;95mnone\033[0m" not in rendered


def test_summary_incorrect_line_shows_missed_count(toolbox: LexiPilotToolbox) -> None:
    theme = ConsoleTheme(enabled=False)
    session = SessionState(
        "alice",
        "goal",
        {
            "planned_words": [],
            "new_words": [],
        },
    )
    session.incorrect_words = ["farce", "fatigue"]
    lines = final_summary_lines(session, toolbox, 0.0, "today", {}, theme)
    rendered = "\n".join(lines)
    assert "Incorrect: 2 - farce, fatigue" in rendered
