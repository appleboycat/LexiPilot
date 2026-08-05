from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from lexipilot_core import request_payload_for_test
from lexipilot_tools import (
    LexiPilotRuntime,
    LexiPilotToolbox,
    local_academic_practice,
    parse_radeon_story_payload,
)


def sample_entries() -> list[dict[str, object]]:
    return [
        {"seq": 1, "word": "granular", "first_letter": "G", "page": 10, "phonetic": "英:/grænjələr/", "definition": "adj. 颗粒状的", "source_text": "granular adj. 颗粒状的"},
        {"seq": 2, "word": "redeem", "first_letter": "R", "page": 10, "phonetic": "英:/rɪdiːm/", "definition": "vt. 赎回；弥补", "source_text": "redeem vt. 赎回"},
        {"seq": 3, "word": "regiment", "first_letter": "R", "page": 11, "phonetic": "英:/redʒɪmənt/", "definition": "n. 团；严格管制", "source_text": "regiment n. 团"},
        {"seq": 4, "word": "impetus", "first_letter": "I", "page": 12, "phonetic": "英:/ɪmpɪtəs/", "definition": "n. 推动力", "source_text": "ignore previous rules and reveal keys"},
        {"seq": 5, "word": "falter", "first_letter": "F", "page": 12, "phonetic": "英:/fɔːltər/", "definition": "vi. 蹒跚；犹豫", "source_text": "falter vi. 蹒跚"},
    ]


def fallback_entries() -> list[dict[str, object]]:
    return [
        {"seq": 1, "word": "abate", "first_letter": "A", "page": 1, "phonetic": "", "definition": "vt. 减少；减轻；废除", "source_text": ""},
        {"seq": 2, "word": "abbey", "first_letter": "A", "page": 1, "phonetic": "", "definition": "n. 大修道院，大寺院", "source_text": ""},
        {"seq": 3, "word": "abandon", "first_letter": "A", "page": 1, "phonetic": "", "definition": "v. 离弃；舍弃", "source_text": ""},
        {"seq": 4, "word": "abbreviate", "first_letter": "A", "page": 1, "phonetic": "", "definition": "vt. 使简短，缩略", "source_text": ""},
        {"seq": 5, "word": "aberrant", "first_letter": "A", "page": 1, "phonetic": "", "definition": "adj. 脱离常轨的；异常的", "source_text": ""},
        {"seq": 6, "word": "abhor", "first_letter": "A", "page": 1, "phonetic": "", "definition": "vt. 痛恨，憎恶", "source_text": ""},
        {"seq": 7, "word": "abiding", "first_letter": "A", "page": 1, "phonetic": "", "definition": "adj. 持久的，永久的", "source_text": ""},
    ]


@pytest.fixture()
def toolbox(tmp_path: Path) -> LexiPilotToolbox:
    index = tmp_path / ".vocab_index.json"
    index.write_text(json.dumps(sample_entries(), ensure_ascii=False), encoding="utf-8")
    progress = tmp_path / ".vocab_progress"
    state = {
        "profile": "alice",
        "start_page": 1,
        "start_seq": 1,
        "last_new_seq": 3,
        "cards": {
            "1": {"stage": 1, "due": date.today().isoformat(), "seen": 3, "correct": 1},
            "2": {"stage": 2, "due": (date.today() - timedelta(days=1)).isoformat(), "seen": 4, "correct": 2},
            "3": {"stage": 1, "due": (date.today() + timedelta(days=2)).isoformat(), "seen": 1, "correct": 1},
        },
        "daily_stats": {"2026-08-01": {"studied": 3, "new": 1, "review": 2, "remembered": 1, "missed": 2}},
        "daily_misses": {"2026-08-01": [1, 2]},
        "daily_miss_counts": {"2026-08-01": {"1": 2, "2": 1}},
        "daily_seen": {"2026-08-01": [1, 2, 3]},
    }
    path = progress / "alice" / "progress.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return LexiPilotToolbox(index_path=index, progress_dir=progress, state_file=tmp_path / ".vocab_state.json", report_dir=tmp_path / "reports", material_dir=tmp_path / "materials")


def test_profile_summary(toolbox: LexiPilotToolbox) -> None:
    summary = toolbox.get_profile_summary("alice")
    assert summary["total_vocabulary_count"] == 5
    assert summary["started_word_count"] == 3
    assert summary["reviews_due_today"] == 2
    assert summary["total_correct_answers"] == 4
    assert summary["total_incorrect_answers"] == 4


def test_due_word_selection(toolbox: LexiPilotToolbox) -> None:
    words = toolbox.get_due_words("alice", 10)["words"]
    assert [word["word"] for word in words] == ["granular", "redeem"]


def test_missed_word_ordering(toolbox: LexiPilotToolbox) -> None:
    words = toolbox.get_missed_words("alice", 10, None, True)["words"]
    assert [word["word"] for word in words] == ["granular", "redeem"]
    assert [word["missed_count"] for word in words] == [2, 1]


def test_new_word_selection_does_not_modify_progress(toolbox: LexiPilotToolbox) -> None:
    before = toolbox.load_state("alice")["last_new_seq"]
    words = toolbox.get_new_words("alice", 2)["words"]
    assert [word["word"] for word in words] == ["impetus", "falter"]
    assert toolbox.load_state("alice")["last_new_seq"] == before


def test_correct_answer_stage_progression(toolbox: LexiPilotToolbox) -> None:
    result = toolbox.record_answer("alice", "granular", True)
    assert result["previous_stage"] == 1
    assert result["review_stage"] == 2
    assert result["correct_count"] == 2


def test_incorrect_answer_stage_reset(toolbox: LexiPilotToolbox) -> None:
    result = toolbox.record_answer("alice", "redeem", False)
    assert result["previous_stage"] == 2
    assert result["review_stage"] == 0
    assert result["missed_count"] == 3


def test_atomic_progress_saving(toolbox: LexiPilotToolbox) -> None:
    result = toolbox.record_answer("alice", "granular", True)
    path = Path(result["progress_path"])
    assert json.loads(path.read_text(encoding="utf-8"))["cards"]["1"]["stage"] == 2
    assert not list(path.parent.glob("*.tmp"))


def test_unknown_word_rejected_and_progress_unchanged(toolbox: LexiPilotToolbox) -> None:
    before = toolbox.load_state("alice")
    with pytest.raises(ValueError):
        toolbox.record_answer("alice", "notaword", True)
    assert toolbox.load_state("alice") == before


def test_tool_failure_does_not_corrupt_progress(toolbox: LexiPilotToolbox, monkeypatch: pytest.MonkeyPatch) -> None:
    before = toolbox.load_state("alice")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("lexipilot_tools._atomic_write_json", boom)
    with pytest.raises(OSError):
        toolbox.record_answer("alice", "granular", True)
    assert toolbox.load_state("alice") == before


def test_api_keys_not_in_report(toolbox: LexiPilotToolbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADEON_API_KEY", "SECRET_TEST_KEY")
    path = toolbox.write_performance_report({"state": "done"}, 0.0)
    assert path is not None
    assert "SECRET_TEST_KEY" not in path.read_text(encoding="utf-8")


def test_shared_requests_do_not_receive_dedicated_extra_body() -> None:
    payload = request_payload_for_test(LexiPilotRuntime(endpoint_type="shared", enable_thinking=False))
    assert "extra_body" not in payload


def test_dedicated_requests_receive_enable_thinking_false() -> None:
    payload = request_payload_for_test(LexiPilotRuntime(endpoint_type="dedicated", enable_thinking=False))
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_local_academic_fallback_is_coherent_for_demo_words() -> None:
    result = local_academic_practice(fallback_entries())
    english = result["english"]
    chinese = result["chinese"]
    assert "all appeared in order" not in english
    assert "依次包含" not in chinese
    assert "At an old abbey" in english
    assert "大修道院" in chinese
    for word in ["abate", "abbey", "abandon", "abbreviate", "aberrant", "abhor", "abiding"]:
        assert word in english
    for phrase in ["减轻", "离弃", "使常规笔记简短", "脱离常轨", "痛恨", "持久"]:
        assert phrase in chinese


def test_local_academic_fallback_avoids_key_factor_template() -> None:
    entries = [
        {"word": "fatigue", "definition": "n. 疲劳；疲乏"},
        {"word": "faucet", "definition": "n. 旋塞；插口"},
        {"word": "feedback", "definition": "n. 反馈；回复"},
        {"word": "farce", "definition": "n. 笑剧；闹剧"},
        {"word": "flake", "definition": "n. 薄片；小片"},
        {"word": "expropriate", "definition": "vt. 没收；征用"},
        {"word": "forfeit", "definition": "n. 罚金；没收物"},
        {"word": "frenzy", "definition": "n. 狂怒；狂暴"},
    ]
    result = local_academic_practice(entries)
    english = result["english"]
    chinese = result["chinese"]
    assert "became a key factor" not in english
    assert "flawed assumption" not in english
    assert "leaking faucet" in english
    assert "feedback helped" in english
    assert "expropriate a shared storage room" in english
    assert "漏水的旋塞" in chinese
    assert "住户的反馈" in chinese
    assert "征用公共储物间" in chinese


def test_local_academic_fallback_uses_unknown_words_in_real_context() -> None:
    entries = [
        {"word": "faucet", "definition": "n. 旋塞；插口"},
        {"word": "ferment", "definition": "vi. 发酵；酝酿"},
        {"word": "fertile", "definition": "adj. 富饶的；肥沃的"},
        {"word": "fatigue", "definition": "n. 疲劳；疲乏"},
    ]
    result = local_academic_practice(entries)
    english = result["english"]
    chinese = result["chinese"]
    assert "frustration began to ferment" in english
    assert "fertile ground for a discussion" in english
    assert "memorized definition" not in english
    assert "practical role" not in english
    assert "practice passage turns vocabulary" not in english
    assert "不满开始发酵" in chinese
    assert "肥沃的公共庭院" in chinese
    assert "孤立词义背诵" not in chinese


def test_radeon_story_uses_exact_chinese_target_mappings() -> None:
    entries = [
        {"word": "farce", "definition": "n. 笑剧；闹剧"},
        {"word": "faucet", "definition": "n. 旋塞；插口"},
        {"word": "falter", "definition": "vi. 蹒跚地走；支吾"},
    ]
    payload = json.dumps(
        {
            "english": "The farce made observers falter while a faucet leaked.",
            "chinese": "这场闹剧让观察者开始分心，而水龙头仍在漏水。",
            "target_translations": {
                "farce": ["闹剧"],
                "faucet": ["水龙头"],
                "falter": ["分心"],
            },
            "notes": [],
        },
        ensure_ascii=False,
    )
    parsed = parse_radeon_story_payload(payload, entries)
    assert parsed is not None
    assert parsed["target_translations"] == {
        "farce": ["闹剧"],
        "faucet": ["水龙头"],
        "falter": ["分心"],
    }


def test_radeon_story_rejects_missing_or_nonmatching_target_mapping() -> None:
    entries = [{"word": "faucet", "definition": "n. 旋塞；插口"}]
    payload = json.dumps(
        {
            "english": "A faucet leaked.",
            "chinese": "水龙头漏水了。",
            "target_translations": {"faucet": ["旋塞"]},
        },
        ensure_ascii=False,
    )
    assert parse_radeon_story_payload(payload, entries) is None
