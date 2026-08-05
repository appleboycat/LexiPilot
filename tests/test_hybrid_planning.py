from __future__ import annotations

import json
from typing import Any

import pytest

from console_theme import Console, ConsoleTheme
from lexipilot_core import (
    LexiPilotAgent,
    ModelPlanningError,
    build_model_session_plan,
    compact_planning_tool_result,
)
from lexipilot_tools import LexiPilotRuntime, LexiPilotToolbox, planning_tool_schemas
from tests.test_lexipilot_tools import toolbox  # noqa: F401


class PlanningClient:
    def __init__(self, final_payload: dict[str, Any] | str | None = None, *, failure: Exception | None = None) -> None:
        self.calls = 0
        self.max_token_requests: list[int] = []
        self.tool_choices: list[str] = []
        self.final_messages: list[dict[str, Any]] = []
        self.failure = failure
        self.final_payload = final_payload or {
            "minutes": 6,
            "review_words": ["granular", "redeem"],
            "new_words": ["impetus"],
            "priority_words": ["granular", "redeem"],
            "selection_reason": "Due and frequently missed words come before one new word.",
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 700,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        self.calls += 1
        self.max_token_requests.append(max_tokens)
        self.tool_choices.append(tool_choice)
        if response_format is not None:
            self.final_messages = messages
        if self.failure is not None:
            raise self.failure
        if self.calls == 1:
            requested = [
                ("summary", "get_profile_summary", {"profile": "alice"}),
                ("due", "get_due_words", {"profile": "alice", "limit": 10}),
                (
                    "missed",
                    "get_missed_words",
                    {"profile": "alice", "limit": 10, "date": None, "highest_first": True},
                ),
                ("new", "get_new_words", {"profile": "alice", "limit": 2, "from_page": None}),
            ]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                                for call_id, name, arguments in requested
                            ],
                        }
                    }
                ]
            }
        content = self.final_payload if isinstance(self.final_payload, str) else json.dumps(self.final_payload)
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class ProseOnlyClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 700,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": 'get_profile_summary({"profile": "alice"})',
                    }
                }
            ]
        }


class PartialPlanningClient:
    """Model first selects summary/new, then follows a missing-tool nudge."""

    def __init__(self, *, repeat_partial: bool = False) -> None:
        self.calls = 0
        self.repeat_partial = repeat_partial
        self.messages: list[dict[str, Any]] = []
        self.requests: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 700,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        self.calls += 1
        self.messages = messages
        self.requests.append(list(messages))
        if response_format is not None:
            payload = {
                "minutes": 15,
                "review_words": ["granular", "redeem"],
                "new_words": ["impetus"],
                "priority_words": ["granular"],
                "selection_reason": "Due and frequently missed words precede one new word.",
            }
            return {"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]}

        if self.calls == 1 or self.repeat_partial:
            requested = [
                ("summary", "get_profile_summary", {"profile": "alice"}),
                ("new", "get_new_words", {"profile": "alice", "limit": 10, "from_page": None}),
            ]
        else:
            requested = [
                ("due", "get_due_words", {"profile": "alice", "limit": 10}),
                (
                    "missed",
                    "get_missed_words",
                    {"profile": "alice", "limit": 10, "date": None, "highest_first": True},
                ),
            ]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                            for call_id, name, arguments in requested
                        ],
                    }
                }
            ]
        }


def dedicated_runtime() -> LexiPilotRuntime:
    return LexiPilotRuntime(
        model_name="Qwen/Qwen3-8B",
        endpoint_type="dedicated",
        base_url="https://model.invalid/v1",
        api_key="test-key",
        performance_reports_enabled=False,
    )


def hybrid_agent(
    toolbox: LexiPilotToolbox,
    client: PlanningClient,
    *,
    debug: bool = False,
    deterministic: bool = False,
) -> LexiPilotAgent:
    toolbox.runtime = dedicated_runtime()
    return LexiPilotAgent(
        "alice",
        toolbox,
        debug=debug,
        console=Console(ConsoleTheme(enabled=False)),
        deterministic=deterministic,
        planner_client=client,
    )


def test_model_planning_is_default_for_dedicated_configuration(toolbox: LexiPilotToolbox) -> None:
    client = PlanningClient()
    agent = hybrid_agent(toolbox, client)
    agent.plan("I have 6 minutes. Review due words and add one new word.")
    assert client.calls == 2
    assert agent.session is not None
    assert agent.session.plan["planning_mode"] == "model"


def test_hybrid_planning_uses_small_two_request_budget(toolbox: LexiPilotToolbox) -> None:
    client = PlanningClient()
    agent = hybrid_agent(toolbox, client)
    agent.plan("I have 6 minutes. Review due words and add one new word.")
    assert client.max_token_requests == [420, 320]
    assert client.tool_choices == ["required", "auto"]
    assert len(client.final_messages) == 2


def test_compact_tool_context_omits_display_and_private_fields() -> None:
    compact = compact_planning_tool_result(
        "get_due_words",
        {
            "profile": "alice",
            "progress_path": "/private/progress.json",
            "words": [
                {
                    "word": "granular",
                    "phonetic": "/secret/",
                    "definition": "adj. 颗粒状的",
                    "source_text": "private source",
                    "review_stage": 1,
                    "due_date": "2026-08-06",
                    "missed_count": 2,
                }
            ],
        },
    )
    serialized = json.dumps(compact)
    assert "granular" in serialized
    assert "review_stage" in serialized
    assert "phonetic" not in serialized
    assert "definition" not in serialized
    assert "source_text" not in serialized
    assert "progress_path" not in serialized


def test_model_selected_read_only_tools_are_executed(toolbox: LexiPilotToolbox) -> None:
    agent = hybrid_agent(toolbox, PlanningClient())
    agent.plan("I have 6 minutes. Review due words and add one new word.")
    names = {event["name"] for event in toolbox.tool_events}
    assert {"get_profile_summary", "get_due_words", "get_missed_words", "get_new_words"} <= names
    assert "record_answer" not in names
    assert "save_session_summary" not in names


def test_partial_tool_selection_is_nudged_only_for_missing_tools(
    toolbox: LexiPilotToolbox,
) -> None:
    toolbox.runtime = dedicated_runtime()
    client = PartialPlanningClient()
    plan = build_model_session_plan(
        toolbox,
        "alice",
        "give me 10 words",
        client=client,
    )
    assert client.calls == 3
    assert plan["planning_mode"] == "model"
    assert "Call only these missing structured tools now: get_due_words, get_missed_words." in str(
        client.requests[1]
    )
    event_names = [event["name"] for event in toolbox.tool_events]
    for name in ("get_profile_summary", "get_new_words", "get_due_words", "get_missed_words"):
        assert event_names.count(name) == 1


def test_repeated_partial_tool_selection_fails_after_targeted_nudge(
    toolbox: LexiPilotToolbox,
) -> None:
    toolbox.runtime = dedicated_runtime()
    client = PartialPlanningClient(repeat_partial=True)
    with pytest.raises(ModelPlanningError, match="repeated tools without inspecting required tools"):
        build_model_session_plan(
            toolbox,
            "alice",
            "give me 10 words",
            client=client,
        )
    assert client.calls == 2


def test_structured_model_plan_creates_interactive_session(toolbox: LexiPilotToolbox) -> None:
    agent = hybrid_agent(toolbox, PlanningClient())
    output = agent.plan("I have 6 minutes. Review due words and add one new word.")
    assert agent.session is not None
    assert [word["word"] for word in agent.session.plan["planned_words"]] == [
        "granular",
        "redeem",
        "impetus",
    ]
    assert "Card 1/3" in output


def test_unknown_model_word_is_rejected(toolbox: LexiPilotToolbox) -> None:
    toolbox.runtime = dedicated_runtime()
    client = PlanningClient(
        {
            "minutes": 6,
            "review_words": ["invented_word"],
            "new_words": [],
            "priority_words": ["invented_word"],
            "selection_reason": "Invalid test plan.",
        }
    )
    with pytest.raises(ModelPlanningError):
        build_model_session_plan(toolbox, "alice", "I have 6 minutes.", client=client)


def test_write_tools_are_unavailable_during_planning() -> None:
    names = {schema["function"]["name"] for schema in planning_tool_schemas()}
    assert names == {
        "get_profile_summary",
        "get_due_words",
        "get_missed_words",
        "get_new_words",
        "get_word_details",
    }
    assert "record_answer" not in names
    assert "save_session_summary" not in names


def test_malformed_model_plan_falls_back_deterministically(
    toolbox: LexiPilotToolbox,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = hybrid_agent(toolbox, PlanningClient("not-json"))
    agent.plan("I have 6 minutes. Review due words.")
    assert agent.session is not None
    assert agent.session.plan["planning_mode"] == "deterministic"
    assert "[WARNING]" in capsys.readouterr().out


def test_plain_text_fake_tool_call_is_rejected(toolbox: LexiPilotToolbox) -> None:
    client = ProseOnlyClient()
    toolbox.runtime = dedicated_runtime()
    agent = LexiPilotAgent(
        "alice",
        toolbox,
        console=Console(ConsoleTheme(enabled=False)),
        planner_client=client,
    )
    agent.plan("I have 6 minutes. Review due words.")
    assert client.calls == 2
    assert agent.session is not None
    assert agent.session.plan["planning_mode"] == "deterministic"


def test_endpoint_failure_falls_back_deterministically(toolbox: LexiPilotToolbox) -> None:
    agent = hybrid_agent(toolbox, PlanningClient(failure=TimeoutError("endpoint timeout")))
    output = agent.plan("I have 6 minutes. Review due words.")
    assert "Card 1/" in output
    assert agent.session is not None
    assert agent.session.plan["planning_mode"] == "deterministic"


def test_deterministic_flag_bypasses_model_planning(toolbox: LexiPilotToolbox) -> None:
    client = PlanningClient(failure=AssertionError("model must not be called"))
    agent = hybrid_agent(toolbox, client, deterministic=True)
    agent.plan("I have 6 minutes. Review due words.")
    assert client.calls == 0
    assert agent.session is not None
    assert agent.session.plan["planning_mode"] == "deterministic"


def test_model_and_controller_labels_are_distinct(
    toolbox: LexiPilotToolbox,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = hybrid_agent(toolbox, PlanningClient(), debug=True)
    agent.plan("I have 6 minutes. Review due words and add one new word.")
    output = capsys.readouterr().out
    assert "[AGENT]" in output
    assert "[MODEL TOOL] get_profile_summary" in output
    assert "[MODEL PLAN]" in output
    assert "[CONTROLLER]" in output
    agent._tool_line("record_answer")
    assert "[TOOL] record_answer" in capsys.readouterr().out


def test_model_plan_error_does_not_leak_credentials(
    toolbox: LexiPilotToolbox,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "private-test-key"
    runtime = dedicated_runtime()
    runtime.api_key = secret
    toolbox.runtime = runtime
    agent = LexiPilotAgent(
        "alice",
        toolbox,
        console=Console(ConsoleTheme(enabled=False)),
        planner_client=PlanningClient(failure=RuntimeError(f"failed with {secret} at {runtime.base_url}")),
    )
    agent.plan("I have 6 minutes.")
    output = capsys.readouterr().out
    assert secret not in output
    assert runtime.base_url not in output
