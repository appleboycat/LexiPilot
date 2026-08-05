#!/usr/bin/env python3
"""Model-free smoke test for the public sample-data hybrid Agent path."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from console_theme import Console, ConsoleTheme
from lexipilot_core import LexiPilotAgent
from lexipilot_tools import LexiPilotRuntime, LexiPilotToolbox
from scripts.setup_demo_data import setup_demo_data


class MockPlanningClient:
    def __init__(self) -> None:
        self.request_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 700,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        self.request_count += 1
        if self.request_count == 1:
            calls = [
                ("call_summary", "get_profile_summary", {"profile": "demo"}),
                ("call_due", "get_due_words", {"profile": "demo", "limit": 12}),
                (
                    "call_missed",
                    "get_missed_words",
                    {"profile": "demo", "limit": 8, "date": None, "highest_first": True},
                ),
                ("call_new", "get_new_words", {"profile": "demo", "limit": 3, "from_page": None}),
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
                                for call_id, name, arguments in calls
                            ],
                        }
                    }
                ]
            }
        payload = {
            "minutes": 15,
            "review_words": ["abhor", "abate", "aberrant", "abandon", "abbey", "abbreviate"],
            "new_words": ["accelerate"],
            "priority_words": ["abhor", "aberrant", "abate"],
            "selection_reason": "Due and historically missed words are prioritized before one new word.",
        }
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="lexipilot_fresh_clone_"))
    try:
        index = REPO_ROOT / "examples" / "sample_vocab_index.json"
        progress_root = root / "profiles"
        setup_demo_data(index_path=index, progress_root=progress_root, profile="demo")
        check(index.exists(), "sample vocabulary loaded")

        runtime = LexiPilotRuntime(
            model_name="Qwen/Qwen3-8B",
            endpoint_type="dedicated",
            base_url="https://mock.invalid/v1",
            api_key="mock-key",
            performance_reports_enabled=False,
        )
        toolbox = LexiPilotToolbox(
            index_path=index,
            progress_dir=progress_root,
            state_file=root / "legacy.json",
            report_dir=root / "reports",
            material_dir=root / "materials",
            runtime=runtime,
        )
        client = MockPlanningClient()
        agent = LexiPilotAgent(
            "demo",
            toolbox,
            console=Console(ConsoleTheme(enabled=False)),
            planner_client=client,
        )
        agent.plan("I have 15 minutes. Review due and frequently missed words, then add one new word.")
        check(client.request_count == 2, "model tools called")
        check(agent.session is not None and agent.session.plan["planning_mode"] == "model", "structured plan validated")
        check(agent.session is not None and len(agent.session.plan["planned_words"]) == 7, "interactive session created")

        first_word = agent.session.current_word()["word"]
        agent.handle_answer("y")
        check(toolbox.find_entry(first_word) is not None, "correct answer recorded")
        second_word = agent.session.current_word()["word"]
        agent.handle_answer("n")
        state = toolbox.load_state("demo")
        second_seq = toolbox.find_entry(second_word)["seq"]
        check(state["cards"][str(second_seq)]["stage"] == 0, "incorrect answer recorded")

        runtime.base_url = ""
        runtime.api_key = ""
        agent.handle_answer("stop")
        check(agent.session.generated_material_path is not None, "practice generated")
        check((progress_root / "demo" / "progress.json").exists(), "progress saved")
    finally:
        shutil.rmtree(root)
    check(not root.exists(), "fresh-clone smoke test")


if __name__ == "__main__":
    main()
