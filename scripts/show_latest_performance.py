#!/usr/bin/env python3
"""Print a safe summary of the latest LexiPilot performance report."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    reports = sorted(Path("performance_reports").glob("lexipilot_*.json"))
    if not reports:
        print("No LexiPilot performance reports found.")
        return 1
    path = reports[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"Report: {path}")
    print(f"Model: {data.get('model_name')}")
    print(f"Endpoint: {data.get('endpoint_type')}")
    print(f"Model requests: {data.get('model_request_count')}")
    print(f"Tool calls: {data.get('tool_call_count')}")
    print(f"Prompt tokens: {data.get('prompt_tokens')}")
    print(f"Completion tokens: {data.get('completion_tokens')}")
    print(f"Story generation duration: {data.get('story_generation_duration')}")
    print(f"Total duration: {data.get('total_task_duration')}")
    print(f"Timing note: {data.get('timing_semantics')}")
    print(f"Session wall seconds: {data.get('session_wall_seconds')}")
    print(f"User interaction seconds: {data.get('user_interaction_wait_seconds')}")
    print(f"Active system seconds: {data.get('active_system_seconds')}")
    breakdown = data.get("non_overlapping_timing_breakdown") or {}
    print("Non-overlapping breakdown:")
    print(f"  User interaction wait: {breakdown.get('user_interaction_wait_seconds')}")
    print(f"  Model/API execution: {breakdown.get('model_api_execution_seconds')}")
    print(f"  Local non-model processing: {breakdown.get('local_non_model_processing_seconds')}")
    print("Overlapping detail timings:")
    print(f"  Model request seconds: {data.get('model_request_seconds')}")
    print(f"  Tool execution seconds: {data.get('tool_execution_seconds')}")
    print(f"  Story generation seconds: {data.get('story_generation_seconds')}")
    print(f"Final state: {data.get('final_session_state')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
