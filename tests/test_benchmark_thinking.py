from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from lexipilot_tools import LexiPilotRuntime
from scripts.benchmark_thinking import (
    WORKLOAD_GENERATION,
    WORKLOAD_PLANNING,
    BenchmarkRun,
    aggregate,
    comparisons,
    generation_payload,
    mock_results,
    percentile_95,
    planning_payload,
    run_order,
    runtime_for_mode,
    sanitized_request_settings,
    throughput,
    validate_generation_response,
    validate_tool_call_message,
    write_reports,
    render_summary_md,
)


def test_warmups_excluded_from_aggregates() -> None:
    results = mock_results(runs=3, warmups=1)
    aggregates = aggregate(results)
    assert sum(1 for run in results if not run.measured) == 4
    assert aggregates[WORKLOAD_PLANNING]["thinking_true"]["total_runs"] == 3


def test_equal_request_settings_except_thinking_extra_body() -> None:
    base = LexiPilotRuntime(model_name="Qwen/Qwen3-8B", endpoint_type="dedicated", base_url="https://example.test/v1", api_key="SECRET")
    enabled = sanitized_request_settings(runtime_for_mode(base, "thinking_true"))
    disabled = sanitized_request_settings(runtime_for_mode(base, "thinking_false"))
    enabled_without_extra = dict(enabled)
    disabled_without_extra = dict(disabled)
    enabled_without_extra.pop("extra_body")
    disabled_without_extra.pop("extra_body")
    assert enabled_without_extra == disabled_without_extra
    assert enabled["extra_body"] is None
    assert disabled["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_alternating_measured_order() -> None:
    order = [item for item in run_order(runs=3, warmups=1) if item[0] == WORKLOAD_PLANNING and item[3]]
    assert [item[1] for item in order] == [
        "thinking_true",
        "thinking_false",
        "thinking_false",
        "thinking_true",
        "thinking_true",
        "thinking_false",
    ]


def test_median_p95_and_comparison_math() -> None:
    assert aggregate(mock_results(3, 0))[WORKLOAD_GENERATION]["thinking_false"]["median_latency"] == 2.43
    assert percentile_95([1, 2, 3, 4, 5]) == 5.0
    comp = comparisons(aggregate(mock_results(3, 0)))
    assert comp[WORKLOAD_PLANNING]["latency_reduction_percent"] is not None


def test_throughput_zero_handling() -> None:
    assert throughput(0, 2.0) == 0
    assert throughput(10, 0) is None
    assert throughput(None, 2.0) is None


def test_failed_runs_retained_and_validation_distinct() -> None:
    rows = [
        BenchmarkRun("t", "thinking_true", WORKLOAD_PLANNING, 0, True, True, False, 1.0, 1, 1, 2, 1.0, "stop", 0, False, "plain_text_fake_tool_call"),
        BenchmarkRun("t", "thinking_true", WORKLOAD_PLANNING, 1, True, False, False, None, None, None, None, None, None, 0, False, "timeout"),
    ]
    agg = aggregate(rows)[WORKLOAD_PLANNING]["thinking_true"]
    assert agg["total_runs"] == 2
    assert agg["successful_runs"] == 1
    assert agg["failed_runs"] == 1
    assert agg["validation_success_rate"] == 0


def test_structured_tool_call_validation_and_fake_rejection() -> None:
    ok, count, error = validate_tool_call_message(
        {
            "tool_calls": [
                {"function": {"name": "get_profile_summary", "arguments": json.dumps({"profile": "benchmark_alice"})}},
                {"function": {"name": "get_due_words", "arguments": json.dumps({"profile": "benchmark_alice", "limit": 10})}},
            ]
        }
    )
    assert ok is True
    assert count == 2
    assert error is None
    ok, _, error = validate_tool_call_message({"content": "I will call get_profile_summary now."})
    assert ok is False
    assert error == "plain_text_fake_tool_call"
    ok, _, error = validate_tool_call_message({"tool_calls": [{"function": {"name": "get_profile_summary", "arguments": "{"}}]})
    assert ok is False
    assert error == "invalid_json_arguments"


def test_bilingual_generation_validation() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "english_passage": "Scholars abhor waste, value abiding patience, and reject exorbitant fees.",
                            "chinese_translation": "学者憎恶浪费，重视持久耐心，并反对过高费用。",
                            "target_translations": {"abhor": ["憎恶"], "abiding": ["持久"], "exorbitant": ["过高"]},
                        }
                    )
                },
                "finish_reason": "stop",
            }
        ]
    }
    ok, error = validate_generation_response(response)
    assert ok is True
    assert error is None
    bad = {"choices": [{"message": {"content": "not json"}}]}
    ok, error = validate_generation_response(bad)
    assert ok is False
    assert error == "invalid_json_response"


def test_payload_shapes() -> None:
    runtime = LexiPilotRuntime(model_name="Qwen/Qwen3-8B", endpoint_type="shared", base_url="https://example.test/v1", api_key="SECRET")
    assert "tools" in planning_payload(runtime)
    assert generation_payload(runtime)["response_format"] == {"type": "json_object"}


def test_atomic_report_writing_and_privacy(tmp_path: Path) -> None:
    runtime = LexiPilotRuntime(model_name="Qwen/Qwen3-8B", endpoint_type="dedicated", base_url="https://private.example/v1", api_key="SECRET_KEY")
    args = Namespace(runs=2, warmups=1)
    paths = write_reports(mock_results(2, 1), runtime, args, tmp_path / "bench", mock=True)
    raw = paths["raw"].read_text(encoding="utf-8")
    summary = paths["summary_json"].read_text(encoding="utf-8")
    assert "SECRET_KEY" not in raw + summary
    assert "private.example" not in raw + summary
    data = json.loads(summary)
    assert data["environment"]["mock_data"] is True
    assert data["environment"]["hardware_result"] is False


def test_incomplete_summary_uses_na_and_no_recommendation() -> None:
    rows = [
        BenchmarkRun("t", "thinking_true", WORKLOAD_PLANNING, 0, True, False, False, None, None, None, None, None, None, 0, False, "url_error"),
        BenchmarkRun("t", "thinking_false", WORKLOAD_PLANNING, 0, True, False, False, None, None, None, None, None, None, 0, False, "url_error"),
        BenchmarkRun("t", "thinking_true", WORKLOAD_GENERATION, 0, True, False, False, None, None, None, None, None, None, 0, None, "url_error"),
        BenchmarkRun("t", "thinking_false", WORKLOAD_GENERATION, 0, True, False, False, None, None, None, None, None, None, 0, None, "url_error"),
    ]
    aggs = aggregate(rows)
    summary = {
        "environment": {
            "model_name": "Qwen/Qwen3-8B",
            "endpoint_type": "dedicated",
            "created_at": "t",
            "mock_data": False,
            "benchmark_complete": False,
        },
        "configuration": {"warmups": 1, "runs": 1},
        "aggregates": aggs,
        "optimization_comparison": comparisons(aggs),
    }
    md = render_summary_md(summary)
    assert "N/A" in md
    assert "did not complete successfully" in md
