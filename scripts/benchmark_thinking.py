#!/usr/bin/env python3
"""Benchmark Qwen thinking on/off for LexiPilot Radeon inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vocab_trainer as vt
from console_theme import Console, ConsoleTheme
from lexipilot_tools import LexiPilotRuntime, load_lexipilot_env, openai_tool_schemas, parse_env_bool


WORKLOAD_PLANNING = "agent_planning_tool_calling"
WORKLOAD_GENERATION = "bilingual_practice_generation"
MODES = ("thinking_true", "thinking_false")
REQUEST_TIMEOUT_SECONDS = 90
MAX_TOKENS = 700
TEMPERATURE = 0
TARGET_WORDS = ["abhor", "abiding", "exorbitant"]
MEASUREMENT_NOTE = (
    "Measurements were collected by the LexiPilot client. Latency and client-observed completion tokens/s "
    "include client, network, endpoint, scheduling, and serving overhead. They are not raw GPU kernel throughput."
)


@dataclass
class BenchmarkRun:
    timestamp: str
    mode: str
    workload: str
    run_index: int
    measured: bool
    request_success: bool
    validation_success: bool
    latency_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    client_observed_completion_tokens_per_second: float | None
    finish_reason: str | None
    tool_call_count: int
    structured_tool_call_success: bool | None
    error_category: str | None


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            if not text.endswith("\n"):
                file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def runtime_for_mode(base: LexiPilotRuntime, mode: str) -> LexiPilotRuntime:
    return LexiPilotRuntime(
        model_name=base.model_name,
        endpoint_type=base.endpoint_type,
        base_url=base.base_url,
        api_key=base.api_key,
        enable_thinking=(mode == "thinking_true"),
        performance_reports_enabled=base.performance_reports_enabled,
    )


def request_extra_body(runtime: LexiPilotRuntime) -> dict[str, Any] | None:
    return runtime.dedicated_extra_body()


def sanitized_request_settings(runtime: LexiPilotRuntime) -> dict[str, Any]:
    return {
        "model": runtime.model_name,
        "endpoint_type": runtime.endpoint_type,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "tool_choice": "auto",
        "extra_body": request_extra_body(runtime),
    }


def planning_tools() -> list[dict[str, Any]]:
    allowed = {"get_profile_summary", "get_due_words", "get_missed_words"}
    return [schema for schema in openai_tool_schemas() if schema["function"]["name"] in allowed]


def planning_payload(runtime: LexiPilotRuntime) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": runtime.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are benchmarking LexiPilot. Use structured tools only. "
                    "Call get_profile_summary, get_due_words, and get_missed_words for profile benchmark_alice."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I have 15 minutes today. Inspect my learning status, due words, and frequently missed words, "
                    "then build a concise adaptive study plan."
                ),
            },
        ],
        "tools": planning_tools(),
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    extra = request_extra_body(runtime)
    if extra:
        payload["extra_body"] = extra
    return payload


def generation_payload(runtime: LexiPilotRuntime) -> dict[str, Any]:
    words = [
        {"word": "abhor", "definition": "vt. 痛恨；憎恶"},
        {"word": "abiding", "definition": "adj. 持久的；永久的"},
        {"word": "exorbitant", "definition": "adj. 过高的；过分的"},
    ]
    payload: dict[str, Any] = {
        "model": runtime.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return strict JSON only with keys english_passage, chinese_translation, "
                    "target_translations. target_translations maps each target word to an array of Chinese phrases."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write one concise academic-style English passage and a Chinese translation. "
                    f"Use all target words: {json.dumps(words, ensure_ascii=False)}"
                ),
            },
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    extra = request_extra_body(runtime)
    if extra:
        payload["extra_body"] = extra
    return payload


def post_chat(runtime: LexiPilotRuntime, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    if not runtime.api_key or not runtime.base_url:
        raise RuntimeError("Missing required Radeon configuration.")
    request = urllib.request.Request(
        f"{runtime.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {runtime.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data, round(time.perf_counter() - start, 4)


def response_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        reason = choices[0].get("finish_reason")
        return str(reason) if reason is not None else None
    return None


def usage_values(response: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    return (
        int(prompt) if prompt is not None else None,
        int(completion) if completion is not None else None,
        int(total) if total is not None else None,
    )


def validate_tool_call_message(message: dict[str, Any]) -> tuple[bool, int, str | None]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content = str(message.get("content") or "").lower()
        if "get_profile_summary" in content or "get_due_words" in content:
            return False, 0, "plain_text_fake_tool_call"
        return False, 0, "missing_tool_calls"
    allowed = {"get_profile_summary", "get_due_words", "get_missed_words"}
    names = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            return False, len(tool_calls), "invalid_tool_call_object"
        name = function.get("name")
        if name not in allowed:
            return False, len(tool_calls), "unexpected_tool_name"
        try:
            args = json.loads(function.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError):
            return False, len(tool_calls), "invalid_json_arguments"
        if not isinstance(args, dict):
            return False, len(tool_calls), "arguments_not_object"
        names.append(name)
    if "get_profile_summary" not in names:
        return False, len(tool_calls), "missing_profile_summary_tool"
    return True, len(tool_calls), None


def response_text(response: dict[str, Any]) -> str:
    message = response_message(response)
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def parse_generation_json(response: dict[str, Any]) -> dict[str, Any] | None:
    text = response_text(response).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        parsed_snap = vt.parse_snap_json(text)
        if not parsed_snap:
            return None
        return {
            "english_passage": parsed_snap.get("english", ""),
            "chinese_translation": parsed_snap.get("chinese", ""),
            "target_translations": {},
        }


def inflected_pattern(word: str) -> str:
    base = word.lower()
    variants = {base, base + "s", base + "ed", base + "d", base + "ing"}
    if base.endswith("r"):
        variants.update({base + "red", base + "ring"})
    return r"\b(?:" + "|".join(re_escape for re_escape in sorted(map(__import__("re").escape, variants), key=len, reverse=True)) + r")\b"


def validate_generation_response(response: dict[str, Any]) -> tuple[bool, str | None]:
    import re

    parsed = parse_generation_json(response)
    if parsed is None:
        return False, "invalid_json_response"
    english = str(parsed.get("english_passage") or parsed.get("english") or "").strip()
    chinese = str(parsed.get("chinese_translation") or parsed.get("chinese") or "").strip()
    mapping = parsed.get("target_translations") or parsed.get("target_mappings") or {}
    if not english:
        return False, "empty_english"
    if not chinese:
        return False, "empty_chinese"
    for word in TARGET_WORDS:
        if not re.search(inflected_pattern(word), english, flags=re.IGNORECASE):
            return False, f"missing_target_{word}"
    if not isinstance(mapping, dict):
        return False, "invalid_mapping"
    return True, None


def safe_error_category(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "url_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return type(exc).__name__


def throughput(completion_tokens: int | None, latency_seconds: float | None) -> float | None:
    if completion_tokens is None or latency_seconds is None or latency_seconds <= 0:
        return None
    return round(completion_tokens / latency_seconds, 4)


def run_request(runtime: LexiPilotRuntime, mode: str, workload: str, run_index: int, measured: bool) -> BenchmarkRun:
    timestamp = iso_now()
    try:
        payload = planning_payload(runtime) if workload == WORKLOAD_PLANNING else generation_payload(runtime)
        response, latency = post_chat(runtime, payload)
        prompt, completion, total = usage_values(response)
        message = response_message(response)
        if workload == WORKLOAD_PLANNING:
            validation, tool_count, validation_error = validate_tool_call_message(message)
            structured = validation
        else:
            validation, validation_error = validate_generation_response(response)
            tool_count = len(message.get("tool_calls") or []) if isinstance(message.get("tool_calls"), list) else 0
            structured = None
        return BenchmarkRun(
            timestamp,
            mode,
            workload,
            run_index,
            measured,
            True,
            validation,
            latency,
            prompt,
            completion,
            total,
            throughput(completion, latency),
            finish_reason(response),
            tool_count,
            structured,
            validation_error,
        )
    except Exception as exc:
        return BenchmarkRun(timestamp, mode, workload, run_index, measured, False, False, None, None, None, None, None, None, 0, False if workload == WORKLOAD_PLANNING else None, safe_error_category(exc))


def run_order(runs: int, warmups: int) -> list[tuple[str, str, int, bool]]:
    order: list[tuple[str, str, int, bool]] = []
    for workload in (WORKLOAD_PLANNING, WORKLOAD_GENERATION):
        for mode in MODES:
            for index in range(warmups):
                order.append((workload, mode, index, False))
        for index in range(runs):
            pair = MODES if index % 2 == 0 else tuple(reversed(MODES))
            for mode in pair:
                order.append((workload, mode, index, True))
    return order


def median(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.median(clean), 4) if clean else None


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def percentile_95(values: list[float | int | None]) -> float | None:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    if len(clean) < 2:
        return clean[0]
    index = math.ceil(0.95 * len(clean)) - 1
    return round(clean[min(max(index, 0), len(clean) - 1)], 4)


def aggregate(results: list[BenchmarkRun]) -> dict[str, Any]:
    measured = [run for run in results if run.measured]
    output: dict[str, Any] = {}
    for workload in (WORKLOAD_PLANNING, WORKLOAD_GENERATION):
        output[workload] = {}
        for mode in MODES:
            rows = [run for run in measured if run.workload == workload and run.mode == mode]
            successes = [run for run in rows if run.request_success]
            validations = [run for run in rows if run.validation_success]
            output[workload][mode] = {
                "total_runs": len(rows),
                "successful_runs": len(successes),
                "failed_runs": len(rows) - len(successes),
                "validation_success_rate": round(len(validations) / len(rows), 4) if rows else None,
                "median_latency": median([run.latency_seconds for run in rows if run.request_success]),
                "min_latency": min((run.latency_seconds for run in rows if run.latency_seconds is not None), default=None),
                "max_latency": max((run.latency_seconds for run in rows if run.latency_seconds is not None), default=None),
                "mean_latency": mean([run.latency_seconds for run in rows if run.request_success]),
                "p95_latency": percentile_95([run.latency_seconds for run in rows if run.request_success]),
                "median_prompt_tokens": median([run.prompt_tokens for run in rows if run.request_success]),
                "median_completion_tokens": median([run.completion_tokens for run in rows if run.request_success]),
                "median_client_observed_completion_tokens_per_second": median([run.client_observed_completion_tokens_per_second for run in rows if run.request_success]),
                "structured_tool_call_success_rate": (
                    round(sum(1 for run in rows if run.structured_tool_call_success) / len(rows), 4)
                    if workload == WORKLOAD_PLANNING and rows
                    else None
                ),
            }
    return output


def percent_change(baseline: float | None, optimized: float | None, *, reduction: bool = False) -> float | None:
    if baseline is None or optimized is None or baseline == 0:
        return None
    value = ((baseline - optimized) / baseline * 100) if reduction else ((optimized - baseline) / baseline * 100)
    return round(value, 2)


def comparisons(aggregates: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for workload, rows in aggregates.items():
        baseline = rows["thinking_true"]
        optimized = rows["thinking_false"]
        output[workload] = {
            "latency_reduction_percent": percent_change(baseline.get("median_latency"), optimized.get("median_latency"), reduction=True),
            "throughput_change_percent": percent_change(
                baseline.get("median_client_observed_completion_tokens_per_second"),
                optimized.get("median_client_observed_completion_tokens_per_second"),
            ),
            "completion_token_reduction_percent": percent_change(
                baseline.get("median_completion_tokens"),
                optimized.get("median_completion_tokens"),
                reduction=True,
            ),
        }
    return output


def mock_results(runs: int, warmups: int) -> list[BenchmarkRun]:
    results: list[BenchmarkRun] = []
    for workload, mode, index, measured in run_order(runs, warmups):
        base_latency = 1.7 if workload == WORKLOAD_PLANNING else 2.4
        mode_delta = 0.45 if mode == "thinking_true" else 0.0
        completion = 110 if workload == WORKLOAD_PLANNING else 230
        if mode == "thinking_false":
            completion -= 35
        latency = round(base_latency + mode_delta + index * 0.03, 4)
        validation = True
        request = True
        error = None
        if measured and workload == WORKLOAD_PLANNING and mode == "thinking_true" and index == runs - 1:
            validation = False
            error = "plain_text_fake_tool_call"
        results.append(
            BenchmarkRun(
                iso_now(),
                mode,
                workload,
                index,
                measured,
                request,
                validation,
                latency,
                520,
                completion,
                520 + completion,
                throughput(completion, latency),
                "stop",
                3 if validation and workload == WORKLOAD_PLANNING else 0,
                validation if workload == WORKLOAD_PLANNING else None,
                error,
            )
        )
    return results


def markdown_table(title: str, rows: dict[str, Any], planning: bool = False) -> str:
    true_row = rows["thinking_true"]
    false_row = rows["thinking_false"]
    metric_lines = [
        ("Successful runs", "successful_runs"),
        ("Validation success rate", "validation_success_rate"),
        ("Median latency", "median_latency"),
        ("P95 latency", "p95_latency"),
        ("Median completion tokens", "median_completion_tokens"),
        ("Client-observed completion tokens/s", "median_client_observed_completion_tokens_per_second"),
    ]
    if planning:
        metric_lines.insert(1, ("Tool-call success rate", "structured_tool_call_success_rate"))
    lines = [f"## {title}", "", "| Metric | Thinking Enabled | Thinking Disabled |", "|---|---:|---:|"]
    for label, key in metric_lines:
        lines.append(f"| {label} | {md_value(true_row.get(key))} | {md_value(false_row.get(key))} |")
    return "\n".join(lines)


def md_value(value: Any) -> str:
    return "N/A" if value is None else str(value)


def render_summary_md(summary: dict[str, Any]) -> str:
    cmp_planning = summary["optimization_comparison"][WORKLOAD_PLANNING]
    cmp_generation = summary["optimization_comparison"][WORKLOAD_GENERATION]
    complete = bool(summary["environment"].get("benchmark_complete"))
    recommendation = (
        "QWEN_ENABLE_THINKING=false when validation reliability is acceptable."
        if complete
        else "No final setting recommendation from this run because the benchmark did not complete successfully."
    )
    lines = [
        "# LexiPilot Radeon Inference Benchmark",
        "",
        "## Environment",
        "",
        f"- Model: {summary['environment']['model_name']}",
        f"- Endpoint type: {summary['environment']['endpoint_type']}",
        "- Backend: OpenAI-compatible vLLM endpoint",
        f"- Benchmark date: {summary['environment']['created_at']}",
        f"- Warm-up runs: {summary['configuration']['warmups']}",
        f"- Measured runs: {summary['configuration']['runs']}",
        f"- Mock data: {summary['environment']['mock_data']}",
        f"- Measurement scope: {MEASUREMENT_NOTE}",
        "",
        markdown_table("Agent Planning and Tool Calling", summary["aggregates"][WORKLOAD_PLANNING], planning=True),
        "",
        markdown_table("Bilingual Practice Generation", summary["aggregates"][WORKLOAD_GENERATION]),
        "",
        "## Observed Optimization",
        "",
        f"- Planning latency: {md_value(cmp_planning.get('latency_reduction_percent'))}%",
        f"- Generation latency: {md_value(cmp_generation.get('latency_reduction_percent'))}%",
        f"- Planning completion-token change: {md_value(cmp_planning.get('completion_token_reduction_percent'))}% reduction",
        f"- Generation completion-token change: {md_value(cmp_generation.get('completion_token_reduction_percent'))}% reduction",
        f"- Recommended demo setting: {recommendation}",
        "",
        "## Measurement Note",
        "",
        MEASUREMENT_NOTE,
    ]
    return "\n".join(lines)


def write_reports(results: list[BenchmarkRun], runtime: LexiPilotRuntime, args: argparse.Namespace, output_dir: Path, mock: bool) -> dict[str, Path]:
    aggregates = aggregate(results)
    benchmark_complete = all(
        aggregates[workload][mode]["successful_runs"] > 0
        for workload in (WORKLOAD_PLANNING, WORKLOAD_GENERATION)
        for mode in MODES
    )
    summary = {
        "environment": {
            "created_at": iso_now(),
            "model_name": runtime.model_name,
            "endpoint_type": runtime.endpoint_type,
            "backend": "OpenAI-compatible vLLM endpoint",
            "mock_data": mock,
            "hardware_result": not mock,
            "benchmark_complete": benchmark_complete,
        },
        "configuration": {
            "runs": args.runs,
            "warmups": args.warmups,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "target_words": TARGET_WORDS,
            "request_settings": {mode: sanitized_request_settings(runtime_for_mode(runtime, mode)) for mode in MODES},
        },
        "run_order": [
            {"workload": run.workload, "mode": run.mode, "run_index": run.run_index, "measured": run.measured}
            for run in results
        ],
        "aggregates": aggregates,
        "optimization_comparison": comparisons(aggregates),
        "limitations": [
            "Client-observed timings include network and serving overhead.",
            "Failed and validation-failed runs are retained in raw_results.json and aggregates.",
            "Mock reports are not hardware results and must not be used as submission performance evidence.",
        ],
    }
    paths = {
        "raw": output_dir / "raw_results.json",
        "summary_json": output_dir / "summary.json",
        "summary_md": output_dir / "summary.md",
    }
    atomic_write_json(paths["raw"], [asdict(run) for run in results])
    atomic_write_json(paths["summary_json"], summary)
    atomic_write_text(paths["summary_md"], render_summary_md(summary))
    return paths


def print_terminal_summary(paths: dict[str, Path], results: list[BenchmarkRun], runtime: LexiPilotRuntime, no_color: bool) -> None:
    console = Console(ConsoleTheme(enabled=False if no_color else None))
    summary = aggregate(results)
    console.line(console.theme.title("LexiPilot Radeon Benchmark"))
    for workload, label in ((WORKLOAD_PLANNING, "Agent planning"), (WORKLOAD_GENERATION, "Practice generation")):
        console.line("")
        console.line(label)
        for mode, mode_label in (("thinking_true", "Thinking enabled"), ("thinking_false", "Thinking disabled")):
            row = summary[workload][mode]
            median_latency = row.get("median_latency")
            latency_text = f"{median_latency}s" if median_latency is not None else "N/A"
            throughput_value = row.get("median_client_observed_completion_tokens_per_second")
            throughput_text = str(throughput_value) if throughput_value is not None else "N/A"
            console.line(
                f"  {mode_label}: median {latency_text}, "
                f"validation {row.get('validation_success_rate')}, "
                f"client-observed completion tokens/s {throughput_text}"
            )
    console.line("")
    console.line(f"Model: {runtime.model_name}")
    console.line(f"Endpoint: {runtime.endpoint_type}")
    console.line("Reports:")
    console.line(f"  {paths['summary_md']}")
    console.line(f"  {paths['summary_json']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LexiPilot Qwen thinking=true vs thinking=false.")
    parser.add_argument("--env-file", help="Optional env file, for example .env")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-dir", default="benchmark_reports")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("--runs must be >= 1 and --warmups must be >= 0")

    if not args.mock:
        load_lexipilot_env(args.env_file)
    runtime = LexiPilotRuntime()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"thinking_{timestamp}"
    if args.mock:
        runtime.api_key = ""
        runtime.base_url = ""
        results = mock_results(args.runs, args.warmups)
    else:
        results = []
        for workload, mode, index, measured in run_order(args.runs, args.warmups):
            results.append(run_request(runtime_for_mode(runtime, mode), mode, workload, index, measured))
    paths = write_reports(results, runtime, args, output_dir, args.mock)
    print_terminal_summary(paths, results, runtime, args.no_color)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
