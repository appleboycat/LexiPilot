#!/usr/bin/env python3
"""Safe Radeon dedicated endpoint verification for LexiPilot."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lexipilot_tools import ConfigError, LexiPilotRuntime, load_lexipilot_env


class VerificationError(RuntimeError):
    pass


def completion_payload(runtime: LexiPilotRuntime, **kwargs: Any) -> dict[str, Any]:
    payload = dict(kwargs)
    extra = runtime.dedicated_extra_body()
    if extra is not None:
        payload["extra_body"] = extra
    return payload


def post_completion(runtime: LexiPilotRuntime, payload: dict[str, Any]) -> dict[str, Any]:
    if not runtime.api_key or not runtime.base_url or not runtime.model_name:
        raise ConfigError("Missing required Radeon configuration: RADEON_API_KEY, RADEON_BASE_URL, RADEON_MODEL")
    request = urllib.request.Request(
        f"{runtime.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {runtime.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def first_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VerificationError("response did not contain choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise VerificationError("response did not contain an assistant message")
    return message


def run_basic_completion(runtime: LexiPilotRuntime) -> None:
    response = post_completion(
        runtime,
        completion_payload(
            runtime,
            model=runtime.model_name,
            messages=[{"role": "user", "content": "Reply with a short confirmation."}],
            temperature=0,
        ),
    )
    content = str(first_message(response).get("content") or "").strip()
    if not content:
        raise VerificationError("basic completion returned an empty assistant response")


def diagnostic_tool_definition() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_test_profile",
                "description": "Return a deterministic diagnostic learner profile.",
                "parameters": {
                    "type": "object",
                    "properties": {"profile": {"type": "string"}},
                    "required": ["profile"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def verify_tool_call_message(message: dict[str, Any], expected_name: str = "get_test_profile") -> None:
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list) or not tool_calls:
        raise VerificationError(
            "response did not include structured tool_calls; vLLM may be missing "
            "--enable-auto-tool-choice or --tool-call-parser hermes"
        )
    function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
    if not isinstance(function, dict):
        raise VerificationError("tool call did not include a function object")
    if function.get("name") != expected_name:
        raise VerificationError(f"unexpected tool call name: {function.get('name') or '<missing>'}")
    raw_arguments = function.get("arguments", "")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VerificationError("tool call arguments were not valid JSON") from exc
    if not isinstance(arguments, dict) or arguments.get("profile") != "demo":
        raise VerificationError("tool call arguments must include profile='demo'")


def run_tool_calling_check(runtime: LexiPilotRuntime) -> None:
    response = post_completion(
        runtime,
        completion_payload(
            runtime,
            model=runtime.model_name,
            messages=[{"role": "user", "content": "Use the get_test_profile tool for profile demo."}],
            tools=diagnostic_tool_definition(),
            tool_choice="auto",
            temperature=0,
        ),
    )
    verify_tool_call_message(first_message(response))


def safe_error_message(exc: BaseException, runtime: LexiPilotRuntime | None) -> str:
    text = str(exc)
    if runtime is not None:
        if runtime.api_key:
            text = text.replace(runtime.api_key, "[redacted]")
        if runtime.base_url:
            text = text.replace(runtime.base_url, "[redacted_base_url]")
    return text[:240]


def diagnose_failure(exc: BaseException, runtime: LexiPilotRuntime | None = None) -> str:
    if isinstance(exc, ConfigError):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {401, 403}:
            return "authentication failure: check the API key and authorization settings"
        if exc.code == 404:
            return "not found: check RADEON_BASE_URL and ensure /v1 is not duplicated"
        if exc.code == 503:
            return "instance not ready: wait for the dedicated model service to finish loading"
    text = safe_error_message(exc, runtime).lower()
    if "model" in text and "not" in text and "found" in text:
        return "model-name mismatch: check RADEON_MODEL against the ready instance"
    if "timed out" in text or "timeout" in text:
        return "connection timeout: instance may not be ready or the endpoint is unavailable"
    if isinstance(exc, VerificationError) and "tool_calls" in text:
        return "tool-calling failure: vLLM may need --enable-auto-tool-choice and --tool-call-parser hermes"
    return f"{type(exc).__name__}: {safe_error_message(exc, runtime)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a LexiPilot Radeon dedicated endpoint.")
    parser.add_argument("--env-file", help="Optional env file, for example .env")
    args = parser.parse_args()
    runtime: LexiPilotRuntime | None = None
    try:
        load_lexipilot_env(args.env_file)
        runtime = LexiPilotRuntime()
        started = time.perf_counter()
        run_basic_completion(runtime)
        print("PASS basic completion")
        run_tool_calling_check(runtime)
        print("PASS tool calling")
        print("PASS Radeon endpoint verification")
        _ = started
        return 0
    except Exception as exc:
        print(f"FAIL Radeon endpoint verification: {diagnose_failure(exc, runtime)}")
        return 1


def fake_tool_message(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    raw = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return {"tool_calls": [{"function": {"name": "get_test_profile", "arguments": raw}}]}


if __name__ == "__main__":
    raise SystemExit(main())
