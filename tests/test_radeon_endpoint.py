from __future__ import annotations

import os
from pathlib import Path

import pytest

from lexipilot_tools import LexiPilotRuntime, load_lexipilot_env, normalize_base_url
from scripts import test_radeon_endpoint as endpoint


RADEON_KEYS = [
    "RADEON_API_KEY",
    "RADEON_BASE_URL",
    "RADEON_MODEL",
    "ENDPOINT_TYPE",
    "QWEN_ENABLE_THINKING",
    "PERFORMANCE_REPORTS_ENABLED",
    "LEXIPILOT_ENV_FILE",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in RADEON_KEYS:
        monkeypatch.delenv(key, raising=False)


def write_env(path: Path, *, key: str, base: str, model: str = "Qwen/Qwen3-8B") -> None:
    path.write_text(
        f"RADEON_API_KEY={key}\n"
        f"RADEON_BASE_URL={base}\n"
        f"RADEON_MODEL={model}\n"
        "ENDPOINT_TYPE=dedicated\n"
        "QWEN_ENABLE_THINKING=false\n"
        "PERFORMANCE_REPORTS_ENABLED=true\n",
        encoding="utf-8",
    )


def test_explicit_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / "demo.env"
    write_env(env_file, key="explicit-key", base="https://explicit.example")
    load_lexipilot_env(env_file)
    runtime = LexiPilotRuntime()
    assert runtime.api_key == "explicit-key"
    assert runtime.base_url == "https://explicit.example/v1"


def test_explicit_env_file_uses_adjacent_env_local_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    write_env(env_file, key="", base="")
    (tmp_path / ".env.local").write_text(
        "RADEON_API_KEY=local-secret\n"
        "RADEON_BASE_URL=https://local-override.example\n"
        "RADEON_MODEL=Qwen/Qwen3-8B\n",
        encoding="utf-8",
    )
    load_lexipilot_env(env_file)
    runtime = LexiPilotRuntime()
    assert runtime.api_key == "local-secret"
    assert runtime.base_url == "https://local-override.example/v1"


def test_env_local_does_not_override_process_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    write_env(env_file, key="", base="")
    (tmp_path / ".env.local").write_text(
        "RADEON_API_KEY=local-secret\n"
        "RADEON_BASE_URL=https://local-override.example\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RADEON_API_KEY", "process-secret")
    load_lexipilot_env(env_file)
    assert os.environ["RADEON_API_KEY"] == "process-secret"


def test_lexipilot_env_file_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / "selected.env"
    write_env(env_file, key="variable-key", base="https://variable.example")
    monkeypatch.setenv("LEXIPILOT_ENV_FILE", str(env_file))
    load_lexipilot_env()
    assert LexiPilotRuntime().api_key == "variable-key"


def test_process_environment_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / "demo.env"
    write_env(env_file, key="file-key", base="https://file.example")
    monkeypatch.setenv("RADEON_API_KEY", "process-key")
    load_lexipilot_env(env_file)
    assert os.environ["RADEON_API_KEY"] == "process-key"


def test_local_env_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_env = tmp_path / ".env"
    write_env(local_env, key="local-key", base="https://local.example")
    monkeypatch.setattr("lexipilot_tools.REPO_ROOT", tmp_path)
    load_lexipilot_env()
    assert LexiPilotRuntime().api_key == "local-key"


def test_sibling_aiagent_env_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sibling = tmp_path / "aiagent.env"
    write_env(sibling, key="sibling-key", base="https://sibling.example")
    monkeypatch.setattr("lexipilot_tools.REPO_ROOT", tmp_path)
    monkeypatch.setattr("lexipilot_tools.SIBLING_AIAGENT_ENV", sibling)
    load_lexipilot_env()
    assert LexiPilotRuntime().api_key == "sibling-key"


@pytest.mark.parametrize(
    ("input_url", "expected"),
    [
        ("https://host.example", "https://host.example/v1"),
        ("https://host.example/", "https://host.example/v1"),
        ("https://host.example/v1", "https://host.example/v1"),
        ("https://host.example/v1/", "https://host.example/v1"),
        ("https://host.example/openai", "https://host.example/openai/v1"),
        ("https://host.example/openai/v1", "https://host.example/openai/v1"),
    ],
)
def test_base_url_normalization(input_url: str, expected: str) -> None:
    assert normalize_base_url(input_url) == expected


def test_valid_structured_tool_call() -> None:
    endpoint.verify_tool_call_message(endpoint.fake_tool_message({"profile": "demo"}))


def test_plain_text_fake_tool_call_rejected() -> None:
    with pytest.raises(endpoint.VerificationError, match="tool_calls"):
        endpoint.verify_tool_call_message({"content": "get_test_profile({profile: demo})"})


def test_safe_diagnostics_redact_key_and_base_url() -> None:
    runtime = LexiPilotRuntime(api_key="SECRET_KEY", base_url="https://host.example/v1")
    message = endpoint.diagnose_failure(RuntimeError("failed SECRET_KEY https://host.example/v1"), runtime)
    assert "SECRET_KEY" not in message
    assert "https://host.example/v1" not in message
