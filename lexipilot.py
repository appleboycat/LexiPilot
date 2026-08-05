#!/usr/bin/env python3
"""CLI entry point for LexiPilot."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from console_theme import Console, ConsoleTheme
from lexipilot_core import LexiPilotAgent, SessionPhase, is_internal_control_only
from lexipilot_tools import ConfigError, LexiPilotRuntime, LexiPilotToolbox, load_lexipilot_env
from scripts.backup_default_profile import backup_default_profile
from scripts.show_study_heatmap import activity_from_daily_stats, render_heatmap

REPO_ROOT = Path(__file__).resolve().parent


def print_banner(profile: str, runtime: LexiPilotRuntime, console: Console) -> None:
    print(console.theme.title("LexiPilot"))
    print("Private Adaptive Vocabulary Learning Agent")
    print(f"Profile: {profile}")
    print(f"Model: {console.theme.cyan(runtime.model_name)}")
    print(f"Endpoint: {console.theme.cyan(runtime.endpoint_type)}")
    print("Commands: /reset, /status, /activity [days], /exit")


def print_response(text: str) -> None:
    if text and not is_internal_control_only(text):
        print(text)


def print_profile_status(
    profile: str,
    toolbox: LexiPilotToolbox,
    console: Console,
    runtime: LexiPilotRuntime,
    *,
    debug: bool = False,
) -> None:
    if debug:
        console.tool("get_profile_summary")
    summary = toolbox.get_profile_summary(profile)
    console.profile_status(summary, {"model": runtime.model_name, "endpoint": runtime.endpoint_type})


def print_profile_activity(
    profile: str,
    toolbox: LexiPilotToolbox,
    console: Console,
    days: int = 28,
) -> None:
    state = toolbox.load_state(profile)
    daily_stats = state.get("daily_stats", {})
    if not isinstance(daily_stats, dict):
        daily_stats = {}
    activity = activity_from_daily_stats(daily_stats, days)
    print(
        render_heatmap(
            activity,
            no_color=not bool(console.theme.enabled),
            source=f"aggregated learner activity for profile {profile}",
        )
    )


def parse_activity_days(command: str) -> int:
    parts = command.split()
    if len(parts) == 1:
        return 28
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError("Usage: /activity [days], where days is between 1 and 365.")
    days = int(parts[1])
    if not 1 <= days <= 365:
        raise ValueError("Activity days must be between 1 and 365.")
    return days


POST_COMPLETION_STUDY_INPUTS = {"y", "yes", "n", "no", "e", "etymology", "s", "skip", "stop"}
TERMINAL_PHASES = {SessionPhase.COMPLETED, SessionPhase.STOPPED, SessionPhase.FAILED}


def should_start_new_request_after_completion(agent: LexiPilotAgent, text: str) -> bool:
    if agent.session is None or agent.session.phase not in TERMINAL_PHASES:
        return False
    return text.strip().lower() not in POST_COMPLETION_STUDY_INPUTS


def looks_like_new_study_request(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in POST_COMPLETION_STUDY_INPUTS or lowered.startswith("/"):
        return False
    if re.search(r"\b\d{1,3}\s*(?:words?|minutes?|mins?)\b", lowered):
        return True
    return any(phrase in lowered for phrase in ("give me", "review", "study", "focus on", "practice words"))


def resolve_data_paths(args: argparse.Namespace) -> tuple[str, Path | None, Path | None]:
    profile = args.profile or ("demo" if args.demo else "default")
    index_path = Path(args.index_file).expanduser() if args.index_file else None
    progress_root = Path(args.progress_root).expanduser() if args.progress_root else None
    if args.demo:
        index_path = index_path or REPO_ROOT / "examples" / "sample_vocab_index.json"
        progress_root = progress_root or REPO_ROOT / ".demo_data" / "profiles"
    return profile, index_path, progress_root


def main() -> None:
    parser = argparse.ArgumentParser(description="LexiPilot private adaptive vocabulary learning agent.")
    parser.add_argument("--profile", help="Learner profile name (default: default, or demo with --demo)")
    parser.add_argument("--env-file", help="Optional environment file (public default: .env)")
    parser.add_argument("--index-file", help="Explicit vocabulary index JSON path")
    parser.add_argument("--progress-root", help="Explicit learner-profile root directory")
    parser.add_argument("--demo", action="store_true", help="Use the reproducible sample vocabulary and synthetic demo profile")
    parser.add_argument("--deterministic", action="store_true", help="Bypass model planning and use the deterministic planner")
    parser.add_argument("--debug", action="store_true", help="Show concise tool timeline")
    parser.add_argument("--no-color", action="store_true", help="Disable terminal colors")
    parser.add_argument("--backup-profile", action="store_true", help="Back up the default profile before recording answers")
    parser.add_argument(
        "--model-loop",
        action="store_true",
        help="Deprecated compatibility alias; hybrid model planning is now the default",
    )
    args = parser.parse_args()
    profile, index_path, progress_root = resolve_data_paths(args)

    if args.demo:
        from scripts.setup_demo_data import setup_demo_data

        assert index_path is not None and progress_root is not None
        demo_progress = progress_root / profile / "progress.json"
        if not demo_progress.exists():
            setup_demo_data(index_path=index_path, progress_root=progress_root, profile=profile)
    if index_path is not None and not index_path.exists():
        raise SystemExit(f"Vocabulary index not found: {index_path}")

    try:
        load_lexipilot_env(args.env_file)
        runtime = LexiPilotRuntime()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    console = Console(ConsoleTheme(enabled=False if args.no_color else None))
    toolbox = LexiPilotToolbox(
        index_path=index_path,
        progress_dir=progress_root,
        state_file=(progress_root / ".legacy_state.json") if progress_root is not None else None,
        report_dir=(REPO_ROOT / ".demo_data" / "performance_reports") if args.demo else None,
        material_dir=(REPO_ROOT / ".demo_data" / "materials") if args.demo else None,
        runtime=runtime,
    )
    agent = LexiPilotAgent(
        profile,
        toolbox,
        debug=args.debug,
        console=console,
        deterministic=args.deterministic,
    )
    print_banner(profile, runtime, console)
    backed_up = False
    real_default = (
        profile == "default"
        and (progress_root is None or progress_root.resolve() == (REPO_ROOT / ".vocab_progress").resolve())
    )
    if real_default:
        console.status("Using existing learner profile: default")
        console.status("A backup is recommended before recording answers.")
        if args.backup_profile:
            path = backup_default_profile()
            backed_up = True
            console.saved(f"Default profile backup: {path}")
    print_profile_status(profile, toolbox, console, runtime, debug=args.debug)

    while True:
        try:
            input_started = time.perf_counter()
            text = input("\n> ").strip()
        except EOFError:
            print()
            break
        finally:
            if "input_started" in locals():
                agent.add_user_wait(time.perf_counter() - input_started)
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/reset":
            agent.session = None
            print("Session reset.")
            continue
        if text == "/status":
            print_profile_status(profile, toolbox, console, runtime, debug=args.debug)
            continue
        if text.startswith("/activity"):
            try:
                print_profile_activity(
                    profile,
                    toolbox,
                    console,
                    parse_activity_days(text),
                )
            except ValueError as exc:
                console.warning(str(exc))
            continue
        if should_start_new_request_after_completion(agent, text):
            agent.session = None
        elif agent.session is not None and looks_like_new_study_request(text):
            console.status("Starting a new study request.")
            agent.session = None
        if agent.session is None:
            print_response(agent.plan(text))
        else:
            if real_default and args.backup_profile and not backed_up:
                path = backup_default_profile()
                backed_up = True
                console.saved(f"Default profile backup: {path}")
            print_response(agent.handle_answer(text))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited.")
        sys.exit(130)
