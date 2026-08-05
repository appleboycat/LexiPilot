#!/usr/bin/env python3
"""CLI entry point for LexiPilot."""

from __future__ import annotations

import argparse
import re
import sys
import time

from console_theme import Console, ConsoleTheme
from lexipilot_core import LexiPilotAgent, SessionPhase, is_internal_control_only, run_tool_call_loop
from lexipilot_tools import ConfigError, LexiPilotRuntime, LexiPilotToolbox, load_lexipilot_env
from scripts.backup_default_profile import backup_default_profile


def print_banner(profile: str, runtime: LexiPilotRuntime, console: Console) -> None:
    print(console.theme.title("LexiPilot"))
    print("Private Adaptive Vocabulary Learning Agent")
    print(f"Profile: {profile}")
    print(f"Model: {console.theme.cyan(runtime.model_name)}")
    print(f"Endpoint: {console.theme.cyan(runtime.endpoint_type)}")
    print("Commands: /reset, /status, /exit")


def print_response(text: str) -> None:
    if text and not is_internal_control_only(text):
        print(text)


def print_profile_status(profile: str, toolbox: LexiPilotToolbox, console: Console, *, debug: bool = False) -> None:
    if debug:
        console.tool("get_profile_summary")
    summary = toolbox.get_profile_summary(profile)
    console.profile_status(summary)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="LexiPilot private adaptive vocabulary learning agent.")
    parser.add_argument("--profile", default="default", help="Learner profile name")
    parser.add_argument("--env-file", help="Optional env file, for example ../aiagent/.env")
    parser.add_argument("--debug", action="store_true", help="Show concise tool timeline")
    parser.add_argument("--no-color", action="store_true", help="Disable terminal colors")
    parser.add_argument("--backup-profile", action="store_true", help="Back up the default profile before recording answers")
    parser.add_argument("--model-loop", action="store_true", help="Use the OpenAI-compatible model tool-calling loop for one request")
    args = parser.parse_args()

    try:
        load_lexipilot_env(args.env_file)
        runtime = LexiPilotRuntime()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    console = Console(ConsoleTheme(enabled=False if args.no_color else None))
    toolbox = LexiPilotToolbox(runtime=runtime)
    agent = LexiPilotAgent(args.profile, toolbox, debug=args.debug, console=console)
    print_banner(args.profile, runtime, console)
    backed_up = False
    if args.profile == "default":
        console.status("Using existing learner profile: default")
        console.status("A backup is recommended before recording answers.")
        if args.backup_profile:
            path = backup_default_profile()
            backed_up = True
            console.saved(f"Default profile backup: {path}")
    print_profile_status(args.profile, toolbox, console, debug=args.debug)

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
            print_profile_status(args.profile, toolbox, console, debug=args.debug)
            continue
        if should_start_new_request_after_completion(agent, text):
            agent.session = None
        elif agent.session is not None and looks_like_new_study_request(text):
            console.status("Starting a new study request.")
            agent.session = None
        if args.model_loop and agent.session is None:
            try:
                print_response(run_tool_call_loop(toolbox, args.profile, text, debug=args.debug))
            except Exception as exc:
                console.error(f"Model tool loop unavailable; falling back to deterministic session planner. {exc}")
                print_response(agent.plan(text))
            continue
        if agent.session is None:
            print_response(agent.plan(text))
        else:
            if args.profile == "default" and args.backup_profile and not backed_up:
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
