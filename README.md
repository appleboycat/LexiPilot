# LexiPilot

**LexiPilot - A Private Adaptive Vocabulary Learning Agent** is a self-hosted learning agent that analyzes vocabulary history, plans personalized study sessions, invokes learner-memory tools, adapts to mistakes, and generates targeted academic vocabulary practice with a model served from a dedicated AMD Radeon Cloud GPU instance.

## Problem

Vocabulary learners often have word lists, review history, and missed-word logs, but still need to manually decide what to study next. Static review flows also struggle to adapt when the learner has only a few minutes or repeatedly misses the same words.

## Target Users

LexiPilot is for learners who want private, adaptive academic vocabulary review on a user-controlled machine or cloud workspace, especially learners who already maintain PDF-derived vocabulary data and long-term progress history.

## Solution

LexiPilot keeps the existing `vocab_trainer.py` vocabulary engine and adds a thin Agent layer around it. The Agent reads real learner state, plans a focused session, records explicit answers, adapts after mistakes, and generates personalized practice material.

## Why This Is an Agent

LexiPilot accepts a natural-language learning objective, uses tools to inspect real learner data, chooses actions, updates memory only after explicit learner answers, generates practice material, saves progress, and reports what happened. It is more than a chat wrapper because its responses are grounded in tool calls and persistent learner state.

## Agent Workflow

1. Read the learner profile summary.
2. Inspect due reviews.
3. Inspect frequently missed words.
4. Estimate session size from the user's time limit.
5. Prefer due and high-missed words before new words.
6. Present a practical plan with selection reasons.
7. Guide review cards with `y`, `n`, `etymology`, `skip`, and `stop`.
8. Record explicit answers through spaced repetition.
9. Generate an academic-style passage from priority words.
10. Save progress, session summary, and a privacy-safe performance report.

## Tool Calling

LexiPilot exposes OpenAI-compatible function tools:

- `get_profile_summary`
- `get_due_words`
- `get_missed_words`
- `get_new_words`
- `get_word_details`
- `record_answer`
- `lookup_etymology`
- `generate_practice_story`
- `get_words_by_page`
- `save_session_summary`

The model loop preserves `tool_choice="auto"` and supports parallel tool calls.

## Adaptive Session Planning

The deterministic CLI planner estimates a reasonable session size from the user's stated time limit, prioritizes due and frequently missed words, and adds new words only when there is space. The OpenAI-compatible model loop can also use the same tools when a dedicated endpoint is configured.

## Long-Term Learner Memory

Progress remains in the existing `.vocab_progress/<profile>/progress.json` format. LexiPilot does not rename profile directories or migrate progress files. Session records are concise JSONL entries and do not include complete prompts, full model responses, credentials, or complete progress files.

## Spaced Repetition

`record_answer` reuses the existing `vocab_trainer.py` spaced-repetition behavior. Correct answers advance the review stage, wrong answers reset the stage, due dates are recalculated, daily stats are updated, and progress is saved atomically.

## Personalized Practice Generation

LexiPilot tries the configured dedicated Radeon endpoint first for practice generation. If no endpoint is configured or the request fails, it falls back to the existing deterministic local practice generator so smoke tests and demos without a model remain reliable.

## Terminal Presentation

The CLI uses a small semantic color system for demo readability:

- `[PLAN]`, `[TOOL]`, `[SELECTED]`, `[ANSWER]`, `[ADAPT]`, `[GENERATE]`, `[SAVED]`, `[STATUS]`, and `[ERROR]` labels are color-coded by meaning.
- Vocabulary cards highlight the target word, phonetics, part of speech, and answer choices.
- Generated English passages highlight target vocabulary words, including simple inflections such as `abhorred` for `abhor`.
- Chinese translations highlight matched target meanings such as `痛恨` and `憎恶` when those phrases appear.

Color is applied only at terminal rendering time. Saved progress, practice JSON, session summaries, and performance reports remain plain text without ANSI codes.

Controls:

```bash
python3 lexipilot.py --no-color
NO_COLOR=1 python3 lexipilot.py
FORCE_COLOR=1 python3 lexipilot.py
```

## Architecture

`lexipilot.py`  
CLI and interactive learning session.

`lexipilot_core.py`  
Model client, Agent loop, session planning, adaptation, and performance reporting orchestration.

`lexipilot_tools.py`  
Vocabulary, learner-memory, review, answer, etymology, material-generation, configuration, and report tools.

`vocab_trainer.py`  
Existing PDF parsing, spaced repetition, persistence, statistics, etymology, story generation, and legacy CLI.

## AMD Radeon Deployment

Serve Qwen on a dedicated AMD Radeon Cloud GPU instance:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

The model is self-hosted on a dedicated AMD Radeon Cloud GPU instance. Learning tools and progress storage run in the user-controlled LexiPilot environment. Relevant learning context is sent to the dedicated endpoint for model-assisted planning or practice generation. Credentials remain outside version control.

## Privacy Design

LexiPilot never prints API keys and never stores API keys in performance reports. It also avoids storing complete prompts, complete responses, full progress files, full PDF text, or private environment-variable values. Vocabulary definitions and imported text are treated as untrusted data and cannot override Agent rules.

## Performance Reporting

Reports are written atomically under `performance_reports/` when enabled. Reports include model name, endpoint type, total duration, model request count and durations, token usage when available, tool-call count, individual tool durations, story-generation duration, and final session state.

Timing fields separate wall time from active processing:

- `session_wall_seconds`: full session duration.
- `user_interaction_wait_seconds`: time spent waiting for terminal input.
- `active_system_seconds`: wall time minus user wait time.
- `model_request_seconds`: sum of model request durations.
- `tool_execution_seconds`: sum of tool durations.
- `story_generation_seconds`: practice-generation duration.
- `planning_seconds` and `finalization_seconds`: focused Agent workflow timings.

Some timing fields overlap by design. For example, story generation includes model request time when the dedicated endpoint is used.

Inspect a safe summary:

```bash
python3 scripts/show_latest_performance.py
```

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Configuration

Create a local `.env` or point LexiPilot at an existing env file:

```bash
python3 lexipilot.py \
  --profile demo_alice \
  --env-file ../aiagent/.env \
  --debug
```

Configuration precedence:

1. Existing process environment variables
2. Explicit `--env-file` or `LEXIPILOT_ENV_FILE`
3. Local `.env`
4. `../aiagent/.env` fallback when required Radeon values are missing
5. Safe defaults

Required keys:

```text
RADEON_API_KEY
RADEON_BASE_URL
RADEON_MODEL
ENDPOINT_TYPE
QWEN_ENABLE_THINKING
PERFORMANCE_REPORTS_ENABLED
```

For `ENDPOINT_TYPE=dedicated` and `QWEN_ENABLE_THINKING=false`, LexiPilot sends:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

This dedicated-vLLM field is not sent to shared endpoints.

## Demo

Verify the Radeon endpoint:

```bash
python3 scripts/test_radeon_endpoint.py \
  --env-file ../aiagent/.env
```

Run the CLI:

```bash
python3 lexipilot.py \
  --profile default \
  --env-file ../aiagent/.env \
  --debug
```

For a write-enabled demo using the real long-term learner profile, create a backup first:

```bash
python3 scripts/backup_default_profile.py
```

Or let the CLI create one before study:

```bash
FORCE_COLOR=1 python3 lexipilot.py \
  --profile default \
  --env-file ../aiagent/.env \
  --backup-profile \
  --debug
```

Restore if needed:

```bash
python3 scripts/restore_default_profile.py \
  --backup .vocab_progress_backups/default_<timestamp>
```

Prompt:

```text
I have 15 minutes today. Review the words due today and the words I miss most
often. Adapt the session to my answers, explain difficult word origins when
requested, and create a short academic-style passage using the words I still
struggle with.
```

Answer cards with:

```text
y
n
etymology
skip
stop
```

## Testing

Compile:

```bash
python3 -m py_compile \
  vocab_trainer.py \
  lexipilot.py \
  lexipilot_core.py \
  lexipilot_tools.py \
  console_theme.py \
  scripts/smoke_lexipilot.py \
  scripts/test_radeon_endpoint.py \
  scripts/show_latest_performance.py \
  scripts/backup_default_profile.py \
  scripts/restore_default_profile.py
```

Run tests:

```bash
python3 -m pytest -q
```

Run the model-free smoke test:

```bash
python3 scripts/smoke_lexipilot.py
```

## Limitations

The MVP is intentionally CLI-first and does not include a GUI. Etymonline lookup requires network access when used interactively. The deterministic planner is used for the main interactive CLI flow; the OpenAI-compatible model tool loop is available with `--model-loop`.

## Roadmap

- Improve adaptive explanations for repeated misses.
- Add richer local analytics over long-term learner memory.
- Add optional export of practice materials.
- Add a GUI after the CLI demo path is stable.
