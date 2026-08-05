# LexiPilot

**LexiPilot - A Private Adaptive Vocabulary Learning Agent** is a self-hosted learning agent that analyzes vocabulary history, plans personalized study sessions, invokes learner-memory tools, adapts to mistakes, and generates targeted academic vocabulary practice with a model served from a dedicated AMD Radeon Cloud GPU instance.

## AMD Radeon Hackathon 2026 Submission

- **Track:** Track 2 - Development & Local Deployment of Private AI Agents
- **Team:** `sheepdog`
- **Members:** `appleboycat`, `du-du-lu`
- [Submission Index](submission/README.md)
- [Project Specification](submission/LexiPilot_Project_Specification.pdf)
- [Presentation](submission/LexiPilot_Presentation.pdf)
- [Demo Video](submission/video/VIDEO_LINK.md)
- [Agent Architecture](submission/architecture/lexipilot_architecture.svg)
- [Radeon Evidence](submission/evidence/evidence_manifest.md)
- [Reproduction Guide](#quickstart)
- [Benchmark Results](docs/benchmark_results/thinking_benchmark.md)
- [Final Submission Checklist](submission/FINAL_SUBMISSION_CHECKLIST.md)

## Problem

Vocabulary learners often have word lists, review history, and missed-word logs, but still need to manually decide what to study next. Static review flows also struggle to adapt when the learner has only a few minutes or repeatedly misses the same words.

## Target Users

LexiPilot is for learners who want private, adaptive academic vocabulary review on a user-controlled machine or cloud workspace, especially learners who already maintain PDF-derived vocabulary data and long-term progress history.

## Solution

LexiPilot keeps the existing `vocab_trainer.py` vocabulary engine and adds a hybrid Agent layer around it. Qwen3-8B selects read-only learner-state tools and returns a validated study plan. A deterministic controller then presents cards, records explicit answers, applies spaced repetition, adapts after mistakes, and finalizes the session exactly once.

## Quickstart

A fresh clone can run with the committed sample vocabulary data and a generated synthetic profile. The private source PDF and real learner history are not required.

```bash
git clone https://github.com/appleboycat/LexiPilot.git
cd LexiPilot

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

cp .env.example .env
# Fill in the Radeon endpoint configuration.

python3 scripts/setup_demo_data.py

FORCE_COLOR=1 python3 lexipilot.py \
  --demo \
  --env-file .env \
  --debug
```

## Why This Is an Agent

LexiPilot accepts a natural-language learning objective and uses Qwen3-8B Tool Calling to inspect real learner state and propose a structured plan. The application validates that every selected word came from tool results, then hands execution to a deterministic controller. It is more than a chat wrapper because model decisions are grounded in tools and become an executable session backed by persistent learner memory.

## Agent Workflow

```text
Natural-language learning goal
→ Qwen3-8B model planning
→ model-selected read-only Tool Calling
→ validated structured study plan
→ deterministic interactive session controller
→ explicit answer recording
→ spaced-repetition persistence
→ model-backed practice generation
```

During planning, the model can inspect the profile summary, due reviews, missed words, new words, and individual word details. It cannot record answers or save a session. Write tools become available only through deterministic controller actions after explicit learner input.

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

Planning exposes only `get_profile_summary`, `get_due_words`, `get_missed_words`, `get_new_words`, and `get_word_details`. On a dedicated endpoint, the first planning request uses `tool_choice="required"` so Qwen immediately selects one or more of those read-only tools instead of returning preliminary prose; parallel tool calls remain supported. General Agent requests and shared endpoints keep `tool_choice="auto"`. `record_answer` and `save_session_summary` are never exposed during planning.

## Adaptive Session Planning

With a configured dedicated endpoint, model-driven planning is the default. The returned JSON plan is accepted only when all words came from read-only tool results, due and missed words are prioritized, no word is invented, and the requested time and practical word limit are respected.

If the endpoint times out, authentication fails, Tool Calling is malformed, JSON is invalid, or the plan fails validation, LexiPilot prints a concise warning and immediately uses the existing deterministic planner. Use `--deterministic` to bypass model planning explicitly. `--model-loop` remains only as a deprecated compatibility flag and is no longer required.

`run.sh` keeps hybrid model planning as the default. For an immediate local plan during routine study, use:

```bash
LEXIPILOT_DETERMINISTIC=1 ./run.sh
```

This skips only model planning. Answer recording, spaced repetition, persistence, and practice generation continue to use the normal controller and configured generation path.

## Long-Term Learner Memory

Progress remains in the existing `.vocab_progress/<profile>/progress.json` format. LexiPilot does not rename profile directories or migrate progress files. Session records are concise JSONL entries and do not include complete prompts, full model responses, credentials, or complete progress files.

## Reproducible Sample Data

`examples/sample_vocab_index.json` contains 40 deterministic demo entries with independently written concise definitions and empty `source_text` fields. It contains no personal progress, credentials, or copied PDF lines. `scripts/setup_demo_data.py` creates an ignored synthetic profile with 12 started words, seven reviews due relative to the current date, four historically missed words, unlearned vocabulary, and 35 days of reproducible seeded activity.

Use explicit paths when integrating another permitted vocabulary source:

```bash
python3 lexipilot.py \
  --index-file examples/sample_vocab_index.json \
  --progress-root .demo_data/profiles \
  --profile demo \
  --env-file .env \
  --debug
```

Validate an index without printing its vocabulary content:

```bash
python3 scripts/validate_vocab_index.py examples/sample_vocab_index.json
```

The private source PDF, generated CSV files, complete local `.vocab_index.json`, real profiles, and backups remain excluded from Git. The full local index is not treated as redistributable project data.

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

Safe profile views:

```text
/status
/activity
/activity 7
/activity 90
```

`/activity` renders a framed, color-depth study-intensity heatmap from
35 days of aggregated daily statistics by default. It does not print individual
words or the complete profile.

## Architecture

`lexipilot.py`  
CLI, path selection, and interactive learning session.

`lexipilot_core.py`  
Read-only model Tool Calling, strict plan validation, deterministic session control, adaptation, and performance reporting orchestration.

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

Some timing fields overlap by design. Model request time is included in tool time when a tool invokes the model. Story generation time is a subset of tool time. Aggregate timing fields must not be summed.

Use `non_overlapping_timing_breakdown` for top-level additive timing:

- `user_interaction_wait_seconds`
- `model_api_execution_seconds`
- `local_non_model_processing_seconds`

Inspect a safe summary:

```bash
python3 scripts/show_latest_performance.py
```

## Radeon Inference Optimization

LexiPilot includes a repeatable benchmark for comparing Qwen thinking mode on the same dedicated Radeon endpoint:

- Baseline: `QWEN_ENABLE_THINKING=true`
- Optimized demo setting: `QWEN_ENABLE_THINKING=false`
- Model: `Qwen/Qwen3-8B`
- Backend: OpenAI-compatible vLLM endpoint on dedicated AMD Radeon Cloud
- Workload A: Agent planning with structured Tool Calling
- Workload B: bilingual academic practice generation

Run a model-free benchmark pipeline check:

```bash
python3 scripts/benchmark_thinking.py \
  --mock \
  --warmups 1 \
  --runs 5
```

Run the real benchmark only after endpoint verification passes:

```bash
python3 scripts/test_radeon_endpoint.py \
  --env-file .env

FORCE_COLOR=1 python3 scripts/benchmark_thinking.py \
  --env-file .env \
  --warmups 1 \
  --runs 5
```

Reports are written under `benchmark_reports/thinking_<timestamp>/`:

- `raw_results.json`
- `summary.json`
- `summary.md`

Use only a non-mock report with `benchmark_complete=true` as submission performance evidence. Mock reports are labeled `mock_data=true` and `hardware_result=false`.

### Measured Result

The final benchmark ran on August 5, 2026 against the dedicated endpoint with one warm-up and five measured requests per mode and workload. Both modes used `temperature=0`, `max_tokens=700`, a 90-second timeout, identical prompts and tools, and alternating measured order.

| Workload | Metric | Thinking Enabled | Thinking Disabled |
|---|---|---:|---:|
| Agent planning | Median latency | 13.7371 s | 13.7773 s |
| Agent planning | P95 latency | 14.2909 s | 14.0693 s |
| Agent planning | Client-observed completion tokens/s | 22.0571 | 21.9927 |
| Agent planning | Structured Tool Calling success | 100% | 100% |
| Bilingual generation | Median latency | 7.0341 s | 6.5433 s |
| Bilingual generation | P95 latency | 11.4237 s | 13.1206 s |
| Bilingual generation | Client-observed completion tokens/s | 17.9127 | 19.2563 |
| Bilingual generation | Task validation success | 100% | 100% |

Disabling thinking produced no clear planning improvement in this sample: median planning latency regressed by 0.29%, while Tool Calling reliability remained 100% in both modes. For bilingual generation, it reduced median latency by 6.98% and increased median client-observed completion tokens/s by 7.50%; completion-token counts were unchanged. Because the sample contains only five measured requests per group and the disabled-mode generation P95 was higher, these results should be treated as observed client-level behavior rather than a hardware-level speedup. LexiPilot uses `QWEN_ENABLE_THINKING=false` for the final demo because it preserved validation reliability and improved the median generation result.

Source report: [docs/benchmark_results/thinking_benchmark.md](docs/benchmark_results/thinking_benchmark.md).

## Installation

Requirements:

- Python 3.10 or newer (validated with Python 3.12)
- `pip` and `venv`
- network access only for endpoint-backed planning, practice generation, and
  interactive etymology
- optional Radeon deployment tools: ROCm, vLLM, and a compatible AMD Radeon GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Optional submission-document rebuild dependencies:

```bash
python3 -m pip install -r requirements-docs.txt
```

## Configuration

Create a local `.env` or point LexiPilot at an existing env file:

```bash
python3 lexipilot.py \
  --demo \
  --env-file .env \
  --debug
```

Configuration precedence:

1. Existing process environment variables
2. Explicit `--env-file` or `LEXIPILOT_ENV_FILE`
3. Local `.env`
4. Optional sibling `../aiagent/.env` fallback when required Radeon values are missing
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

For local development only, an existing sibling configuration can still be selected explicitly with `--env-file ../aiagent/.env`. Public quickstart and demo commands use the repository-local `.env`.

## Demo

Verify the Radeon endpoint:

```bash
python3 scripts/test_radeon_endpoint.py \
  --env-file .env
```

Run the reproducible sample-data hybrid Agent:

```bash
python3 scripts/setup_demo_data.py

FORCE_COLOR=1 python3 lexipilot.py \
  --demo \
  --env-file .env \
  --debug
```

Run the sample-data deterministic offline path without an API key:

```bash
python3 scripts/setup_demo_data.py
python3 lexipilot.py \
  --demo \
  --deterministic \
  --no-color
```

The debug timeline distinguishes model and controller responsibility:

```text
[AGENT] Requesting a model-generated study plan
[MODEL TOOL] get_profile_summary
[MODEL TOOL] get_due_words
[MODEL TOOL] get_missed_words
[MODEL PLAN] 6 reviews, 1 new word
[CONTROLLER] Starting the interactive study session
```

For a write-enabled demo using an existing real long-term learner profile, create a backup first:

```bash
python3 scripts/backup_default_profile.py
```

Or let the CLI create one before study:

```bash
FORCE_COLOR=1 python3 lexipilot.py \
  --profile toefl2026 \
  --env-file .env \
  --backup-profile \
  --debug
```

Restore if needed:

```bash
python3 scripts/restore_default_profile.py \
  --backup .vocab_progress_backups/toefl2026_<timestamp>
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
  scripts/smoke_fresh_clone.py \
  scripts/setup_demo_data.py \
  scripts/validate_vocab_index.py \
  scripts/test_radeon_endpoint.py \
  scripts/benchmark_thinking.py \
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
python3 scripts/smoke_fresh_clone.py
python3 scripts/validate_vocab_index.py examples/sample_vocab_index.json
```

Reproduce the report-generation pipeline without a model:

```bash
python3 scripts/benchmark_thinking.py \
  --mock \
  --warmups 1 \
  --runs 5
```

## Troubleshooting

### Model planning falls back

Run:

```bash
python3 scripts/test_radeon_endpoint.py --env-file .env
```

Confirm the model name, endpoint readiness, and vLLM startup flags
`--enable-auto-tool-choice --tool-call-parser hermes`. LexiPilot intentionally
falls back when Tool Calling, JSON, or plan validation fails.

### Base URL errors

Set `RADEON_BASE_URL` to the OpenAI-compatible server root ending in `/v1`.
LexiPilot normalizes the path and avoids `/v1/v1`. The CLI never prints the
private URL.

### Missing demo profile

Run:

```bash
python3 scripts/setup_demo_data.py --force
```

### Terminal color or symbol problems

Use:

```bash
NO_COLOR=1 python3 lexipilot.py --demo --deterministic
```

Status progress bars use ASCII characters. Activity levels retain distinct
ASCII markers when color is disabled.

### No private PDF or complete index

Use `--demo`. The public sample index and synthetic profile are sufficient for
tests and the evaluator workflow.

## Limitations

The MVP is intentionally CLI-first and does not include a GUI. Etymonline lookup requires network access when used interactively. Model planning requires a compatible endpoint with structured Tool Calling; deterministic fallback keeps the session usable when that dependency is unavailable. The committed sample index is intentionally small and does not replace the user's full vocabulary source.

## Roadmap

- Improve adaptive explanations for repeated misses.
- Add richer local analytics over long-term learner memory.
- Add optional export of practice materials.
- Add a GUI after the CLI demo path is stable.
