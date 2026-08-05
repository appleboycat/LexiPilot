# Pull Request

## Required Title

```text
Track 2, sheepdog, LexiPilot
```

Team `sheepdog` members: `appleboycat` and `du-du-lu`.

## PR Body

### Project Overview

LexiPilot is a private adaptive vocabulary learning agent that converts a
natural-language study goal into a validated, personalized learning session.
It combines persistent learner memory, Qwen3-8B planning, read-only Tool
Calling, deterministic answer handling, spaced repetition, and mistake-driven
bilingual practice.

Source repository: https://github.com/appleboycat/LexiPilot

### Why It Fits Track 2

LexiPilot is a hybrid private AI Agent rather than a chat wrapper:

```text
learner goal
-> model-selected learner-state tools
-> validated JSON study plan
-> deterministic interactive controller
-> explicit answer recording
-> persistent memory
-> targeted bilingual practice
```

The planning model can observe learner state but cannot write progress.
Controller-owned writes happen only after explicit user answers.

### Agent Architecture

- Qwen3-8B performs structured planning through read-only tools.
- Learner profile, due reviews, missed words, and candidate new words ground the
  plan in real state.
- Strict validation rejects unknown words, malformed JSON, inconsistent plans,
  oversized sessions, fake Tool Calling, and unavailable tools.
- A deterministic controller presents cards, validates commands, applies the
  existing spaced-repetition algorithm, and finalizes idempotently.
- Endpoint or model failures activate a deterministic fallback planner.

Architecture:
https://github.com/appleboycat/LexiPilot/blob/main/submission/architecture/lexipilot_architecture.svg

### AMD Radeon Deployment

- Model: `Qwen/Qwen3-8B`
- Backend: OpenAI-compatible vLLM
- Acceleration environment: ROCm on a dedicated AMD Radeon Cloud GPU instance
- Tool parser: vLLM Hermes parser with automatic Tool Calling
- Final demo setting: `QWEN_ENABLE_THINKING=false`

Learner tools and progress remain in the user-controlled LexiPilot environment.
Only minimum relevant planning or generation context is sent to the dedicated
endpoint.

### Core Capabilities

- natural-language learning objectives;
- persistent learner memory;
- due and high-missed word inspection;
- model-selected parallel read-only tools;
- validated adaptive plans;
- deterministic spaced-repetition writes;
- etymology support;
- current-mistake-first reinforcement;
- coherent bilingual practice with exact target-term highlighting;
- privacy-safe performance reports;
- sample-data and fresh-clone reproduction.

### Demo Video

https://github.com/appleboycat/LexiPilot/blob/main/submission/video/lexi_pilot_en.mp4

Video metadata:
https://github.com/appleboycat/LexiPilot/blob/main/submission/video/VIDEO_LINK.md

### Submission Documents

- Project Specification:
  https://github.com/appleboycat/LexiPilot/blob/main/submission/LexiPilot_Project_Specification.pdf
- Presentation:
  https://github.com/appleboycat/LexiPilot/blob/main/submission/LexiPilot_Presentation.pdf
- Radeon evidence manifest:
  https://github.com/appleboycat/LexiPilot/blob/main/submission/evidence/evidence_manifest.md

### Reproduction

```bash
git clone https://github.com/appleboycat/LexiPilot.git
cd LexiPilot
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
python3 scripts/setup_demo_data.py
python3 -m pytest -q
python3 scripts/smoke_fresh_clone.py
FORCE_COLOR=1 python3 lexipilot.py --demo --env-file .env --debug
```

A model-free deterministic demo is also available:

```bash
python3 lexipilot.py --demo --deterministic --no-color
```

### Test Results

See:
https://github.com/appleboycat/LexiPilot/blob/main/submission/RELEASE_NOTES.md

### Measured Radeon Inference Results

The non-mock benchmark used one warm-up and five measured requests per mode and
workload.

- Planning Tool Calling validation: 100% for thinking enabled and disabled.
- Planning median latency: 13.7371 s enabled; 13.7773 s disabled.
- Generation validation: 100% for both modes.
- Generation median latency: 7.0341 s enabled; 6.5433 s disabled.
- Generation client-observed completion tokens/s: 17.9127 enabled; 19.2563
  disabled.

These are client-observed end-to-end measurements that include network,
endpoint, scheduling, and serving overhead. They are not raw GPU throughput.

### Privacy Design

- `.env`, learner profiles, backups, personal materials, complete local indexes,
  private PDFs, and reports are excluded from Git.
- Planning tools are read-only.
- Credentials and private base URLs are redacted from errors and reports.
- Reports exclude complete prompts, responses, profiles, PDF content, and tool
  output.
- Public reproduction uses synthetic progress and independently written sample
  definitions.

### Limitations

- CLI-first interface; no GUI in this submission.
- Small benchmark sample.
- No fine-tuning or kernel-level optimization claim.
- Dedicated endpoint deployment; a future roadmap item is localhost Radeon
  workstation inference.
- GPU model, ROCm/vLLM versions, and utilization are supplied through manually
  captured deployment evidence rather than inferred by the repository.
