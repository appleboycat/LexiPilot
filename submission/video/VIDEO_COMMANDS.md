# Demo Recording Commands

Run every command from the LexiPilot repository root. Keep `.env` and the
private endpoint out of the visible editor and shell history.

## 1. Preflight

```bash
git status --short
python3 --version
python3 -m pytest -q
python3 scripts/setup_demo_data.py
python3 scripts/smoke_fresh_clone.py
python3 scripts/validate_vocab_index.py examples/sample_vocab_index.json
```

## 2. Verify the Dedicated Endpoint

```bash
python3 scripts/test_radeon_endpoint.py --env-file .env
```

Expected safe output:

```text
PASS basic completion
PASS tool calling
PASS Radeon endpoint verification
```

## 3. Capture Radeon Evidence

Run on the Radeon instance when shell access is available:

```bash
rocm-smi
rocminfo
python3 -m pip show vllm
```

Do not run these locally and present the output as Radeon Cloud evidence.

## 4. Recommended Public Sample Demo

This avoids showing real learner history:

```bash
python3 scripts/setup_demo_data.py --force
FORCE_COLOR=1 python3 lexipilot.py \
  --demo \
  --env-file .env \
  --debug
```

Optional status commands:

```text
/status
/activity 28
```

Enter this learning goal:

```text
I have 15 minutes. Review the words due today and the words I miss most often, then create targeted bilingual practice.
```

During the cards:

```text
y
n
e
stop
```

Ensure the recording shows:

- `[AGENT]`
- `[MODEL TOOL] get_profile_summary`
- `[MODEL TOOL] get_due_words`
- `[MODEL TOOL] get_missed_words`
- `[MODEL PLAN]`
- `[CONTROLLER]`
- one correct answer;
- one incorrect answer;
- one etymology request;
- `[TOOL] record_answer`;
- `[GENERATE]`;
- `[SAVED]`;
- highlighted English and Chinese target terms;
- one completion summary and one performance report.

Exit:

```text
/exit
```

## 5. Optional Real Default-Profile Demo

Back up first:

```bash
python3 scripts/backup_default_profile.py
FORCE_COLOR=1 python3 lexipilot.py \
  --profile default \
  --env-file .env \
  --backup-profile \
  --debug
```

Do not show private profile contents. Restore only when needed:

```bash
python3 scripts/restore_default_profile.py \
  --backup .vocab_progress_backups/default_<timestamp>
```

## 6. Show Safe Performance Results

```bash
python3 scripts/show_latest_performance.py
sed -n '1,120p' docs/benchmark_results/thinking_benchmark.md
```

## 7. Model-Free Fallback Demo

```bash
python3 lexipilot.py --demo --deterministic --no-color
```

This demonstrates deterministic planning and local fallback behavior; do not
present it as Radeon inference.
