# LexiPilot Submission v1.0 Candidate

**Validation date:** August 6, 2026

**Base commit tested:** `bd2a5a4` plus the current uncommitted submission worktree

**Recommended tag:** `submission-v1.0` (not created)

## Submission Scope

- Hybrid Qwen3-8B planning with model-selected read-only Tool Calling.
- Strict structured-plan validation and deterministic fallback.
- Deterministic interactive controller and explicit spaced-repetition writes.
- Idempotent finalization and privacy-safe performance reporting.
- Coherent bilingual practice with exact English/Chinese target highlighting.
- Public sample vocabulary index and synthetic demo profile.
- Project Specification, architecture diagram, presentation, evidence manifest,
  video script, official-repository package, and PR body.

## Validation Results

| Check | Result |
|---|---|
| Python compilation | PASS |
| Automated tests | PASS - 127 tests |
| Model-free smoke test | PASS |
| Fresh-clone smoke test | PASS |
| Sample index validation | PASS - 40 entries, 40 unique words |
| Mock benchmark report pipeline | PASS - explicitly marked as mock |
| Dedicated endpoint verification | PASS - completion and structured Tool Calling |
| Markdown relative-link check | PASS - 27 links |
| Public-text credential/private-path scan | PASS - 51 files |
| Submission attachment validation | PASS - 21 required files |
| Independent temporary clone | PASS - dependency install, tests, smoke, and submission checks |

The independent validation cloned the repository into a temporary
directory, applied the complete current worktree diff, installed
`requirements.txt` into a new virtual environment, generated only synthetic
demo data, and ran the full offline validation. The temporary directory was
removed afterward.

The local private vocabulary index was also validated in place without printing
its contents: 4,250 entries, 4,250 unique words, sequences 1 through 4,297, and
160 source pages. It remains excluded from the submission.

## Real Benchmark on Dedicated Endpoint

Source: `docs/benchmark_results/thinking_benchmark.json`

- Date: August 5, 2026
- Model: `Qwen/Qwen3-8B`
- Backend: OpenAI-compatible vLLM dedicated endpoint
- Warm-ups: 1 per mode and workload
- Measured runs: 5 per mode and workload
- Mock data: false
- Benchmark complete: true
- Planning structured Tool Calling success: 100% in both modes
- Bilingual generation validation success: 100% in both modes

The report does not identify the GPU model and does not represent raw GPU
kernel throughput.

## Deliberately Excluded

- `.env` and credentials;
- private endpoint URL;
- real learner profiles and backups;
- generated personal practice materials;
- complete local vocabulary index;
- private source PDF and derived CSV files;
- model files;
- unsanitized performance and benchmark report directories.

## Known Limitations

- Final video URL and participant/team name remain manual placeholders.
- Radeon Cloud screenshots and exact GPU/software versions require manual
  capture.
- The product is CLI-first.
- Etymology requires network access.
