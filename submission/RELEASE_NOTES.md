# LexiPilot Submission v1.0 Candidate

**Validation date:** `<FILLED BY FINAL VALIDATION>`  
**Commit tested:** `<FILLED BY FINAL VALIDATION>`  
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

Final results are populated only after the independent fresh-clone validation:

| Check | Result |
|---|---|
| Python compilation | `<PENDING>` |
| Automated tests | `<PENDING>` |
| Model-free smoke test | `<PENDING>` |
| Fresh-clone smoke test | `<PENDING>` |
| Sample index validation | `<PENDING>` |
| Mock benchmark | `<PENDING>` |
| Dedicated endpoint verification | `<PENDING>` |
| Markdown link check | `<PENDING>` |
| Credential and private-path scan | `<PENDING>` |
| Independent temporary clone | `<PENDING>` |

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
