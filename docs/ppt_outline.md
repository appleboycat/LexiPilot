# LexiPilot PPT Outline

## Slide 1 — Problem

- Vocabulary apps are static.
- Users forget different words.
- Fixed study plans ignore real progress and available time.

## Slide 2 — LexiPilot Solution

- Natural-language study goals.
- Adaptive plans.
- Long-term learner memory.
- Spaced repetition.
- Personalized bilingual practice material.

## Slide 3 — Why This Is an Agent

```text
User Goal
→ Agent Planning
→ Tool Calling
→ Learner Memory
→ Answer-Based Adaptation
→ Practice Generation
→ State Persistence
```

## Slide 4 — Architecture and Privacy

```text
LexiPilot CLI
→ Agent Core
→ Vocabulary Tools
→ Learner Progress

Agent Core
↔ Dedicated Qwen3-8B
   vLLM + ROCm + AMD Radeon
```

Privacy boundary: vocabulary tools and progress storage run in the user-controlled environment; relevant task context is sent to the dedicated Radeon endpoint; credentials remain outside version control.

## Slide 5 — Radeon Optimization and Performance

Model: `Qwen/Qwen3-8B`

Backend: OpenAI-compatible vLLM + ROCm on dedicated AMD Radeon Cloud.

Method: one warm-up and five measured requests per mode and workload; identical request settings; alternating measured order.

| Workload | Metric | Thinking Enabled | Thinking Disabled |
|---|---|---:|---:|
| Agent planning | Median latency | 13.7371 s | 13.7773 s |
| Agent planning | Tool Calling success | 100% | 100% |
| Bilingual generation | Median latency | 7.0341 s | 6.5433 s |
| Bilingual generation | Completion tokens/s | 17.9127 | 19.2563 |
| Bilingual generation | Validation success | 100% | 100% |

Observed result:

- Planning latency: 0.29% regression; no clear improvement.
- Generation median latency: 6.98% reduction.
- Generation client-observed completion tokens/s: 7.50% increase.
- Completion-token count: unchanged.
- Final demo setting: `QWEN_ENABLE_THINKING=false`.

Measurement disclaimer: results include client, network, endpoint, scheduling, and serving overhead. They are not raw GPU kernel throughput.

Manual visual: insert a screenshot of `benchmark_reports/thinking_20260805_230647/summary.md` or the terminal benchmark summary.

## Slide 6 — Demo and Results

- Real `default` learner profile.
- Due/missed-word inspection.
- Correct and incorrect answers.
- Etymology lookup.
- Adaptive priority-word practice.
- Highlighted English and Chinese terms.
- Saved progress and performance report.
- Endpoint verification, benchmark, tests, and smoke-test results.

Manual screenshots to insert: Radeon Cloud instance, vLLM deployment, endpoint verification, LexiPilot demo, benchmark summary.
