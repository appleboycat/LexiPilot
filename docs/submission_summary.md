# LexiPilot — A Private Adaptive Vocabulary Learning Agent

## Problem

Static vocabulary applications do not adapt to a learner's real review history, repeated mistakes, available time, and preferred learning context.

## Solution

LexiPilot lets the learner describe a study goal in natural language. The Agent reads long-term learner memory, checks due reviews and frequently missed words, plans an adaptive session, records explicit answers through spaced repetition, supports etymology lookup, and generates mistake-driven bilingual practice material.

The final demo uses a dedicated AMD Radeon Cloud endpoint serving `Qwen/Qwen3-8B` through an OpenAI-compatible vLLM API.

## Why It Is an Agent

LexiPilot follows an autonomous tool-using loop:

```text
goal understanding
→ learner-state inspection
→ tool selection
→ session planning
→ answer-driven adaptation
→ content generation
→ persistent state update
```

It is grounded in real tools and persistent learner state rather than static prompts.

## Radeon Optimization

LexiPilot compares Qwen thinking mode on the same dedicated Radeon endpoint:

- Baseline: `QWEN_ENABLE_THINKING=true`
- Optimized demo setting: `QWEN_ENABLE_THINKING=false`

The benchmark records median latency, token counts, client-observed completion tokens/s, Tool Calling success, and task validation success for Agent planning and bilingual practice generation.

The final non-mock benchmark used one warm-up and five measured requests per mode and workload, with identical request settings and alternating measured order.

| Workload | Metric | Thinking Enabled | Thinking Disabled |
|---|---|---:|---:|
| Agent planning | Median latency | 13.7371 s | 13.7773 s |
| Agent planning | Structured Tool Calling success | 100% | 100% |
| Bilingual generation | Median latency | 7.0341 s | 6.5433 s |
| Bilingual generation | Client-observed completion tokens/s | 17.9127 | 19.2563 |
| Bilingual generation | Validation success | 100% | 100% |

Disabling thinking did not improve planning latency in this sample: the median regressed by 0.29%, with identical Tool Calling reliability. It reduced median bilingual-generation latency by 6.98% and increased median client-observed completion tokens/s by 7.50%, with no change in median completion tokens. LexiPilot therefore uses `QWEN_ENABLE_THINKING=false` for the final demo.

These client-observed measurements include network, endpoint, scheduling, and serving overhead. They are not raw GPU or kernel throughput. Each aggregate contains five measured requests, so the result describes this sample rather than a guaranteed performance improvement.

## Privacy

Learner tools and progress storage remain in the user-controlled LexiPilot environment. Only relevant task context is sent to the dedicated model endpoint. Credentials stay outside version control, and real profile files are excluded from Git.

LexiPilot does not claim that no user data leaves the local computer.
