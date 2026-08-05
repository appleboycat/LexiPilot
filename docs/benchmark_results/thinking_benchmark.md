# LexiPilot Radeon Inference Benchmark

## Environment

- Model: Qwen/Qwen3-8B
- Endpoint type: dedicated
- Backend: OpenAI-compatible vLLM endpoint
- Benchmark date: 2026-08-05T15:11:22+00:00
- Warm-up runs: 1
- Measured runs: 5
- Mock data: False
- Measurement scope: Measurements were collected by the LexiPilot client. Latency and client-observed completion tokens/s include client, network, endpoint, scheduling, and serving overhead. They are not raw GPU kernel throughput.

## Agent Planning and Tool Calling

| Metric | Thinking Enabled | Thinking Disabled |
|---|---:|---:|
| Successful runs | 5 | 5 |
| Tool-call success rate | 1.0 | 1.0 |
| Validation success rate | 1.0 | 1.0 |
| Median latency | 13.7371 | 13.7773 |
| P95 latency | 14.2909 | 14.0693 |
| Median completion tokens | 303.0 | 303.0 |
| Client-observed completion tokens/s | 22.0571 | 21.9927 |

## Bilingual Practice Generation

| Metric | Thinking Enabled | Thinking Disabled |
|---|---:|---:|
| Successful runs | 5 | 5 |
| Validation success rate | 1.0 | 1.0 |
| Median latency | 7.0341 | 6.5433 |
| P95 latency | 11.4237 | 13.1206 |
| Median completion tokens | 126.0 | 126.0 |
| Client-observed completion tokens/s | 17.9127 | 19.2563 |

## Observed Optimization

- Planning latency: -0.29%
- Generation latency: 6.98%
- Planning completion-token change: 0.0% reduction
- Generation completion-token change: 0.0% reduction
- Recommended demo setting: QWEN_ENABLE_THINKING=false when validation reliability is acceptable.

## Measurement Note

Measurements were collected by the LexiPilot client. Latency and client-observed completion tokens/s include client, network, endpoint, scheduling, and serving overhead. They are not raw GPU kernel throughput.
