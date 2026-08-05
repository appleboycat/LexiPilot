# Radeon Evidence

This file separates evidence LexiPilot can verify automatically from evidence that must be captured manually from the Radeon Cloud environment.

## Automatically Verifiable Evidence

- `scripts/test_radeon_endpoint.py` verifies a basic OpenAI-compatible completion.
- `scripts/test_radeon_endpoint.py` verifies structured Tool Calling with a diagnostic tool.
- `scripts/benchmark_thinking.py` records benchmark timestamps, model name, endpoint type, thinking mode, token counts, latencies, and validation results.
- LexiPilot performance reports record model name, endpoint type, model request durations, tool durations, and final session state.
- The dedicated request behavior is verifiable in code: for `ENDPOINT_TYPE=dedicated` and `QWEN_ENABLE_THINKING=false`, LexiPilot sends `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

These artifacts do not include API keys, authorization headers, private endpoint URLs, full learner profiles, full prompts, or full model responses.

## Manually Captured Evidence

Capture screenshots or logs from the actual Radeon Cloud environment:

1. Radeon Cloud instance page showing the dedicated instance.
2. GPU model shown by the platform.
3. ROCm/vLLM image or deployment configuration.
4. Serve command or deployment configuration:

   ```bash
   vllm serve Qwen/Qwen3-8B \
     --host 0.0.0.0 \
     --port 8000 \
     --enable-auto-tool-choice \
     --tool-call-parser hermes \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.85
   ```

5. Instance status showing ready/running.
6. vLLM startup log, when available.
7. GPU activity or memory evidence, when available.
8. Terminal output showing `PASS Radeon endpoint verification`.
9. A real LexiPilot Agent session using the `default` profile.
10. The final non-mock benchmark summary.

Optional commands when shell access to the Radeon instance exists:

```bash
rocm-smi
rocminfo
```

Do not fabricate screenshots, command output, GPU names, utilization, or memory numbers.

