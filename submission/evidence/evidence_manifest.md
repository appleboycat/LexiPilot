# Radeon Evidence Manifest

The following artifacts must be captured from the actual deployment. Fields
marked "Capture during final Radeon demo" are intentionally not populated with
guessed values.

| # | Suggested filename | Required visible evidence | Redact before use | Place in submission |
|---:|---|---|---|---|
| 1 | `01_radeon_cloud_instance.png` | Radeon Cloud instance page, running/ready state, capture date | Account ID, private URL/IP, billing details | Specification Section 9; PPT Slide 6 |
| 2 | `02_gpu_model.png` | GPU model reported by the platform | Account and instance identifiers not needed for verification | Specification Section 9; PPT Slide 6 |
| 3 | `03_rocminfo.png` | Relevant `rocminfo` agent/device lines | Hostname, usernames, unrelated devices | Evidence appendix; optional PPT Slide 6 |
| 4 | `04_rocm_smi.png` | `rocm-smi` GPU identity, memory, and activity during inference | Hostname and private paths | Specification Section 9; PPT Slide 6 |
| 5 | `05_rocm_version.png` | Installed ROCm version from the actual instance | Private repository/package credentials | Specification Section 9 notes |
| 6 | `06_vllm_version.png` | `python -m pip show vllm` or `vllm --version` | Local paths if sensitive | Specification Section 9 notes |
| 7 | `07_qwen_vllm_startup.png` | Qwen3-8B model name, vLLM startup readiness, tool parser enabled | API key, bearer token, private endpoint URL/IP | Specification Section 9; PPT Slide 6 |
| 8 | `08_endpoint_verification.png` | Three PASS lines from `scripts/test_radeon_endpoint.py` | Shell history containing credentials or endpoint URL | PPT Slide 6; video |
| 9 | `09_model_tool_calling.png` | `[AGENT]` and actual `[MODEL TOOL]` lines | Real learner words if considered sensitive | PPT Slide 4 or 7; video |
| 10 | `10_validated_model_plan.png` | `[MODEL PLAN]` plus selected count and reason | Private endpoint URL | PPT Slide 5; video |
| 11 | `11_controller_session.png` | `[CONTROLLER]`, one card, and explicit answer controls | Real profile details if not intended for publication | PPT Slide 3 or 8; video |
| 12 | `12_progress_persistence.png` | `[TOOL] record_answer`, answer result, and `[SAVED]` without file contents | Local home path if displayed | Specification Section 8; video |
| 13 | `13_bilingual_practice.png` | Coherent English and Chinese passage with highlighted target terms | Real learning history; unrelated terminal content | PPT Slide 8; video |
| 14 | `14_benchmark_summary.png` | Non-mock thinking-mode result table and report path | Endpoint URL and credentials | Specification Section 10; PPT Slide 7 |
| 15 | `15_gpu_activity.png` | GPU memory/utilization while a LexiPilot request is active | Account ID, private IP, other tenant details | PPT Slide 6 or 7 |

## Capture During Final Radeon Demo

- GPU model: **Capture during final Radeon demo**
- ROCm version: **Capture during final Radeon demo**
- vLLM version: **Capture during final Radeon demo**
- GPU memory/utilization: **Capture during final Radeon demo**

## Safe Commands

Run these on the Radeon instance only when shell access exists:

```bash
rocm-smi
rocminfo
python3 -m pip show vllm
```

Run this in the LexiPilot terminal:

```bash
python3 scripts/test_radeon_endpoint.py --env-file .env
```

The verification script prints only:

```text
PASS basic completion
PASS tool calling
PASS Radeon endpoint verification
```

Do not capture the contents of `.env`, an Authorization header, the private
endpoint URL, or the source PDF.
