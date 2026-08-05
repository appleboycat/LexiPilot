# LexiPilot Demo Video Script

Target duration: approximately 4 minutes. The narration uses short sentences
for a non-native English speaker. Replace the participant placeholder before
recording.

## 0:00-0:25 - Problem

**Screen:** Title slide, then the problem slide.

**Narration:**

Hello. We are appleboycat and du-du-lu from team sheepdog. This is LexiPilot, a
private adaptive vocabulary learning agent for AMD Radeon Hackathon Track 2.

Most vocabulary tools show static lists. They do not understand which words I
miss repeatedly, which reviews are due, or how much time I have today.
LexiPilot turns that history into a safe, personalized learning session.

## 0:25-0:50 - Architecture

**Screen:** Architecture slide or `submission/architecture/lexipilot_architecture.png`.

**Narration:**

LexiPilot uses a hybrid Agent architecture. Qwen3-8B selects read-only tools to
inspect learner state and proposes a structured study plan. Local code validates
every word and every field. A deterministic controller handles answers,
spaced repetition, and progress writes. The planning model cannot modify the
profile.

If the endpoint or plan fails, LexiPilot falls back to a deterministic planner,
so the study workflow remains available.

## 0:50-1:15 - Radeon Environment

**Screen:** Radeon Cloud instance, redacted deployment view, then endpoint PASS.

**Narration:**

Qwen3-8B is served by vLLM with ROCm on a dedicated AMD Radeon Cloud instance.
The endpoint supports OpenAI-compatible completion and structured Tool Calling.
This verification checks both behaviors without printing credentials.

The exact GPU, ROCm, and vLLM versions shown here are captured from the running
instance, not inferred by the application.

## 1:15-2:45 - Live Agent Demo

**Screen:** Terminal with `FORCE_COLOR=1`; run the sample or backed-up `toefl2026`
profile demo.

**Narration:**

First, LexiPilot shows safe profile aggregates. Matching Done and InProgress
bars show started-word coverage and the current vocabulary cursor. The
underlying counts remain visible. I can also enter slash activity to see recent
study intensity without exposing individual history.

Now I give one natural-language goal:

I have fifteen minutes. Review the words due today and the words I miss most
often, then create targeted bilingual practice.

The Agent asks Qwen to create a plan. These MODEL TOOL lines are tools actually
selected by the model. The returned plan is validated before the controller
starts.

I answer one word correctly. I answer another word incorrectly. Only these
explicit answers update spaced repetition. For this difficult word, I enter
`e` to request etymology. Then I stop the card session.

The incorrect word becomes the first practice priority. LexiPilot creates one
coherent academic passage and a Chinese translation. The English target words
and their exact Chinese meanings are highlighted. Progress, one session record,
and one performance report are saved.

## 2:45-3:25 - Privacy and Reliability

**Screen:** Safe design slide, tests, and fresh-clone smoke output.

**Narration:**

The model sees only minimum task context. Planning tools are read-only.
Unknown words, fake tool calls, invalid JSON, and oversized plans are rejected.
The controller owns all writes and finalizes only once.

API keys, private endpoint URLs, full profiles, prompts, responses, and PDF text
are not stored in reports. The public repository includes a sanitized
forty-word index and synthetic profile, so evaluators do not need my private
PDF or learner data.

## 3:25-4:00 - Benchmark and Conclusion

**Screen:** Benchmark table and final slide.

**Narration:**

I compared Qwen thinking enabled and disabled with one warm-up and five measured
runs per mode and workload. Tool Calling validation was one hundred percent in
both modes. Disabling thinking did not clearly improve planning median latency,
but bilingual generation median latency improved from 7.03 to 6.54 seconds in
this sample.

These are client-observed end-to-end measurements, not raw GPU throughput.

LexiPilot demonstrates a useful private Agent pattern: model-based planning,
strict validation, deterministic writes, persistent memory, and dedicated AMD
Radeon inference. Thank you.
