# Final Submission Checklist

## Automatically Completed

- [x] Complete LexiPilot source code is present.
- [x] Hybrid Agent trust boundaries are implemented and documented.
- [x] Public sample vocabulary index is present.
- [x] Synthetic demo profile generator is present.
- [x] Model-free smoke and fresh-clone smoke scripts are present.
- [x] Project Specification Markdown is present.
- [x] Project Specification PDF generated and inspected (8 pages).
- [x] Mermaid architecture source is present.
- [x] Architecture SVG and PNG generated and inspected.
- [x] Presentation PPTX and PDF generated and inspected (8 slides).
- [x] Radeon evidence manifest is present.
- [x] Four-minute English video script is present.
- [x] Exact video commands are present.
- [x] PR body and title template are present.
- [x] Root README submission entry is complete.
- [x] Official repository submission directory is structurally complete.
- [x] Full tests and fresh-clone validation recorded in release notes.
- [x] Credential/private-path scan completed.
- [x] Markdown relative links checked.
- [x] Dedicated endpoint completion and structured Tool Calling verified.
- [x] Real benchmark metadata checked and kept separate from mock output.
- [x] Team name and both member names populated.

## Manual Actions Required

### 1. Capture Real Radeon Evidence

Follow `submission/evidence/evidence_manifest.md`. Capture only real output.
Redact API keys, tokens, endpoint URLs, account IDs, private IPs, and unrelated
personal information.

### 2. Confirm the Demo Video

The current English-narrated candidate is `2:10`, compressed to `4.8 MB`, and
linked from the submission documents. Its duration is shorter than the
officially recommended 3-5 minutes, but the current candidate is being used for
submission.

Use:

- `submission/video/VIDEO_SCRIPT.md`
- `submission/video/VIDEO_COMMANDS.md`

Before opening the Pull Request, verify that the public GitHub video link opens
without authentication and plays or downloads successfully.

### 3. Fork, Push, and Open the Official Pull Request

Fork:

```text
https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/
```

Copy `official_submission/LexiPilot/` into the fork using a non-conflicting
directory selected after checking the latest fork tree. The official repository
currently does not publish a fixed participant-submission directory template.

Use the title:

```text
Track 2, sheepdog, LexiPilot
```

Use `submission/PR_BODY.md` as the body. Verify all links from the branch before
opening the PR.

### 4. Confirm and Publish the Final Source Revision

Only after placeholders and video links are complete:

```bash
git add .
git commit -m "Prepare Radeon Hackathon Track 2 submission"
git tag -a submission-v1.0 -m "Radeon Hackathon Track 2 submission"
git push origin main
git push origin submission-v1.0
```

These commands are recommendations only. They are not executed by the
submission preparation process.

## Final Human Review

- [x] Current demo video is linked from all submission documents.
- [ ] Video shows actual Radeon endpoint inference.
- [ ] No credential or private endpoint appears in video or screenshots.
- [ ] GPU model and software versions come from actual evidence.
- [ ] README, Specification, PPT, benchmark JSON, and PR use consistent values.
- [ ] No mock result is presented as hardware performance.
- [ ] No private PDF or real learner progress is staged.
- [ ] Video links are accessible to judges.
- [x] Team and member placeholders are fully replaced.
