# Final Submission Checklist

## Automatically Completed

- [x] Complete LexiPilot source code is present.
- [x] Hybrid Agent trust boundaries are implemented and documented.
- [x] Public sample vocabulary index is present.
- [x] Synthetic demo profile generator is present.
- [x] Model-free smoke and fresh-clone smoke scripts are present.
- [x] Project Specification Markdown is present.
- [ ] Project Specification PDF generated and inspected.
- [x] Mermaid architecture source is present.
- [ ] Architecture SVG and PNG generated and inspected.
- [ ] Presentation PPTX and PDF generated and inspected.
- [x] Radeon evidence manifest is present.
- [x] Four-minute English video script is present.
- [x] Exact video commands are present.
- [x] PR body and title template are present.
- [ ] Root README submission entry is complete.
- [ ] Official repository submission directory is complete.
- [ ] Full tests and fresh-clone validation recorded in release notes.
- [ ] Credential/private-path scan completed.
- [ ] Markdown links checked.

## Manual Actions Required

### 1. Confirm Participant or Team Name

Replace `<PARTICIPANT_OR_TEAM_NAME>` in:

- `submission/README.md`
- `submission/LexiPilot_Project_Specification.md`
- `submission/PR_BODY.md`
- `submission/video/VIDEO_SCRIPT.md`
- `official_submission/LexiPilot/README.md`

If no team name was supplied during registration, the official contest README
requires using the participant's own name.

### 2. Capture Real Radeon Evidence

Follow `submission/evidence/evidence_manifest.md`. Capture only real output.
Redact API keys, tokens, endpoint URLs, account IDs, private IPs, and unrelated
personal information.

### 3. Record and Upload the 3-5 Minute Video

Use:

- `submission/video/VIDEO_SCRIPT.md`
- `submission/video/VIDEO_COMMANDS.md`

After upload, update:

- `submission/video/VIDEO_LINK.md`
- `submission/PR_BODY.md`
- `official_submission/LexiPilot/demo_video.md`
- `official_submission/LexiPilot/README.md`

### 4. Fork, Push, and Open the Official Pull Request

Fork:

```text
https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/
```

Copy `official_submission/LexiPilot/` into the fork using a non-conflicting
directory selected after checking the latest fork tree. The official repository
currently does not publish a fixed participant-submission directory template.

Use the title:

```text
Track 2, <PARTICIPANT_OR_TEAM_NAME>, LexiPilot
```

Use `submission/PR_BODY.md` as the body. Verify all links from the branch before
opening the PR.

### 5. Confirm and Publish the Final Source Revision

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

- [ ] Video duration is 3-5 minutes.
- [ ] Video shows actual Radeon endpoint inference.
- [ ] No credential or private endpoint appears in video or screenshots.
- [ ] GPU model and software versions come from actual evidence.
- [ ] README, Specification, PPT, benchmark JSON, and PR use consistent values.
- [ ] No mock result is presented as hardware performance.
- [ ] No private PDF or real learner progress is staged.
- [ ] Video links are accessible to judges.
- [ ] Participant/team placeholder is fully replaced.
