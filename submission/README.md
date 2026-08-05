# LexiPilot - AMD Radeon Hackathon 2026 Submission

**Track:** Track 2 - Development & Local Deployment of Private AI Agents  
**Project:** LexiPilot - A Private Adaptive Vocabulary Learning Agent  
**Team:** `sheepdog`

**Members:** `appleboycat`, `du-du-lu`

**Source:** https://github.com/appleboycat/LexiPilot

## Submission Materials

- [Project Specification (Markdown)](LexiPilot_Project_Specification.md)
- [Project Specification (PDF)](LexiPilot_Project_Specification.pdf)
- [Presentation (PPTX)](LexiPilot_Presentation.pptx)
- [Presentation (PDF)](LexiPilot_Presentation.pdf)
- [Architecture source](architecture/lexipilot_architecture.mmd)
- [Architecture SVG](architecture/lexipilot_architecture.svg)
- [Architecture PNG](architecture/lexipilot_architecture.png)
- [Radeon evidence manifest](evidence/evidence_manifest.md)
- [Demo video script](video/VIDEO_SCRIPT.md)
- [Demo commands](video/VIDEO_COMMANDS.md)
- [Demo video link](video/VIDEO_LINK.md)
- [Pull Request body](PR_BODY.md)
- [Release notes](RELEASE_NOTES.md)
- [Final checklist](FINAL_SUBMISSION_CHECKLIST.md)

## Track 2 Requirement Mapping

The official contest README requires the following Track 2 deliverables:

| Official requirement | LexiPilot artifact |
|---|---|
| Project Specification Document | `LexiPilot_Project_Specification.pdf` |
| Application scenarios | Specification, Sections 3 and 4 |
| Agent architecture diagram | `architecture/lexipilot_architecture.svg` |
| Core capabilities | Specification, Sections 4-8 |
| Model and local deployment plan | Specification, Section 9 |
| Radeon inference optimization | Specification, Section 10 |
| Complete project source code | LexiPilot source repository |
| Environment, startup, dependencies | Root `README.md`, `.env.example`, `requirements.txt` |
| 3-5 minute operational demo | `video/VIDEO_LINK.md` (manual recording required) |
| PPT or Poster | `LexiPilot_Presentation.pptx` and `.pdf` |

Official instructions also require a fork of the contest repository, an English
Pull Request, and a title in the form:

```text
Track 2, sheepdog, LexiPilot
```

Source: https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/

## Reproduce

The public demo uses a sanitized 40-word vocabulary index and a generated
synthetic profile. It does not require the private source PDF or a real learner
profile.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
python3 scripts/setup_demo_data.py
python3 scripts/smoke_fresh_clone.py
FORCE_COLOR=1 python3 lexipilot.py --demo --env-file .env --debug
```

For a model-free deterministic demonstration:

```bash
python3 lexipilot.py --demo --deterministic --no-color
```

## Evidence Status

Code, automated tests, sanitized benchmark reports, and endpoint verification
tools are reproducible from this repository. Radeon Cloud console screenshots,
GPU model, ROCm version, vLLM version, utilization, and the final video must be
captured manually from the actual deployment. No hardware value is inferred or
fabricated in these materials.
