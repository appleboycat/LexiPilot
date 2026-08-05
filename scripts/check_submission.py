#!/usr/bin/env python3
"""Validate the public LexiPilot submission package without exposing secrets."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "submission/README.md",
    "submission/LexiPilot_Project_Specification.md",
    "submission/LexiPilot_Project_Specification.pdf",
    "submission/LexiPilot_Presentation.pptx",
    "submission/LexiPilot_Presentation.pdf",
    "submission/architecture/lexipilot_architecture.mmd",
    "submission/architecture/lexipilot_architecture.svg",
    "submission/architecture/lexipilot_architecture.png",
    "submission/evidence/README.md",
    "submission/evidence/evidence_manifest.md",
    "submission/video/VIDEO_SCRIPT.md",
    "submission/video/VIDEO_COMMANDS.md",
    "submission/video/VIDEO_LINK.md",
    "submission/PR_BODY.md",
    "submission/RELEASE_NOTES.md",
    "submission/FINAL_SUBMISSION_CHECKLIST.md",
    "official_submission/LexiPilot/README.md",
    "official_submission/LexiPilot/project_specification.pdf",
    "official_submission/LexiPilot/presentation.pdf",
    "official_submission/LexiPilot/demo_video.md",
)

SCAN_ROOTS = (
    "README.md",
    ".env.example",
    "docs",
    "examples",
    "official_submission",
    "scripts",
    "submission",
    "tests",
    "console_theme.py",
    "lexipilot.py",
    "lexipilot_core.py",
    "lexipilot_tools.py",
    "vocab_trainer.py",
)

TEXT_SUFFIXES = {".md", ".mmd", ".py", ".json", ".txt", ".example", ".sh"}
IGNORED_LINK_PREFIXES = ("http://", "https://", "mailto:", "#")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
SECRET_PATTERNS = (
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{12,}", re.I)),
    (
        "assigned API key",
        re.compile(r"RADEON_API_KEY[ \t]*=[ \t]*[^\s#\"']+", re.I),
    ),
    (
        "private endpoint",
        re.compile(r"https?://[^\s)\"']*(?:radeon-global|anruicloud)[^\s)\"']*", re.I),
    ),
    ("private home path", re.compile(r"/home/[A-Za-z0-9._-]+/")),
)


def iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for relative in SCAN_ROOTS:
        path = ROOT / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES
            )
    return sorted(set(files))


def check_required_files(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required file: {relative}")


def check_binary_formats(errors: list[str]) -> None:
    pdfs = (
        "submission/LexiPilot_Project_Specification.pdf",
        "submission/LexiPilot_Presentation.pdf",
        "official_submission/LexiPilot/project_specification.pdf",
        "official_submission/LexiPilot/presentation.pdf",
    )
    for relative in pdfs:
        path = ROOT / relative
        if path.is_file() and not path.read_bytes()[:5] == b"%PDF-":
            errors.append(f"invalid PDF header: {relative}")

    pptx = ROOT / "submission/LexiPilot_Presentation.pptx"
    if pptx.is_file():
        try:
            with zipfile.ZipFile(pptx) as archive:
                slide_count = len(
                    [
                        name
                        for name in archive.namelist()
                        if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
                    ]
                )
            if slide_count != 8:
                errors.append(f"presentation has {slide_count} slides; expected 8")
        except zipfile.BadZipFile:
            errors.append("invalid PPTX archive: submission/LexiPilot_Presentation.pptx")


def check_markdown_links(errors: list[str]) -> int:
    checked = 0
    for path in iter_scan_files():
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(IGNORED_LINK_PREFIXES):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            checked += 1
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"link escapes repository: {path.relative_to(ROOT)}")
                continue
            if not resolved.exists():
                errors.append(
                    f"broken link in {path.relative_to(ROOT)}: {target}"
                )
    return checked


def check_public_text(errors: list[str]) -> int:
    checked = 0
    for path in iter_scan_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 public text file: {path.relative_to(ROOT)}")
            continue
        checked += 1
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"{label} detected in {path.relative_to(ROOT)}")
    return checked


def check_benchmark(errors: list[str]) -> None:
    path = ROOT / "docs/benchmark_results/thinking_benchmark.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"benchmark summary unreadable: {type(exc).__name__}")
        return
    environment = report.get("environment", {})
    if environment.get("mock_data") is not False:
        errors.append("committed benchmark is not marked as real")
    if environment.get("hardware_result") is not True:
        errors.append("committed benchmark is not marked as a hardware result")
    if environment.get("benchmark_complete") is not True:
        errors.append("committed benchmark is incomplete")


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_binary_formats(errors)
    link_count = check_markdown_links(errors)
    text_count = check_public_text(errors)
    check_benchmark(errors)

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        print(f"FAIL submission validation ({len(errors)} issue(s))")
        return 1

    print(f"PASS required submission files ({len(REQUIRED_FILES)})")
    print("PASS PDF and PPTX structure")
    print(f"PASS Markdown relative links ({link_count})")
    print(f"PASS public text privacy scan ({text_count} files)")
    print("PASS real benchmark metadata")
    print("PASS LexiPilot submission validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
