#!/usr/bin/env python3
"""Restore the primary LexiPilot learner profile from a backup."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROFILE_ROOT = Path(".vocab_progress")
DEFAULT_PROFILE_NAME = "toefl2026"


def restore_default_profile(
    backup: Path,
    *,
    yes: bool = False,
    profile: str = DEFAULT_PROFILE_NAME,
) -> Path:
    backup = backup.expanduser()
    profile_dir = PROFILE_ROOT / profile
    if not backup.exists() or not backup.is_dir():
        raise SystemExit(f"Backup directory not found: {backup}")
    if not (backup / "progress.json").exists():
        raise SystemExit(f"Backup does not look like a learner profile: {backup}")
    if not yes:
        answer = input(f"Restore {backup} over {profile_dir}? Type yes to continue: ").strip().lower()
        if answer != "yes":
            raise SystemExit("Restore cancelled.")
    tmp = profile_dir.with_name(f"{profile}.restore_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(backup, tmp)
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    tmp.replace(profile_dir)
    return profile_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore a LexiPilot profile from a backup.")
    parser.add_argument(
        "--backup",
        required=True,
        help="Backup directory, for example .vocab_progress_backups/toefl2026_YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--yes", action="store_true", help="Restore without interactive confirmation")
    args = parser.parse_args()
    path = restore_default_profile(Path(args.backup), yes=args.yes, profile=args.profile)
    print(f"Restored profile: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
