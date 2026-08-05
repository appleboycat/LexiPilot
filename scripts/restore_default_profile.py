#!/usr/bin/env python3
"""Restore the default LexiPilot/vocab_trainer profile from a backup."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROFILE_DIR = Path(".vocab_progress/default")


def restore_default_profile(backup: Path, *, yes: bool = False) -> Path:
    backup = backup.expanduser()
    if not backup.exists() or not backup.is_dir():
        raise SystemExit(f"Backup directory not found: {backup}")
    if not (backup / "progress.json").exists():
        raise SystemExit(f"Backup does not look like a default profile: {backup}")
    if not yes:
        answer = input(f"Restore {backup} over {PROFILE_DIR}? Type yes to continue: ").strip().lower()
        if answer != "yes":
            raise SystemExit("Restore cancelled.")
    tmp = PROFILE_DIR.with_name("default.restore_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(backup, tmp)
    if PROFILE_DIR.exists():
        shutil.rmtree(PROFILE_DIR)
    tmp.replace(PROFILE_DIR)
    return PROFILE_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore .vocab_progress/default from a backup.")
    parser.add_argument("--backup", required=True, help="Backup directory, for example .vocab_progress_backups/default_YYYYMMDD_HHMMSS")
    parser.add_argument("--yes", action="store_true", help="Restore without interactive confirmation")
    args = parser.parse_args()
    path = restore_default_profile(Path(args.backup), yes=args.yes)
    print(f"Restored default profile: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
