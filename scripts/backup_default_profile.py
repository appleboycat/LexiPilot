#!/usr/bin/env python3
"""Back up the primary LexiPilot learner profile."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


PROFILE_ROOT = Path(".vocab_progress")
BACKUP_ROOT = Path(".vocab_progress_backups")
DEFAULT_PROFILE_NAME = "toefl2026"


def backup_default_profile(profile: str = DEFAULT_PROFILE_NAME) -> Path:
    profile_dir = PROFILE_ROOT / profile
    if not profile_dir.exists():
        raise SystemExit(f"Profile does not exist: {profile_dir}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUP_ROOT / f"{profile}_{timestamp}"
    if target.exists():
        raise SystemExit(f"Backup already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(profile_dir, target)
    return target


def main() -> int:
    path = backup_default_profile()
    print(f"Backup created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
